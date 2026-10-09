"""Gesture Studio - 커스텀 제스처 수집 / 학습 / 실시간 추론을 한 화면에서 하는 GUI 앱

실행: python gesture_studio.py

1. [라벨] 제스처 이름을 입력하고 '추가'
2. [수집] 라벨을 선택하고 '녹화 시작' (또는 SPACE) - 목표 개수에 도달하면 자동 정지
3. [학습] '학습 시작' - 결과는 아래 로그 창에 표시, 끝나면 모델 자동 로드
4. [추론] '실시간 추론' 체크 - 화면에 제스처 이름과 확률 표시
"""
import csv
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageTk

from gesture_features import CSV_HEADER, landmarks_to_row, normalize
from train_gestures import train

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
DATA_PATH = os.path.join(BASE_DIR, "data", "gestures.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "custom_gesture.joblib")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FONT = ("Malgun Gothic", 10)
FONT_BOLD = ("Malgun Gothic", 11, "bold")


def read_counts():
    counts = {}
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


class GestureStudio:
    def __init__(self, root):
        self.root = root
        root.title("Gesture Studio - 커스텀 제스처 수집/학습/추론")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        self.counts = read_counts()
        self.labels = sorted(self.counts)

        self.recording = False
        self.rec_file = None
        self.rec_writer = None
        self.rec_added = 0
        self.countdown_end = 0.0
        self.training = False
        self.log_queue = queue.Queue()
        self.clf = None
        self.closed = False

        self.cap = None
        self.landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ))
        self.start_time = time.monotonic()
        self.last_ts = -1
        self.prev_frame_time = time.time()

        self.build_ui()
        self.refresh_label_list()
        self.open_camera()
        self.load_model(silent=True)
        self.root.after(10, self.update_frame)
        self.root.after(100, self.drain_log)

    # ------------------------------------------------------------------ UI
    def build_ui(self):
        style = ttk.Style()
        style.configure(".", font=FONT)
        style.configure("Rec.TButton", font=FONT_BOLD)

        main = ttk.Frame(self.root, padding=8)
        main.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # 왼쪽: 영상 + 로그
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.video = ttk.Label(left, anchor="center", background="black")
        self.video.grid(row=0, column=0, sticky="nsew")

        log_frame = ttk.LabelFrame(left, text="로그", padding=4)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=10, font=("Consolas", 9), state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log_text["yscrollcommand"] = sb.set

        # 오른쪽: 컨트롤 패널
        panel = ttk.Frame(main, padding=(10, 0, 0, 0), width=300)
        panel.grid(row=0, column=1, sticky="ns")

        # 카메라
        cam = ttk.LabelFrame(panel, text="카메라", padding=6)
        cam.pack(fill="x", pady=(0, 8))
        ttk.Label(cam, text="번호").pack(side="left")
        self.cam_var = tk.IntVar(value=0)
        ttk.Spinbox(cam, from_=0, to=9, width=4, textvariable=self.cam_var).pack(side="left", padx=4)
        ttk.Button(cam, text="연결", command=self.open_camera).pack(side="left")

        # 라벨
        lab = ttk.LabelFrame(panel, text="① 제스처 라벨", padding=6)
        lab.pack(fill="x", pady=(0, 8))
        self.label_list = tk.Listbox(lab, height=8, font=FONT, exportselection=False)
        self.label_list.pack(fill="x")
        self.label_list.bind("<<ListboxSelect>>", lambda e: self.update_status())
        row = ttk.Frame(lab)
        row.pack(fill="x", pady=(6, 0))
        self.new_label = tk.StringVar()
        entry = ttk.Entry(row, textvariable=self.new_label, width=14)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: self.add_label())
        ttk.Button(row, text="추가", width=5, command=self.add_label).pack(side="left", padx=(4, 0))
        ttk.Button(lab, text="선택 라벨 데이터 삭제", command=self.delete_label).pack(fill="x", pady=(6, 0))

        # 수집
        col = ttk.LabelFrame(panel, text="② 데이터 수집", padding=6)
        col.pack(fill="x", pady=(0, 8))
        r1 = ttk.Frame(col)
        r1.pack(fill="x")
        ttk.Label(r1, text="목표 개수").pack(side="left")
        self.target_var = tk.IntVar(value=300)
        ttk.Spinbox(r1, from_=10, to=5000, increment=50, width=7,
                    textvariable=self.target_var).pack(side="left", padx=4)
        self.countdown_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(r1, text="3초 대기", variable=self.countdown_var).pack(side="left", padx=(6, 0))
        self.rec_btn = ttk.Button(col, text="● 녹화 시작 (SPACE)", style="Rec.TButton",
                                  command=self.toggle_record)
        self.rec_btn.pack(fill="x", pady=(6, 0))
        self.progress = ttk.Progressbar(col, maximum=300)
        self.progress.pack(fill="x", pady=(6, 0))

        # 학습
        tr = ttk.LabelFrame(panel, text="③ 학습", padding=6)
        tr.pack(fill="x", pady=(0, 8))
        self.train_btn = ttk.Button(tr, text="학습 시작", command=self.start_training)
        self.train_btn.pack(fill="x")
        self.acc_var = tk.StringVar(value="모델 없음")
        ttk.Label(tr, textvariable=self.acc_var).pack(anchor="w", pady=(4, 0))

        # 추론
        inf = ttk.LabelFrame(panel, text="④ 실시간 추론", padding=6)
        inf.pack(fill="x", pady=(0, 8))
        self.infer_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(inf, text="추론 켜기", variable=self.infer_var,
                        command=self.on_toggle_infer).pack(anchor="w")
        r2 = ttk.Frame(inf)
        r2.pack(fill="x", pady=(4, 0))
        ttk.Label(r2, text="임계값").pack(side="left")
        self.thr_var = tk.DoubleVar(value=0.6)
        self.thr_text = tk.StringVar(value="0.60")
        ttk.Scale(r2, from_=0.0, to=1.0, variable=self.thr_var,
                  command=lambda v: self.thr_text.set(f"{float(v):.2f}")).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Label(r2, textvariable=self.thr_text, width=4).pack(side="left")
        self.pred_var = tk.StringVar(value="-")
        ttk.Label(inf, textvariable=self.pred_var, font=("Malgun Gothic", 16, "bold"),
                  foreground="#1a73e8").pack(anchor="w", pady=(6, 0))

        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w",
                  padding=(6, 2)).grid(row=1, column=0, sticky="ew")

        self.root.bind("<space>", self.on_space)
        self.root.bind("<Escape>", lambda e: self.on_close())

    # ------------------------------------------------------------ 카메라
    def open_camera(self):
        if self.recording:
            self.stop_record()
        if self.cap is not None:
            self.cap.release()
        idx = self.cam_var.get()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.log(f"웹캠 {idx}번을 열 수 없습니다. 다른 번호를 선택하세요.")
        else:
            self.log(f"웹캠 {idx}번 연결됨")

    def update_frame(self):
        if self.closed:
            return
        frame = None
        if self.cap is not None and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                frame = None
        if frame is not None:
            frame = cv2.flip(frame, 1)  # 거울 모드
            self.process(frame)
            fps = 1.0 / max(time.time() - self.prev_frame_time, 1e-6)
            self.prev_frame_time = time.time()
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video.configure(image=img)
            self.video.image = img
        self.root.after(10, self.update_frame)

    def process(self, frame):
        h, w = frame.shape[:2]
        ts = int((time.monotonic() - self.start_time) * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        in_countdown = self.recording and time.time() < self.countdown_end
        preds = []
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            hand = handedness[0].category_name
            row = landmarks_to_row(hand_landmarks, w, h)
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            color = (0, 0, 255) if self.recording and not in_countdown else (0, 255, 0)
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], color, 2)
            for p in pts:
                cv2.circle(frame, p, 4, (255, 255, 255), -1)

            if self.recording and not in_countdown:
                self.save_sample(hand, row)
            elif self.infer_var.get() and self.clf is not None:
                proba = self.clf.predict_proba([normalize(row, hand)])[0]
                best = proba.argmax()
                name = str(self.clf.classes_[best]) if proba[best] >= self.thr_var.get() else "Unknown"
                preds.append(f"{hand}: {name} ({proba[best]:.2f})")
                x0 = min(p[0] for p in pts)
                y0 = min(p[1] for p in pts) - 10
                cv2.putText(frame, f"{name} {proba[best]:.2f}", (x0, max(y0, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        if self.infer_var.get():
            self.pred_var.set("\n".join(preds) if preds else "손 없음")

        if self.recording:
            if in_countdown:
                n = int(self.countdown_end - time.time()) + 1
                cv2.putText(frame, str(n), (w // 2 - 30, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
            else:
                cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
                cv2.putText(frame, f"REC {self.rec_label} {self.rec_added}/{self.target_var.get()}",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # -------------------------------------------------------------- 라벨
    def current_label(self):
        sel = self.label_list.curselection()
        return self.labels[sel[0]] if sel else None

    def refresh_label_list(self, select=None):
        cur = select or self.current_label()
        self.label_list.delete(0, "end")
        for name in self.labels:
            self.label_list.insert("end", f"{name}  ({self.counts.get(name, 0)})")
        if cur in self.labels:
            i = self.labels.index(cur)
            self.label_list.selection_set(i)
            self.label_list.see(i)
        self.update_status()

    def add_label(self):
        name = self.new_label.get().strip().replace(",", "_")
        if not name:
            return
        if name not in self.labels:
            self.labels.append(name)
        self.new_label.set("")
        self.refresh_label_list(select=name)
        self.root.focus_set()  # SPACE 키가 입력창이 아닌 녹화로 가도록

    def delete_label(self):
        name = self.current_label()
        if name is None or self.recording:
            return
        n = self.counts.get(name, 0)
        if n and not messagebox.askyesno("데이터 삭제", f"'{name}' 라벨의 데이터 {n}개를 삭제할까요?\n(되돌릴 수 없습니다)"):
            return
        if n:
            with open(DATA_PATH, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.reader(f)]
            with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(rows[0])
                w.writerows(r for r in rows[1:] if r and r[0] != name)
        self.labels.remove(name)
        self.counts.pop(name, None)
        self.refresh_label_list()
        self.log(f"'{name}' 라벨 삭제 ({n}개)")

    # -------------------------------------------------------------- 수집
    def on_space(self, event):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry, ttk.Spinbox)):
            return
        self.toggle_record()

    def toggle_record(self):
        if self.recording:
            self.stop_record()
        else:
            self.start_record()

    def start_record(self):
        if self.current_label() is None:
            messagebox.showinfo("라벨 선택", "먼저 수집할 라벨을 추가/선택하세요.")
            return
        if self.training:
            return
        new_file = not os.path.exists(DATA_PATH)
        self.rec_file = open(DATA_PATH, "a", newline="", encoding="utf-8")
        self.rec_writer = csv.writer(self.rec_file)
        if new_file:
            self.rec_writer.writerow(CSV_HEADER)
        self.rec_added = 0
        self.rec_label = self.current_label()
        self.progress["maximum"] = max(self.target_var.get(), 1)
        self.progress["value"] = 0
        self.countdown_end = time.time() + (3 if self.countdown_var.get() else 0)
        self.recording = True
        self.rec_btn.configure(text="■ 녹화 정지 (SPACE)")
        self.root.focus_set()

    def stop_record(self):
        self.recording = False
        if self.rec_file:
            self.rec_file.close()
            self.rec_file = None
        self.rec_btn.configure(text="● 녹화 시작 (SPACE)")
        if self.rec_added:
            self.log(f"'{self.rec_label}' {self.rec_added}개 저장")
        self.refresh_label_list()

    def save_sample(self, hand, row):
        label = self.rec_label
        self.rec_writer.writerow([label, hand] + row)
        self.rec_added += 1
        self.counts[label] = self.counts.get(label, 0) + 1
        self.progress["value"] = self.rec_added
        if self.rec_added % 10 == 0:
            self.refresh_label_list()
        if self.rec_added >= self.target_var.get():
            self.stop_record()

    # -------------------------------------------------------------- 학습
    def start_training(self):
        if self.training:
            return
        if self.recording:
            self.stop_record()
        self.training = True
        self.train_btn.configure(state="disabled", text="학습 중...")
        self.log("=" * 40 + "\n학습 시작")

        def worker():
            try:
                acc = train(DATA_PATH, MODEL_PATH, log=self.log_queue.put)
                self.log_queue.put(("done", acc))
            except Exception as e:  # noqa: BLE001 - GUI에 오류를 보여주기 위함
                self.log_queue.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def drain_log(self):
        if self.closed:
            return
        while not self.log_queue.empty():
            item = self.log_queue.get()
            if isinstance(item, tuple):
                kind, value = item
                self.training = False
                self.train_btn.configure(state="normal", text="학습 시작")
                if kind == "done":
                    self.log(f"학습 완료 - 검증 정확도 {value * 100:.1f}%")
                    self.load_model()
                    self.acc_var.set(f"검증 정확도 {value * 100:.1f}%")
                else:
                    self.log(f"학습 실패: {value}")
                    messagebox.showerror("학습 실패", value)
            else:
                self.log(item)
        self.root.after(100, self.drain_log)

    def load_model(self, silent=False):
        if not os.path.exists(MODEL_PATH):
            return
        bundle = joblib.load(MODEL_PATH)
        self.clf = bundle["model"]
        self.acc_var.set(f"모델 로드됨: {', '.join(bundle['labels'])}")
        if not silent:
            self.log(f"모델 로드: {bundle['labels']}")

    def on_toggle_infer(self):
        if self.infer_var.get() and self.clf is None:
            self.infer_var.set(False)
            messagebox.showinfo("모델 없음", "먼저 학습을 진행하세요.")
        if not self.infer_var.get():
            self.pred_var.set("-")

    # -------------------------------------------------------------- 기타
    def log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def update_status(self):
        total = sum(self.counts.values())
        self.status_var.set(f"선택 라벨: {self.current_label() or '-'}   |   전체 샘플: {total}   |   데이터: {DATA_PATH}")

    def on_close(self):
        self.closed = True
        if self.recording:
            self.stop_record()
        if self.cap is not None:
            self.cap.release()
        self.landmarker.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1180x800")
    GestureStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
