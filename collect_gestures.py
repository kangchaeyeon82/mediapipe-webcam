"""커스텀 제스처 데이터 수집 (웹캠 + Hand Landmarker)

사용 예:
    python collect_gestures.py --labels none rock scissors paper

키 조작:
    1~9     : 수집할 라벨 선택 (--labels 순서대로)
    SPACE   : 녹화 시작/정지 (녹화 중에는 손이 보이는 매 프레임이 저장됨)
    q / ESC : 종료

데이터는 data/gestures.csv 에 계속 이어서(append) 저장된다.
"""
import argparse
import csv
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gesture_features import CSV_HEADER, landmarks_to_row

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
DEFAULT_CSV = os.path.join(BASE_DIR, "data", "gestures.csv")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def count_existing(csv_path):
    counts = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="커스텀 제스처 랜드마크 수집")
    parser.add_argument("--labels", nargs="+", required=True, help="수집할 제스처 이름들 (최대 9개)")
    parser.add_argument("--out", default=DEFAULT_CSV, help="저장할 CSV 경로")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 인덱스")
    parser.add_argument("--hands", type=int, default=1, help="한 프레임에서 저장할 최대 손 개수")
    args = parser.parse_args()
    labels = args.labels[:9]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    counts = count_existing(args.out)
    new_file = not os.path.exists(args.out)
    f = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(CSV_HEADER)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({args.camera})을 열 수 없습니다.")

    current = 0
    recording = False
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        start = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드 (추론 코드와 동일하게)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((time.monotonic() - start) * 1000))

            label = labels[current]
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                hand = handedness[0].category_name
                if recording:
                    writer.writerow([label, hand] + landmarks_to_row(hand_landmarks, w, h))
                    counts[label] = counts.get(label, 0) + 1

                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for p in pts:
                    cv2.circle(frame, p, 3, (0, 0, 255), -1)

            # 상태 표시
            color = (0, 0, 255) if recording else (200, 200, 200)
            status = "REC" if recording else "PAUSED"
            cv2.putText(frame, f"[{status}] label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if recording:
                cv2.circle(frame, (w - 25, 25), 10, (0, 0, 255), -1)
            for i, name in enumerate(labels):
                mark = ">" if i == current else " "
                cv2.putText(frame, f"{mark}{i + 1}. {name}: {counts.get(name, 0)}", (10, 65 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(frame, "1-9: label  SPACE: rec on/off  q: quit", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("Collect Gestures", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = not recording
                f.flush()
            if ord("1") <= key <= ord("9") and key - ord("1") < len(labels):
                current = key - ord("1")
                recording = False
                f.flush()

    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print("저장 완료:", os.path.abspath(args.out))
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
