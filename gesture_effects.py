"""커스텀 제스처 이모지 이펙트 (웹캠 + Hand Landmarker + 학습한 분류기)

학습한 모델(models/custom_gesture.joblib)의 제스처에 따라 화면 이펙트를 띄운다.
    okay     -> 👌 이모지가 손 위에 '뿅' 하고 나타남 (제스처를 유지하는 동안 표시)
    heart    -> ❤️ 하트들이 손에서 피어올라 화면 위로 둥둥 떠다님
    fuck you -> 🖕 이모지가 화면 가운데로 갑자기 튀어나옴 (흔들림 + 플래시)

실행: python gesture_effects.py    (종료: q 또는 ESC, l: 랜드마크 표시 on/off)
"""
import math
import os
import random
import threading
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

from gesture_features import landmarks_to_row, normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
GESTURE_MODEL_PATH = os.path.join(BASE_DIR, "models", "custom_gesture.joblib")
EMOJI_FONT = r"C:\Windows\Fonts\seguiemj.ttf"

CAMERA_INDEX = 0
THRESHOLD = 0.7      # 이 확률 이상일 때만 제스처로 인정
ON_FRAMES = 3        # 연속 N프레임 감지되어야 이펙트 시작 (깜빡임 방지)
OFF_FRAMES = 6       # 연속 N프레임 사라져야 이펙트 종료

# 학습 라벨 이름 -> 이펙트 종류. 라벨은 소문자/공백·_·- 제거 후 비교한다.
LABEL_TO_EFFECT = {
    "ok": "ok", "okay": "ok",
    "heart": "heart", "love": "heart",
    "fuckyou": "fu", "fu": "fu", "middlefinger": "fu",
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# --------------------------------------------------------------- 이모지 그리기
def render_emoji(char, size=256):
    """컬러 이모지를 BGRA numpy 이미지로 렌더링 (여백 제거)."""
    font = ImageFont.truetype(EMOJI_FONT, 109)  # Segoe UI Emoji는 109px에서 가장 선명
    l, t, r, b = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), char, font=font, embedded_color=True)
    img = Image.new("RGBA", (r - l, b - t), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((-l, -t), char, font=font, embedded_color=True)
    img = img.crop(img.getbbox())
    scale = size / max(img.size)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def overlay(frame, sprite, cx, cy, size, alpha=1.0, angle=0.0):
    """BGRA 스프라이트를 (cx, cy) 중심에 size 크기로 알파 합성."""
    if size < 4 or alpha <= 0.01:
        return
    sh, sw = sprite.shape[:2]
    s = size / max(sh, sw)
    img = cv2.resize(sprite, (max(1, int(sw * s)), max(1, int(sh * s))), interpolation=cv2.INTER_LINEAR)
    if angle:
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    h, w = img.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    if fx0 >= fx1 or fy0 >= fy1:
        return
    crop = img[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    a = (crop[:, :, 3:4].astype(np.float32) / 255.0) * alpha
    roi = frame[fy0:fy1, fx0:fx1]
    roi[:] = (crop[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)


def ease_out_back(t):
    """0->1 로 가면서 살짝 튕기는(오버슈트) 이징."""
    c1, c3 = 1.70158, 2.70158
    t = min(max(t, 0.0), 1.0)
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


# --------------------------------------------------------------- 이펙트
class OkEffect:
    """👌: 손 위에 팝업, 제스처 유지 중 표시, 놓으면 페이드아웃."""

    def __init__(self):
        self.sprite = render_emoji("\U0001F44C")
        self.active = False
        self.t_on = 0.0
        self.t_off = 0.0
        self.pos = (0, 0)

    def start(self, now):
        self.active, self.t_on = True, now

    def stop(self, now):
        self.active, self.t_off = False, now

    def draw(self, frame, now, anchor):
        if anchor is not None:
            self.pos = anchor
        if self.active:
            k = ease_out_back((now - self.t_on) / 0.35)
            alpha = 1.0
        else:
            fade = (now - self.t_off) / 0.3
            if fade >= 1 or self.t_on == 0:
                return
            k, alpha = 1.0, 1 - fade
        bob = math.sin(now * 6) * 6
        size = int(frame.shape[0] * 0.28 * k)
        overlay(frame, self.sprite, self.pos[0], self.pos[1] - size * 0.6 + bob, size, alpha,
                angle=math.sin(now * 4) * 8)


class HeartEffect:
    """❤️: 제스처 유지 중 하트가 계속 생성되어 위로 흔들리며 떠오름."""

    def __init__(self):
        self.sprite = render_emoji("\u2764\uFE0F")
        self.active = False
        self.hearts = []  # dict(x, y, vy, size, phase, born, life)
        self.last_spawn = 0.0
        self.anchor = None

    def start(self, now):
        self.active = True

    def stop(self, now):
        self.active = False

    def draw(self, frame, now, anchor):
        h, w = frame.shape[:2]
        if anchor is not None:
            self.anchor = anchor
        if self.active and now - self.last_spawn > 0.08:
            self.last_spawn = now
            ax, ay = self.anchor if self.anchor else (w / 2, h * 0.8)
            for _ in range(2):
                self.hearts.append(dict(
                    x=ax + random.uniform(-w * 0.25, w * 0.25),
                    y=ay + random.uniform(-20, 40),
                    vy=random.uniform(90, 200),
                    size=random.uniform(h * 0.06, h * 0.16),
                    phase=random.uniform(0, math.tau),
                    sway=random.uniform(15, 45),
                    born=now,
                    life=random.uniform(2.0, 3.5),
                ))
        alive = []
        for p in self.hearts:
            age = now - p["born"]
            if age > p["life"]:
                continue
            alive.append(p)
            y = p["y"] - p["vy"] * age
            x = p["x"] + math.sin(p["phase"] + age * 3) * p["sway"]
            grow = min(age / 0.25, 1.0)                       # 처음엔 작게 시작
            alpha = min(1.0, (p["life"] - age) / 0.8)          # 끝날 때 서서히 사라짐
            pulse = 1 + 0.08 * math.sin(age * 10 + p["phase"])  # 두근두근
            overlay(frame, self.sprite, x, y, int(p["size"] * grow * pulse), alpha,
                    angle=math.sin(p["phase"] + age * 2) * 15)
        self.hearts = alive[-200:]


class FuEffect:
    """🖕: 화면 중앙으로 갑자기 튀어나옴 (줌인 + 흔들림 + 플래시), 놓으면 쏙 들어감."""

    def __init__(self):
        self.sprite = render_emoji("\U0001F595", size=512)
        self.active = False
        self.t_on = 0.0
        self.t_off = 0.0

    def start(self, now):
        self.active, self.t_on = True, now

    def stop(self, now):
        self.active, self.t_off = False, now

    def draw(self, frame, now, anchor):
        h, w = frame.shape[:2]
        if self.active:
            t = now - self.t_on
            if t < 0.22:
                k = ease_out_back(t / 0.22) * 1.25  # 순간적으로 크게 튀어나옴
            else:
                k = 1.0 + 0.25 * math.exp(-(t - 0.22) * 8)  # 원래 크기로 복귀
            alpha = 1.0
            # 플래시
            if t < 0.15:
                white = np.full_like(frame, 255)
                cv2.addWeighted(white, 0.6 * (1 - t / 0.15), frame, 1 - 0.6 * (1 - t / 0.15), 0, frame)
            shake = max(0.0, 1 - t / 0.6) * 25 + 3
        else:
            t = now - self.t_off
            if t >= 0.25 or self.t_on == 0:
                return
            k = 1 - t / 0.25
            alpha = k
            shake = 0
        dx = random.uniform(-shake, shake)
        dy = random.uniform(-shake, shake)
        size = int(h * 0.85 * k)
        overlay(frame, self.sprite, w / 2 + dx, h / 2 + dy, size, alpha, angle=random.uniform(-shake, shake) * 0.3)


# --------------------------------------------------------------- 메인
_latest = None
_lock = threading.Lock()


def on_result(result, image, timestamp_ms):
    global _latest
    with _lock:
        _latest = (result, image.width, image.height)


def effect_key(label):
    return LABEL_TO_EFFECT.get(label.lower().replace(" ", "").replace("_", "").replace("-", ""))


def main():
    bundle = joblib.load(GESTURE_MODEL_PATH)
    clf = bundle["model"]
    mapping = {lab: effect_key(lab) for lab in bundle["labels"]}
    print("라벨 -> 이펙트:", mapping)
    if not any(mapping.values()):
        print("경고: 이펙트에 연결된 라벨이 없습니다. LABEL_TO_EFFECT 를 수정하세요.")

    effects = {"ok": OkEffect(), "heart": HeartEffect(), "fu": FuEffect()}
    on_count = {k: 0 for k in effects}
    off_count = {k: 0 for k in effects}

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    show_landmarks = False
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        start = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드 (수집 때와 동일)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                    int((time.monotonic() - start) * 1000))
            now = time.monotonic()

            with _lock:
                latest = _latest
            detected = {}  # effect -> 손 위치(앵커)
            label_text = []
            if latest is not None:
                result, rw, rh = latest
                for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    hand = handedness[0].category_name
                    proba = clf.predict_proba([normalize(landmarks_to_row(hand_landmarks, rw, rh), hand)])[0]
                    best = proba.argmax()
                    label = str(clf.classes_[best])
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                    if show_landmarks:
                        for a, b in HAND_CONNECTIONS:
                            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                    if proba[best] >= THRESHOLD:
                        label_text.append(f"{label} {proba[best]:.2f}")
                        key = mapping.get(label)
                        if key:
                            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                            detected[key] = (sum(xs) / len(xs), min(ys))  # 손 위쪽 중앙

            # 디바운스 후 이펙트 on/off
            for key, eff in effects.items():
                if key in detected:
                    on_count[key] += 1
                    off_count[key] = 0
                    if not eff.active and on_count[key] >= ON_FRAMES:
                        eff.start(now)
                else:
                    off_count[key] += 1
                    on_count[key] = 0
                    if eff.active and off_count[key] >= OFF_FRAMES:
                        eff.stop(now)

            # 하트(배경) -> 오케이 -> 퍽유(최상단) 순으로 그림
            for key in ("heart", "ok", "fu"):
                effects[key].draw(frame, now, detected.get(key))

            if label_text:
                cv2.putText(frame, " | ".join(label_text), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Gesture Effects", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("l"):
                show_landmarks = not show_landmarks

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
