"""커스텀 제스처 실시간 추론 (웹캠 + Hand Landmarker + 학습한 분류기)

사용 예:
    python custom_gesture_webcam.py
    python custom_gesture_webcam.py --model models/custom_gesture.joblib --threshold 0.7

확률이 threshold 미만이면 "Unknown" 으로 표시한다.  종료: q 또는 ESC
"""
import argparse
import os
import threading
import time

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gesture_features import landmarks_to_row, normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

_latest = None  # (result, width, height)
_lock = threading.Lock()


def on_result(result: vision.HandLandmarkerResult, image: mp.Image, timestamp_ms: int):
    global _latest
    with _lock:
        _latest = (result, image.width, image.height)


def main():
    parser = argparse.ArgumentParser(description="커스텀 제스처 실시간 추론")
    parser.add_argument("--model", default=os.path.join(BASE_DIR, "models", "custom_gesture.joblib"))
    parser.add_argument("--threshold", type=float, default=0.6, help="이 확률 미만이면 Unknown")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hands", type=int, default=2)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["model"]
    print("라벨:", bundle["labels"])

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=args.hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({args.camera})을 열 수 없습니다.")

    prev = time.time()
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        start = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드 (수집 코드와 동일하게)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, int((time.monotonic() - start) * 1000))

            with _lock:
                latest = _latest
            if latest is not None:
                result, rw, rh = latest
                for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    hand = handedness[0].category_name
                    feat = normalize(landmarks_to_row(hand_landmarks, rw, rh), hand)
                    proba = clf.predict_proba([feat])[0]
                    best = proba.argmax()
                    name = clf.classes_[best] if proba[best] >= args.threshold else "Unknown"

                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                    for p in pts:
                        cv2.circle(frame, p, 4, (0, 0, 255), -1)
                    x0 = min(p[0] for p in pts)
                    y0 = min(p[1] for p in pts) - 10
                    cv2.putText(frame, f"{hand}: {name} ({proba[best]:.2f})", (x0, max(y0, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Custom Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
