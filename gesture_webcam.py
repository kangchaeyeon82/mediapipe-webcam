"""MediaPipe Gesture Recognizer - 웹캠 실시간 손 제스처 인식 (LIVE_STREAM 모드)

인식 제스처: None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down,
            Thumb_Up, Victory, ILoveYou
실행: python gesture_webcam.py   (종료: q 또는 ESC)
"""
import os
import time
import threading

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task")
CAMERA_INDEX = 0
NUM_HANDS = 2

# 21개 랜드마크 연결 (손목-손가락 뼈대)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),     # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),   # 약지
    (13, 17), (17, 18), (18, 19), (19, 20),  # 새끼
    (0, 17),
]

_latest_result = None
_lock = threading.Lock()


def on_result(result: vision.GestureRecognizerResult, image: mp.Image, timestamp_ms: int):
    global _latest_result
    with _lock:
        _latest_result = result


def draw_result(frame, result):
    h, w = frame.shape[:2]
    for i, hand_landmarks in enumerate(result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)

        # 좌우 + 제스처 이름/점수 표시 (화면을 거울 모드로 뒤집었으므로 라벨 그대로 사용)
        hand = result.handedness[i][0].category_name if result.handedness else ""
        if result.gestures and result.gestures[i]:
            g = result.gestures[i][0]
            text = f"{hand}: {g.category_name} ({g.score:.2f})"
        else:
            text = hand
        x0 = min(p[0] for p in pts)
        y0 = min(p[1] for p in pts) - 10
        cv2.putText(frame, text, (x0, max(y0, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


def main():
    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    prev = time.time()
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        start = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            recognizer.recognize_async(mp_image, timestamp_ms)

            with _lock:
                result = _latest_result
            if result is not None:
                draw_result(frame, result)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("MediaPipe Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
