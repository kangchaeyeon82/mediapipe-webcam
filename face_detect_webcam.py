"""MediaPipe Face Detector - 웹캠 실시간 얼굴 검출 (LIVE_STREAM 모드)

BlazeFace(short-range) 모델: 얼굴 바운딩 박스 + 6개 키포인트(눈, 코, 입, 귀)
실행: python face_detect_webcam.py   (종료: q 또는 ESC)
"""
import os
import time
import threading

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blaze_face_short_range.tflite")
CAMERA_INDEX = 0

_latest_result = None
_lock = threading.Lock()


def on_result(result: vision.FaceDetectorResult, image: mp.Image, timestamp_ms: int):
    global _latest_result
    with _lock:
        _latest_result = result


def draw_result(frame, result):
    h, w = frame.shape[:2]
    for det in result.detections:
        box = det.bounding_box  # 픽셀 좌표
        x, y = box.origin_x, box.origin_y
        cv2.rectangle(frame, (x, y), (x + box.width, y + box.height), (0, 255, 0), 2)

        score = det.categories[0].score if det.categories else 0.0
        cv2.putText(frame, f"face {score:.2f}", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for kp in det.keypoints:  # 정규화 좌표 (0~1)
            cv2.circle(frame, (int(kp.x * w), int(kp.y * h)), 4, (0, 0, 255), -1)


def main():
    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        min_detection_confidence=0.5,
        min_suppression_threshold=0.3,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    prev = time.time()
    with vision.FaceDetector.create_from_options(options) as detector:
        start = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            detector.detect_async(mp_image, timestamp_ms)

            with _lock:
                result = _latest_result
            if result is not None:
                draw_result(frame, result)
                cv2.putText(frame, f"Faces: {len(result.detections)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("MediaPipe Face Detector", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
