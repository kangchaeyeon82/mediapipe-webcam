"""MediaPipe Face Landmarker - 웹캠 실시간 얼굴 랜드마크 검출 (LIVE_STREAM 모드)

478개 얼굴 랜드마크(홍채 포함) + 블렌드쉐이프(표정 점수) 표시
실행: python face_webcam.py   (종료: q 또는 ESC, m: 메시 표시 on/off)
"""
import os
import time
import threading

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
CAMERA_INDEX = 0
NUM_FACES = 1
TOP_BLENDSHAPES = 5

FLC = vision.FaceLandmarksConnections
CONTOUR_STYLE = [
    (FLC.FACE_LANDMARKS_FACE_OVAL, (220, 220, 220)),
    (FLC.FACE_LANDMARKS_LEFT_EYE, (0, 255, 0)),
    (FLC.FACE_LANDMARKS_RIGHT_EYE, (0, 255, 0)),
    (FLC.FACE_LANDMARKS_LEFT_EYEBROW, (0, 200, 255)),
    (FLC.FACE_LANDMARKS_RIGHT_EYEBROW, (0, 200, 255)),
    (FLC.FACE_LANDMARKS_LEFT_IRIS, (255, 128, 0)),
    (FLC.FACE_LANDMARKS_RIGHT_IRIS, (255, 128, 0)),
    (FLC.FACE_LANDMARKS_LIPS, (0, 0, 255)),
    (FLC.FACE_LANDMARKS_NOSE, (255, 0, 255)),
]

_latest_result = None
_lock = threading.Lock()


def on_result(result: vision.FaceLandmarkerResult, image: mp.Image, timestamp_ms: int):
    global _latest_result
    with _lock:
        _latest_result = result


def draw_result(frame, result, show_mesh):
    h, w = frame.shape[:2]
    for face_landmarks in result.face_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
        if show_mesh:
            for c in FLC.FACE_LANDMARKS_TESSELATION:
                cv2.line(frame, pts[c.start], pts[c.end], (90, 90, 90), 1)
        for connections, color in CONTOUR_STYLE:
            for c in connections:
                cv2.line(frame, pts[c.start], pts[c.end], color, 1)

    # 첫 번째 얼굴의 상위 블렌드쉐이프(표정) 점수
    if result.face_blendshapes:
        top = sorted(result.face_blendshapes[0], key=lambda c: c.score, reverse=True)
        for i, c in enumerate(top[:TOP_BLENDSHAPES]):
            y = 60 + i * 24
            cv2.rectangle(frame, (10, y - 14), (10 + int(c.score * 150), y + 4), (0, 180, 255), -1)
            cv2.putText(frame, f"{c.category_name} {c.score:.2f}", (165, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=NUM_FACES,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    show_mesh = True
    prev = time.time()
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        start = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            with _lock:
                result = _latest_result
            if result is not None:
                draw_result(frame, result, show_mesh)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("MediaPipe Face Landmarker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_mesh = not show_mesh

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
