"""커스텀 제스처 공통 모듈: 랜드마크 -> 특징 벡터 변환

수집(collect_gestures.py), 학습(train_gestures.py), 추론(custom_gesture_webcam.py)이
모두 같은 변환을 써야 하므로 한 곳에 둔다.
"""
import numpy as np

NUM_LANDMARKS = 21
CSV_HEADER = ["label", "handedness"] + [f"{a}{i}" for i in range(NUM_LANDMARKS) for a in "xyz"]


def landmarks_to_row(hand_landmarks, width, height):
    """정규화 좌표(0~1)를 픽셀 비율로 바꿔 [x0,y0,z0, ..., x20,y20,z20] 리스트로 반환.
    (가로/세로 비율 왜곡을 없애기 위해 x, z는 width, y는 height를 곱한다)"""
    row = []
    for lm in hand_landmarks:
        row += [lm.x * width, lm.y * height, lm.z * width]
    return row


def normalize(raw, handedness):
    """raw: 길이 63 벡터. 손목 기준 이동 + 크기 정규화 + 왼손은 좌우 반전(오른손 기준으로 통일)."""
    pts = np.asarray(raw, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
    pts = pts - pts[0]                      # 손목(0번)을 원점으로
    if handedness == "Left":
        pts[:, 0] = -pts[:, 0]              # 왼손 -> 오른손 모양으로 미러링
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts = pts / scale                   # 손 크기/카메라 거리 영향 제거
    return pts.flatten()
