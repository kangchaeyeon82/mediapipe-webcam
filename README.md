# MediaPipe Webcam Examples

[MediaPipe Tasks](https://developers.google.com/edge/mediapipe/solutions/guide) Vision 모델 4종을 **웹캠으로 실시간 실행**하는 Python 예제 모음입니다.
모든 예제는 `LIVE_STREAM` 모드(비동기 콜백)로 동작하며, OpenCV로 결과를 화면에 그립니다.

| 스크립트 | 태스크 | 모델 파일 | 출력 |
|---|---|---|---|
| `hand_webcam.py` | [Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) | `hand_landmarker.task` | 손 21개 랜드마크, 좌/우 손 구분 (최대 2손) |
| `gesture_webcam.py` | [Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer) | `gesture_recognizer.task` | 손 랜드마크 + 제스처 7종 분류 |
| `face_webcam.py` | [Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) | `face_landmarker.task` | 얼굴 478개 랜드마크(홍채 포함) + 블렌드쉐이프(표정 점수) |
| `face_detect_webcam.py` | [Face Detector](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector) | `blaze_face_short_range.tflite` | 얼굴 바운딩 박스 + 6개 키포인트 |

## 환경

- Windows 11, Python 3.12
- `mediapipe` 1.1.0, `opencv-python` 5.0

## 설치

```bash
pip install -r requirements.txt
```

모델 파일은 저장소에 포함되어 있습니다 (아래 [모델 출처](#모델-출처) 참고).

## 실행

```bash
python hand_webcam.py          # 손 랜드마크
python gesture_webcam.py       # 제스처 인식
python face_webcam.py          # 얼굴 랜드마크 (m: 메시 on/off)
python face_detect_webcam.py   # 얼굴 검출
```

- 공통 종료 키: `q` 또는 `ESC`
- 화면은 거울 모드(좌우 반전)로 표시됩니다.
- 웹캠이 열리지 않으면 각 스크립트 상단의 `CAMERA_INDEX`를 `1` 등으로 바꿔보세요.
- 실행 시 출력되는 TensorFlow Lite / XNNPACK 경고 로그는 정상입니다.

## 예제별 설명

### 1. Hand Landmarker — `hand_webcam.py`
- 손 21개 키포인트(손목, 각 손가락 관절)와 뼈대를 그립니다.
- 손마다 `Left` / `Right` 라벨과 신뢰도를 표시합니다.
- 설정: `NUM_HANDS = 2`

### 2. Gesture Recognizer — `gesture_webcam.py`
- 손 랜드마크 위에 인식된 제스처 이름과 점수를 표시합니다.
- 기본 제스처: `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou` (해당 없음은 `None`)

### 3. Face Landmarker — `face_webcam.py`
- 478개 얼굴 메시(회색) + 윤곽/눈/눈썹/홍채/입술/코(색상별)를 그립니다.
- 좌측에 블렌드쉐이프 상위 5개(`eyeBlinkLeft`, `jawOpen`, `mouthSmileLeft` 등)를 막대그래프로 표시합니다.
- `m` 키로 전체 메시 표시를 켜고 끌 수 있습니다.
- 설정: `NUM_FACES = 1`, `TOP_BLENDSHAPES = 5`

### 4. Face Detector — `face_detect_webcam.py`
- BlazeFace short-range 모델로 얼굴 박스와 신뢰도, 키포인트 6개(양 눈, 코끝, 입, 양 귀)를 표시합니다.
- 화면에 검출된 얼굴 수를 표시합니다.
- 공식 문서에 full-range / sparse 모델도 있으나 short-range(카메라에서 2m 이내, 셀카·웹캠용)를 사용합니다.

## 공통 구조

```python
options = vision.XxxOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=on_result,      # 결과는 별도 스레드 콜백으로 전달
)
with vision.Xxx.create_from_options(options) as task:
    while True:
        frame = cv2.flip(cap.read()[1], 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        task.detect_async(mp_image, timestamp_ms)   # 타임스탬프는 단조 증가해야 함
        # 콜백이 저장한 최신 결과를 lock으로 읽어 그리기
```

## 모델 출처

Google AI Edge 공식 문서에서 다운로드한 모델이며 Apache License 2.0을 따릅니다.

| 파일 | URL |
|---|---|
| `hand_landmarker.task` | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task |
| `gesture_recognizer.task` | https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task |
| `face_landmarker.task` | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task |
| `blaze_face_short_range.tflite` | https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite |
