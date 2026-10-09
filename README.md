# MediaPipe Webcam Examples

[MediaPipe Tasks](https://developers.google.com/edge/mediapipe/solutions/guide) Vision 모델 4종을 **웹캠으로 실시간 실행**하는 Python 예제 모음입니다.
모든 예제는 `LIVE_STREAM` 모드(비동기 콜백)로 동작하며, OpenCV로 결과를 화면에 그립니다.

### 🌐 웹 데모: https://kangchaeyeon82.github.io/mediapipe-webcam/

브라우저에서 바로 웹캠으로 커스텀 제스처 이모지 이펙트(👌 ❤️ 🖕)를 체험할 수 있습니다. (설치 불필요, 영상은 서버로 전송되지 않음)

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

## 커스텀 제스처 학습

내가 원하는 제스처를 직접 수집 → 학습 → 실시간 추론합니다.
Hand Landmarker가 뽑은 손 랜드마크 21점(x, y, z)을 특징으로 쓰고, scikit-learn MLP 분류기로 학습합니다.
(MediaPipe Model Maker는 Windows / Python 3.12에서 설치되지 않아 이 방식을 사용)

### GUI 앱 (권장) — `gesture_studio.py`

```bash
python gesture_studio.py
```

수집 → 학습 → 추론을 한 화면에서 진행합니다. 왼쪽은 웹캠 화면과 로그, 오른쪽은 컨트롤 패널입니다.

1. **① 제스처 라벨**: 이름 입력 후 `추가` (Enter). 목록에 라벨별 수집 개수가 표시됩니다.
2. **② 데이터 수집**: 라벨 선택 → `● 녹화 시작` 또는 `SPACE`.
   - `3초 대기`를 켜면 카운트다운 후 녹화 시작, `목표 개수`(기본 300)에 도달하면 자동 정지.
   - 녹화 중엔 손 뼈대가 빨간색으로 바뀌고 진행 바가 찹니다.
   - `선택 라벨 데이터 삭제`로 잘못 모은 라벨을 지울 수 있습니다 (확인 창 표시).
3. **③ 학습**: `학습 시작` → 검증 결과가 로그 창에 출력되고, 끝나면 모델이 자동으로 로드됩니다.
4. **④ 실시간 추론**: `추론 켜기` 체크 → 화면과 패널에 제스처/확률 표시. `임계값` 슬라이더 미만이면 `Unknown`.

카메라가 안 잡히면 상단 `카메라` 번호를 바꾸고 `연결`을 누르세요.

### CLI 스크립트

| 파일 | 역할 |
|---|---|
| `gesture_studio.py` | GUI 앱 (수집/학습/추론 통합) |
| `collect_gestures.py` | 웹캠으로 라벨별 랜드마크 수집 → `data/gestures.csv` |
| `train_gestures.py` | CSV로 분류기 학습, 검증 결과 출력 → `models/custom_gesture.joblib` |
| `custom_gesture_webcam.py` | 학습한 모델로 실시간 추론 |
| `gesture_features.py` | 공통 전처리 (손목 기준 이동, 크기 정규화, 왼손 미러링) |

### 1) 수집

```bash
python collect_gestures.py --labels none rock scissors paper
```

| 키 | 동작 |
|---|---|
| `1`~`9` | 수집할 라벨 선택 (`--labels` 순서) |
| `SPACE` | 녹화 시작/정지 — 녹화 중엔 손이 보이는 매 프레임이 저장됨 |
| `q` / `ESC` | 종료 |

- 라벨당 **200~500개** 권장 (30fps 기준 10~15초 녹화).
- 녹화 중에 손을 조금씩 돌리고, 거리를 바꾸고, 양손을 번갈아 쓰면 일반화가 잘 됩니다.
- **`none` 라벨**(아무 제스처도 아닌 평범한 손)을 함께 수집하면 오인식이 크게 줄어듭니다.
- 여러 번 실행해도 같은 CSV에 이어서 저장되므로, 나중에 라벨을 추가해도 됩니다.

### 2) 학습

```bash
python train_gestures.py
```

- 데이터 80%로 학습, 20%로 검증해 정확도/혼동 행렬을 출력한 뒤, 전체 데이터로 다시 학습해 저장합니다.
- 특정 라벨끼리 헷갈리면 해당 라벨 데이터를 더 수집하고 다시 학습하세요.

### 3) 추론

```bash
python custom_gesture_webcam.py --threshold 0.6
```

- 손마다 `Left/Right: 제스처 (확률)` 을 표시하고, 확률이 `--threshold` 미만이면 `Unknown` 으로 표시합니다.
- 왼손은 좌우 반전해서 오른손 모양으로 맞추므로, 한쪽 손으로만 수집해도 양손에서 인식됩니다.

## 제스처 이모지 이펙트 — `gesture_effects.py`

학습한 커스텀 제스처 모델로 화면에 이모지 이펙트를 띄웁니다.

```bash
python gesture_effects.py      # q/ESC: 종료, l: 손 랜드마크 표시 on/off
```

| 라벨 | 이펙트 |
|---|---|
| `okay` / `ok` | 👌 손 위에 튕기듯 팝업, 제스처 유지 중 살랑살랑 표시 |
| `heart` | ❤️ 손 주변에서 하트가 계속 피어올라 흔들리며 위로 떠다님 |
| `fuck you` | 🖕 화면 가운데로 갑자기 튀어나옴 (플래시 + 흔들림) |

- 연속 3프레임 이상 인식돼야 시작하고, 6프레임 사라지면 종료합니다(깜빡임 방지). `THRESHOLD`(기본 0.7)로 민감도 조절.
- 라벨 이름이 다르면 스크립트 상단 `LABEL_TO_EFFECT`에 추가하세요.
- 이모지는 Windows 기본 폰트 `Segoe UI Emoji`(seguiemj.ttf)로 그립니다.

## 웹 데모 (GitHub Pages) — `docs/`

`gesture_effects.py`를 브라우저용으로 옮긴 버전입니다. GitHub Pages가 `main` 브랜치의 `docs/` 폴더를 배포합니다.

| 파일 | 설명 |
|---|---|
| `docs/index.html` | 페이지 + MediaPipe Tasks Vision(JS)로 손 랜드마크 검출 + 이펙트 |
| `docs/custom_gesture.json` | 학습한 분류기(StandardScaler + MLP 가중치)를 JSON으로 내보낸 것 |
| `docs/hand_landmarker.task` | 손 랜드마크 모델 |
| `export_web_model.py` | `models/custom_gesture.joblib` → `docs/custom_gesture.json` 변환 |

모델을 다시 학습했다면 웹 데모에도 반영하기 위해:

```bash
python export_web_model.py
git add docs/custom_gesture.json && git commit -m "Update web model" && git push
```

로컬에서 미리 보기: `cd docs && python -m http.server 8000` → http://localhost:8000

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
