"""학습한 커스텀 제스처 모델(scikit-learn)을 웹 데모용 JSON으로 내보내기

실행: python export_web_model.py
출력: docs/custom_gesture.json  (docs/index.html 이 읽어서 브라우저에서 추론)

모델을 다시 학습했다면 이 스크립트를 다시 실행하고 커밋/푸시하면 웹 데모에도 반영된다.
"""
import json
import os

import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "custom_gesture.joblib")
OUT_PATH = os.path.join(BASE_DIR, "docs", "custom_gesture.json")


def main():
    model = joblib.load(MODEL_PATH)["model"]
    scaler, mlp = model[0], model[-1]
    data = {
        "labels": [str(c) for c in mlp.classes_],
        "mean": scaler.mean_.round(6).tolist(),
        "scale": scaler.scale_.round(6).tolist(),
        "activation": mlp.activation,
        "out_activation": mlp.out_activation_,
        # weights[i]: (입력 크기 x 출력 크기) 행렬, biases[i]: 출력 크기
        "weights": [w.round(6).tolist() for w in mlp.coefs_],
        "biases": [b.round(6).tolist() for b in mlp.intercepts_],
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"저장: {OUT_PATH}  ({os.path.getsize(OUT_PATH) / 1024:.0f} KB), 라벨: {data['labels']}")


if __name__ == "__main__":
    main()
