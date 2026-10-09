"""커스텀 제스처 분류기 학습 (scikit-learn MLP)

사용 예:
    python train_gestures.py
    python train_gestures.py --data data/gestures.csv --out models/custom_gesture.joblib

입력: collect_gestures.py 로 만든 CSV
출력: 학습된 모델(joblib) - custom_gesture_webcam.py 에서 사용
"""
import argparse
import csv
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gesture_features import normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dataset(path):
    X, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 헤더
        for row in reader:
            if len(row) < 65:
                continue
            label, hand, raw = row[0], row[1], [float(v) for v in row[2:65]]
            X.append(normalize(raw, hand))
            y.append(label)
    return np.array(X), np.array(y)


def train(data_path, out_path, test_size=0.2, log=print):
    """CSV로 분류기를 학습해 out_path에 저장하고 검증 정확도를 반환. GUI에서도 재사용한다."""
    X, y = load_dataset(data_path)
    counts = Counter(y)
    log(f"샘플 수: {len(y)}")
    for name, n in sorted(counts.items()):
        log(f"  {name}: {n}")
    if len(counts) < 2:
        raise ValueError("라벨이 2개 이상 필요합니다. 데이터를 더 수집하세요.")
    if min(counts.values()) < 5:
        raise ValueError("샘플이 5개 미만인 라벨이 있습니다. 데이터를 더 수집하세요.")
    if min(counts.values()) < 100:
        log("경고: 샘플이 100개 미만인 라벨이 있습니다. 라벨당 200개 이상을 권장합니다.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000,
                      early_stopping=len(y_train) >= 50, random_state=42),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    labels = [str(c) for c in model.classes_]
    acc = float((y_pred == y_test).mean())
    log("\n[검증 결과]")
    log(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    log("혼동 행렬 (행: 정답, 열: 예측)")
    log(f"labels: {labels}")
    log(str(confusion_matrix(y_test, y_pred, labels=labels)))

    # 최종 모델은 전체 데이터로 다시 학습
    model.fit(X, y)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    joblib.dump({"model": model, "labels": labels}, out_path)
    log(f"\n모델 저장: {os.path.abspath(out_path)}")
    return acc


def main():
    parser = argparse.ArgumentParser(description="커스텀 제스처 분류기 학습")
    parser.add_argument("--data", default=os.path.join(BASE_DIR, "data", "gestures.csv"))
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "models", "custom_gesture.joblib"))
    parser.add_argument("--test-size", type=float, default=0.2, help="검증용 데이터 비율")
    args = parser.parse_args()
    try:
        train(args.data, args.out, args.test_size)
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
