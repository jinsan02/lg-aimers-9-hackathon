"""추론 시간 벤치마크 — 평가 서버(6 vCPU) 조건 시뮬레이션.

실제 평가 데이터 크기(245,789행)로 모델 로드 + 예측 시간을 측정한다.
taskset -c 0-5 로 6코어 제한 실행:
  taskset -c 0-5 ~/.venvs/aimers/bin/python src/bench_inference.py
"""

import time

import joblib
import numpy as np
import pandas as pd

N_TEST = 245_789
ID, TARGET = "row_id", "control_success"

t0 = time.time()
train = pd.read_csv("./data/train.csv", encoding="utf-8-sig", nrows=400_000)
X = (train.drop(columns=[TARGET])
     .sample(n=N_TEST, replace=True, random_state=0)
     .reset_index(drop=True))
X[ID] = [f"TEST_{i:06d}" for i in range(N_TEST)]
print(f"모의 test 생성 {X.shape} :: {time.time() - t0:.1f}s")

t0 = time.time()
model = joblib.load("./model/rf.pkl")
print(f"모델 로드 :: {time.time() - t0:.1f}s")

t0 = time.time()
preds = model.predict_proba(X.drop(columns=[ID]))[:, 1]
el = time.time() - t0
print(f"RF(100트리, depth10) 245,789행 예측 :: {el:.1f}s")
print(f"→ 10분 한도 대비 {el / 600 * 100:.1f}% 사용")
print(f"예측 통계: mean={preds.mean():.4f} std={preds.std():.4f}")
