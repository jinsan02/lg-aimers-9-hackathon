"""캘리브레이션 상수 δ 오프라인 계산 (train 데이터만 사용 — 규칙 적합).

δ = logit(r_hat_2025) - logit(m_2024)
  r_hat_2025: 2019~2024 시즌 성공률 선형추세의 2025 외삽
  m_2024: refit 앙상블이 train의 2024 행에 내는 예측 평균 (모델의 '현재 눈높이')

script.py는 이 δ를 상수로 받아 각 행에 독립 적용 → test 분포 미사용.
실행: ~/.venvs/aimers/bin/python src/compute_delta.py
"""

import joblib
import numpy as np
import pandas as pd

from features import add_features

MODELS = [("model/cat_fv2.pkl", 0.5), ("model/cat_fv2s3.pkl", 1 / 3),
          ("model/cat_fv2s2.pkl", 1 / 6)]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    rates = pd.Series({2019: 0.5647, 2020: 0.5327, 2021: 0.5328,
                       2022: 0.5289, 2023: 0.5000, 2024: 0.4861})
    r_hat = float(np.polyval(np.polyfit(rates.index, rates.values, 1), 2025))
    print(f"r_hat(2025) 선형추세 = {r_hat:.4f}")

    test_cols = pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=0).columns
    train = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    t24 = train[train.season == 2024].reset_index(drop=True)

    preds = np.zeros(len(t24))
    for path, w in MODELS:
        pack = joblib.load(path)
        src = t24
        if pack.get("feat_v2"):
            src, _ = add_features(t24, pack["priors"])
        X = src[pack["features"]].copy()
        for c in pack["cat_cols"]:
            X[c] = X[c].astype(str)
        preds += w * pack["model"].predict_proba(X)[:, 1]

    m24 = preds.mean()
    delta = float(logit(np.array([r_hat]))[0] - logit(np.array([m24]))[0])
    print(f"m_2024(refit 앙상블, in-sample) = {m24:.4f}")
    print(f"DELTA = {delta:.5f}")


if __name__ == "__main__":
    main()
