"""CatBoost 피처v2 가중 앙상블 추론 — 제출 zip에서 script.py 로 사용.

가중치: cat_fv2 0.5 / cat_fv2s3 1/3 / cat_fv2s2 1/6 (2024 홀드아웃 greedy).
DELTA: 사전 계산된 로짓 시프트 상수(각 행에 독립 적용, train만으로 산출 —
평가 데이터 분포 미사용). 0이면 보정 없음.
"""

import os

import joblib
import numpy as np
import pandas as pd

from features import add_features

ID_COL = "row_id"
TARGET_COL = "control_success"
MODELS = [("./model/cat_fv2.pkl", 0.5),
          ("./model/cat_fv2s3.pkl", 1 / 3),
          ("./model/cat_fv2s2.pkl", 1 / 6)]
DELTA = 0.0  # 캘리브레이션 상수 (compute_delta.py 산출값을 cal 버전에 주입)


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼 불일치: {list(sub.columns)}")
    print(f"test={len(test)} submission={len(sub)} | DELTA={DELTA}")

    preds = np.zeros(len(test))
    for path, w in MODELS:
        pack = joblib.load(path)
        src = test
        if pack.get("feat_v2"):
            src, _ = add_features(test, pack["priors"])
        X = src[pack["features"]].copy()
        for c in pack["cat_cols"]:
            X[c] = X[c].astype(str)
        p = pack["model"].predict_proba(X)[:, 1]
        preds += w * p
        print(f"  {os.path.basename(path)} (w={w:.3f}): mean={p.mean():.4f}")

    if DELTA != 0.0:
        z = np.log(np.clip(preds, 1e-6, 1 - 1e-6) /
                   (1 - np.clip(preds, 1e-6, 1 - 1e-6)))
        preds = 1 / (1 + np.exp(-(z + DELTA)))
    preds = np.clip(preds, 0.0, 1.0)
    print(f"FINAL mean={preds.mean():.4f}")

    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)})")


if __name__ == "__main__":
    main()
