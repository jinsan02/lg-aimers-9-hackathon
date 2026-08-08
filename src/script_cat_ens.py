"""CatBoost 시드 앙상블 추론 — 제출 zip에서 script.py 로 사용.

model/ 안의 cat_*.pkl 전부에 대해 predict_proba 후 가중 평균.
가중치: cat_l2_3=0.5, 나머지 균등 (2024 홀드아웃 greedy 탐색 결과).
"""

import os

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
MAIN = "cat_l2_3"
MAIN_W = 0.5
MODELS = ["./model/cat_l2_3.pkl", "./model/cat_s1.pkl",
          "./model/cat_s2.pkl", "./model/cat_s3.pkl"]


def main():
    paths = [p for p in MODELS if os.path.exists(p)]
    assert len(paths) == len(MODELS), f"모델 누락: {set(MODELS) - set(paths)}"
    print(f"models: {[os.path.basename(p) for p in paths]}")

    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼 불일치: {list(sub.columns)}")
    print(f"test={len(test)} submission={len(sub)}")

    preds, weights = [], []
    for p in paths:
        pack = joblib.load(p)
        src = test
        if pack.get("feat_v2"):
            from features import add_features
            src, _ = add_features(test, pack["priors"])
        X = src[pack["features"]].copy()
        for c in pack["cat_cols"]:
            X[c] = X[c].astype(str)
        pr = pack["model"].predict_proba(X)[:, 1]
        w = MAIN_W if MAIN in p else None
        preds.append(pr)
        weights.append(w)
        print(f"  {os.path.basename(p)}: mean={pr.mean():.4f}")

    n_rest = sum(1 for w in weights if w is None)
    rest_w = (1 - MAIN_W) / n_rest if n_rest else 0
    weights = [w if w is not None else rest_w for w in weights]
    final = np.average(preds, axis=0, weights=weights)
    final = np.clip(final, 0.0, 1.0)
    print(f"ENSEMBLE mean={final.mean():.4f} weights={weights}")

    pred_map = dict(zip(test[ID_COL], final))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)})")


if __name__ == "__main__":
    main()
