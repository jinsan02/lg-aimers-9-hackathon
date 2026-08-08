"""최종 블렌드 추론 (CatBoost×4 + XGBoost×1, 피처 v2) — 제출 zip의 script.py.

가중치는 2024 홀드아웃 greedy ensemble selection 결과(로컬 BSS 801.11).
모든 모델은 2024 포함 전체 데이터로 refit된 것이어야 함 (E18 교훈).
"""

import os

import joblib
import numpy as np
import pandas as pd

from features import add_features

ID_COL = "row_id"
TARGET_COL = "control_success"
WEIGHTS = [("./model/cat_fv2.pkl", 0.286),
           ("./model/cat_fv2s3.pkl", 0.286),
           ("./model/cat_fv2s2.pkl", 0.143),
           ("./model/cat_fv2s4.pkl", 0.143),
           ("./model/xgb_fv2.pkl", 0.143)]


def predict_one(pack, test):
    src = test
    if pack.get("feat_v2"):
        src, _ = add_features(test, pack["priors"])
    X = src[pack["features"]].copy()
    model = pack["model"]
    if type(model).__module__.startswith("xgboost"):
        import xgboost as xgb
        for c in pack["cat_cols"]:
            X[c] = X[c].astype("category")
        return model.predict(xgb.DMatrix(X, enable_categorical=True))
    for c in pack["cat_cols"]:
        X[c] = X[c].astype(str)
    return model.predict_proba(X)[:, 1]


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼 불일치: {list(sub.columns)}")
    print(f"test={len(test)} submission={len(sub)}")

    total_w = sum(w for _, w in WEIGHTS)
    preds = np.zeros(len(test))
    for path, w in WEIGHTS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 없음: {path}")
        p = predict_one(joblib.load(path), test)
        preds += (w / total_w) * p
        print(f"  {os.path.basename(path)} (w={w:.3f}): mean={p.mean():.4f}")

    preds = np.clip(preds, 0.0, 1.0)
    print(f"BLEND mean={preds.mean():.4f}")

    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)})")


if __name__ == "__main__":
    main()
