"""최종 블렌드 v4 (CatBoost 7종 + XGBoost 1종) — 제출 zip의 script.py.

가중치: 2024 홀드아웃 greedy ensemble selection (로컬 BSS 805.83).
모든 모델은 2024 포함 전체 데이터 refit본 (E18 교훈).
피처 v2 + (모델별로) 역할·규칙/체력 파생을 재현한다.
"""

import os

import joblib
import numpy as np
import pandas as pd

from features import add_features

ID_COL = "row_id"
TARGET_COL = "control_success"
WEIGHTS = [("./model/cat_hs_d7_l10_r0.08.pkl", 0.167),
           ("./model/cat_hs_d8_l3_r0.05.pkl", 0.167),
           ("./model/cat_fv2s3.pkl", 0.167),
           ("./model/cat_role42.pkl", 0.167),
           ("./model/cat_fv2.pkl", 0.083),
           ("./model/cat_fv2s4.pkl", 0.083),
           ("./model/xgb_fv2.pkl", 0.083),
           ("./model/cat_fatig.pkl", 0.083)]


def predict_one(pack, test):
    src = test
    if pack.get("feat_v2"):
        src, _ = add_features(src, pack["priors"])
    if pack.get("feat_role") and pack.get("role_table") is not None:
        from pitcher_role import add_role
        src, _ = add_role(src, pack["role_table"])
    if pack.get("feat_rules"):
        from rules import add_rule_features
        src, _ = add_rule_features(src)
    if pack.get("feat_fatigue"):
        from rules import add_fatigue_features
        src, _ = add_fatigue_features(src)

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
