"""blendv3 구성 + 최신 시즌(2024) ×3 가중 refit — 제출 zip의 script.py.

blendv3(LB 861.66)와 **모델 구성·greedy 가중이 완전히 동일**하고,
refit 단계에서 2024 시즌 샘플에만 ×3 가중을 준 것만 다르다.
→ E85(최신 시즌 가중)의 순수 A/B.

근거: F리그는 2023이 ABS 1년차·2024가 2년차로 R리그의 2024→2025와 구조가 같다.
F리그 예행연습(3시드 평균)에서 W=3이 +54.1(≈3σ)로 최적.
검증 점수(2024 홀드아웃)로는 판정 불가 — refit만 바뀌므로 로컬 점수는 blendv3와 동일하다.
"""

import os

import joblib
import numpy as np
import pandas as pd

from features import add_features

ID_COL = "row_id"
TARGET_COL = "control_success"
# blendv3와 동일한 greedy 가중 (2024 홀드아웃에서 산출, 로컬 804.70)
WEIGHTS = [("./model/cat_w3_hs_d7_l10_r0.08.pkl", 0.200),
           ("./model/cat_w3_hs_d8_l3_r0.05.pkl", 0.200),
           ("./model/cat_w3_fv2s3.pkl", 0.200),
           ("./model/cat_w3_hs_d8_l10_r0.05.pkl", 0.100),
           ("./model/cat_w3_fv2.pkl", 0.100),
           ("./model/cat_w3_fv2s4.pkl", 0.100),
           ("./model/xgb_w3_fv2x.pkl", 0.100)]


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
    print(f"BLEND(W3) mean={preds.mean():.4f}")

    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)})")


if __name__ == "__main__":
    main()
