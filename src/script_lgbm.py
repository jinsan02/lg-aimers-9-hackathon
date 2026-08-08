"""LightGBM 추론 스크립트 — 제출 zip에서 script.py 로 사용.

zip: model/lgbm_v3.pkl, script.py, requirements.txt(lightgbm 포함)
"""

import os

import joblib
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


def main():
    TEST_PATH = "./data/test.csv"
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    OUT_PATH = "./output/submission.csv"

    print("Load model...")
    pack = joblib.load("./model/lgbm_v3.pkl")
    booster, features = pack["booster"], pack["features"]
    cat_cols, cat_levels = pack["cat_cols"], pack["cat_levels"]

    print("Load test data...")
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sub = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼 불일치: {list(sub.columns)}")
    print(f" test={len(test)} submission={len(sub)}")

    print("Preprocess...")
    X = test[features].copy()
    for c in cat_cols:  # 학습 때와 동일한 category 레벨로 고정 (미지 레벨 → NaN)
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])

    print("Inference...")
    preds = booster.predict(X)
    print(f" preds={len(preds)} mean={preds.mean():.4f}")

    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
