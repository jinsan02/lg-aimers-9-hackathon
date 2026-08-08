"""CatBoost 추론 스크립트 — 제출 zip에서 script.py 로 사용.

zip: model/cat_best.pkl, script.py, requirements.txt(catboost)
"""

import os

import joblib
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


def main():
    print("Load model...")
    pack = joblib.load("./model/cat_best.pkl")
    model, features, cat_cols = pack["model"], pack["features"], pack["cat_cols"]

    print("Load test data...")
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼 불일치: {list(sub.columns)}")
    print(f" test={len(test)} submission={len(sub)}")

    X = test[features].copy()
    for c in cat_cols:  # 학습 때와 동일: str 캐스팅
        X[c] = X[c].astype(str)

    print("Inference...")
    preds = model.predict_proba(X)[:, 1]
    print(f" preds={len(preds)} mean={preds.mean():.4f}")

    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)})")


if __name__ == "__main__":
    main()
