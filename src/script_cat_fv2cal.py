"""CatBoost ?쇱쿂v2 媛以??숈긽釉?異붾줎 ???쒖텧 zip?먯꽌 script.py 濡??ъ슜.

媛以묒튂: cat_fv2 0.5 / cat_fv2s3 1/3 / cat_fv2s2 1/6 (2024 ??쒖븘??greedy).
DELTA: ?ъ쟾 怨꾩궛??濡쒖쭞 ?쒗봽???곸닔(媛??됱뿉 ?낅┰ ?곸슜, train留뚯쑝濡??곗텧 ???됯? ?곗씠??遺꾪룷 誘몄궗??. 0?대㈃ 蹂댁젙 ?놁쓬.
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
DELTA = -0.05409  # 罹섎━釉뚮젅?댁뀡 ?곸닔 (compute_delta.py ?곗텧媛믪쓣 cal 踰꾩쟾??二쇱엯)


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 而щ읆 遺덉씪移? {list(sub.columns)}")
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

