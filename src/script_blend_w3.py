"""?ㅼ뼇??洹좊벑 釉붾젋?????쒖텧 zip??script.py.

?좏깮 洹쇨굅(tools/diversity_blend.py ?쇰컲??寃??:
  ?꾨컲湲곕줈 寃곗젙 ??**?꾨컲湲?誘몄궗??援ш컙) ?됯?**
    greedy 媛以?     613.36
    ?ㅼ뼇??洹좊벑      **621.30**  ??+7.9 ?곗꽭
    ?꾩껜 洹좊벑        609.49
  greedy??寃利앹뀑??留욎떠??濡쒖뺄 ?먯닔???믪?留?804.70 vs 797.45) ?쇰컲?붽? ?섏걯??
  ?됯?(2025)???꾩쟾 誘몄궗??援ш컙?대?濡?**?ㅼ뼇??洹좊벑??梨꾪깮**?쒕떎.

援ъ꽦: ?쇱쿂 怨꾩뿴???쒕줈 ?ㅻⅨ 6媛?紐⑤뜽 洹좊벑 媛以?
  hs(?섏씠??蹂?? / fv2(湲곕낯) / xgb(?ㅻⅨ ?뚭퀬由ъ쬁) / role(?ъ닔 ??븷) /
  fatig(洹쒖튃쨌泥대젰) / mgr(媛먮룆 ?댁슜)
"""

import os

import joblib
import numpy as np
import pandas as pd

from features import add_features

ID_COL = "row_id"
TARGET_COL = "control_success"
MODELS = ["./model/cat_w3_hs_d7_l10_r0.08.pkl",
          "./model/cat_w3_fv2.pkl",
          "./model/xgb_w3_fv2x.pkl",
          "./model/cat_w3_role42.pkl",
          "./model/cat_w3_fatig.pkl",
          "./model/cat_w3_mgr42.pkl"]


def predict_one(pack, test):
    src = test
    if pack.get("feat_v2"):
        src, _ = add_features(src, pack["priors"])
    if pack.get("feat_mgr") and pack.get("mgr_table") is not None:
        from manager_feat import add_style
        src, _ = add_style(src, pack["mgr_table"])
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
        raise ValueError(f"sample_submission 而щ읆 遺덉씪移? {list(sub.columns)}")
    print(f"test={len(test)} submission={len(sub)}")

    preds = np.zeros(len(test))
    for path in MODELS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"紐⑤뜽 ?놁쓬: {path}")
        p = predict_one(joblib.load(path), test)
        preds += p / len(MODELS)
        print(f"  {os.path.basename(path)}: mean={p.mean():.4f}")

    preds = np.clip(preds, 0.0, 1.0)
    print(f"BLEND(洹좊벑 {len(MODELS)}媛? mean={preds.mean():.4f}")

    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)})")


if __name__ == "__main__":
    main()

