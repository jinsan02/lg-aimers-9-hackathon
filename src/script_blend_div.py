"""다양성 균등 블렌드 — 제출 zip의 script.py.

선택 근거(tools/diversity_blend.py 일반화 검정):
  전반기로 결정 → **후반기(미사용 구간) 평가**
    greedy 가중      613.36
    다양성 균등      **621.30**  ← +7.9 우세
    전체 균등        609.49
  greedy는 검증셋에 맞춰져 로컬 점수는 높지만(804.70 vs 797.45) 일반화가 나쁘다.
  평가(2025)는 완전 미사용 구간이므로 **다양성 균등을 채택**한다.

구성: 피처 계열이 서로 다른 6개 모델 균등 가중.
  hs(하이퍼 변형) / fv2(기본) / xgb(다른 알고리즘) / role(투수 역할) /
  fatig(규칙·체력) / mgr(감독 운용)
"""

import os

import joblib
import numpy as np
import pandas as pd

from features import add_features

ID_COL = "row_id"
TARGET_COL = "control_success"
MODELS = ["./model/cat_hs_d7_l10_r0.08.pkl",
          "./model/cat_fv2.pkl",
          "./model/xgb_fv2.pkl",
          "./model/cat_role42.pkl",
          "./model/cat_fatig.pkl",
          "./model/cat_mgr42.pkl"]


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
        raise ValueError(f"sample_submission 컬럼 불일치: {list(sub.columns)}")
    print(f"test={len(test)} submission={len(sub)}")

    preds = np.zeros(len(test))
    for path in MODELS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 없음: {path}")
        p = predict_one(joblib.load(path), test)
        preds += p / len(MODELS)
        print(f"  {os.path.basename(path)}: mean={p.mean():.4f}")

    preds = np.clip(preds, 0.0, 1.0)
    print(f"BLEND(균등 {len(MODELS)}개) mean={preds.mean():.4f}")

    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)})")


if __name__ == "__main__":
    main()
