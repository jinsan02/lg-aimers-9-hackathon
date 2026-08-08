"""블렌드 v5 — 제출 zip의 script.py.

v3(LB 861.66) / w3v3(LB 865.13) 대비 바뀐 점:
  1) **학습률 0.05 → 0.02** (E92). A100 3시드 페어드 +24.96, t=4.44.
     트리 150~250개 → 550~700개. 시드 분산도 σ 13.3 → 3.6 으로 감소.
  2) **시즌 expanding 타깃 인코딩 추가** (E91, src/target_enc.py).
     신호감사 오라클이 투수x카운트 2740.9 / 투수x타자손 1425.5 로 미개발 영역을
     가리켰다. raw ID는 유해했으므로(E88 -105) 수축된 수치 요약으로 넣는다.
  3) refit 시 2024 시즌 x3 가중 유지 (E85, LB 실측 +3.47).

TE 표는 pkl 안에 (키, 표) 형태로 동봉된다. 표의 season=2025 행은 2019~2024 누적이며
평가 행마다 독립적으로 조인되므로, test 내부 다른 행을 쓰지 않는다(규칙 준수).
"""

import os

import joblib
import numpy as np
import pandas as pd

from features import add_features

ID_COL = "row_id"
TARGET_COL = "control_success"
# tools/group_blend.py 결과 — 시드는 균등평균, 설정끼리만 greedy (로컬 809.66)
WEIGHTS = [("./model/cat_n6_d6.pkl", 0.1667),
           ("./model/cat_w3_hs_d7_l10_r0.08.pkl", 0.0833),
           ("./model/cat_w3_hs_d8_l3_r0.05.pkl", 0.0833),
           ("./model/cat_w3_hs_d8_l10_r0.05.pkl", 0.0833),
           ("./model/cat_n3_d8l10.pkl", 0.0648),
           ("./model/cat_n4_s3.pkl", 0.0648),
           ("./model/cat_n5_s4.pkl", 0.0648),
           ("./model/cat_n_s5.pkl", 0.0648),
           ("./model/cat_n_s6.pkl", 0.0648),
           ("./model/cat_n_s8.pkl", 0.0648),
           ("./model/cat_n_s9.pkl", 0.0648),
           ("./model/cat_n_s11.pkl", 0.0648),
           ("./model/cat_n_s21.pkl", 0.0648)]


def predict_one(pack, test):
    src = test
    if pack.get("feat_v2"):
        src, _ = add_features(src, pack["priors"])
    if pack.get("te_tables"):
        import target_enc as te_mod
        src = te_mod.add_inning_bucket(src)
        src, _ = te_mod.add_te(src, pack["te_tables"], dev=pack.get("te_dev"))

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
    print(f"BLEND v5 mean={preds.mean():.4f}")

    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)})")


if __name__ == "__main__":
    main()
