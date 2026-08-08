"""블렌드 추론 스크립트 — 제출 zip에서 script.py 로 사용.

0.65 × LGBM(E06) + 0.35 × TabM v4 3시드 평균.
zip: model/{lgbm_v3.pkl, prep_v1.pkl, tabm_v4s*.pt}, script.py,
     tabm_reference.py, rtdl_num_embeddings.py, requirements.txt(lightgbm)
"""

import glob
import os

import joblib
import numpy as np
import pandas as pd
import torch

from tabm_reference import Model

ID_COL = "row_id"
TARGET_COL = "control_success"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.65  # LGBM 비중 (2024 홀드아웃에서 탐색)

CAT_NAMES = ["top_bottom", "game_type", "base_state", "pitcher_hand",
             "batter_hand", "pitcher_team_id", "batter_team_id",
             "pitcher_id", "batter_id"]


def load_tabm(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    cfg = ckpt["config"]
    bins = [b.to(DEVICE) for b in ckpt["bins"]] if ckpt["bins"] else None
    model = Model(n_num_features=cfg["n_num_features"],
                  cat_cardinalities=cfg["cat_cardinalities"], n_classes=None,
                  backbone=cfg["backbone"], bins=bins,
                  num_embeddings=cfg["num_embeddings"],
                  arch_type=cfg["arch_type"], k=cfg["k"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    kept = cfg.get("cat_cols_kept") or CAT_NAMES
    return model, [CAT_NAMES.index(c) for c in kept]


@torch.no_grad()
def tabm_predict(model, X_num, X_cat, batch=16384):
    preds = []
    for i in range(0, len(X_num), batch):
        xn = torch.as_tensor(X_num[i:i + batch]).to(DEVICE)
        xc = torch.as_tensor(X_cat[i:i + batch]).to(DEVICE)
        out = model(xn, xc).squeeze(-1)
        preds.append(torch.sigmoid(out.float()).mean(1).cpu())
    return torch.cat(preds).numpy()


def main():
    print(f"Device: {DEVICE}")
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼 불일치: {list(sub.columns)}")
    print(f"test={len(test)} submission={len(sub)}")

    # ---- LGBM ----
    pack = joblib.load("./model/lgbm_v3.pkl")
    X = test[pack["features"]].copy()
    for c in pack["cat_cols"]:
        X[c] = pd.Categorical(X[c], categories=pack["cat_levels"][c])
    p_lgbm = pack["booster"].predict(X)
    print(f"LGBM: mean={p_lgbm.mean():.4f}")

    # ---- TabM 시드 앙상블 ----
    prep = joblib.load("./model/prep_v1.pkl")
    X_cat_full = prep["cat_enc"].transform(test[prep["cat_cols"]]).astype(np.int64) + 1
    X_num = prep["num_pipe"].transform(test[prep["num_cols"]]).astype(np.float32)
    tabm_preds = []
    for path in sorted(glob.glob("./model/tabm_v4s*_best.pt")):
        model, cat_idx = load_tabm(path)
        tabm_preds.append(tabm_predict(model, X_num, X_cat_full[:, cat_idx]))
        print(f"TabM {os.path.basename(path)}: mean={tabm_preds[-1].mean():.4f}")
    assert tabm_preds, "TabM 체크포인트 없음"
    p_tabm = np.mean(tabm_preds, axis=0)

    # ---- 블렌드 ----
    preds = ALPHA * p_lgbm + (1 - ALPHA) * p_tabm
    preds = np.clip(preds, 0.0, 1.0)
    print(f"BLEND(a={ALPHA}): mean={preds.mean():.4f}")

    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)})")


if __name__ == "__main__":
    main()
