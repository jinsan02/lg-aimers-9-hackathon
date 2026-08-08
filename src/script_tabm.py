"""TabM 추론 스크립트 — 제출 zip에서 script.py 로 사용.

zip 구조 (루트에 vendored 모듈 포함):
  model/prep_v1.pkl, model/tabm_v1_best.pt, script.py,
  tabm_reference.py, rtdl_num_embeddings.py, requirements.txt(빈 파일)

서버: torch 2.7.1+cu128 기본 설치, L4 GPU. CPU 폴백도 지원.
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch

from tabm_reference import Model

ID_COL = "row_id"
TARGET_COL = "control_success"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    cfg = ckpt["config"]
    bins = [b.to(DEVICE) for b in ckpt["bins"]] if ckpt["bins"] else None
    model = Model(
        n_num_features=cfg["n_num_features"],
        cat_cardinalities=cfg["cat_cardinalities"],
        n_classes=None,
        backbone=cfg["backbone"],
        bins=bins,
        num_embeddings=cfg["num_embeddings"],
        arch_type=cfg["arch_type"],
        k=cfg["k"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, X_num, X_cat, batch=16384):
    preds = []
    for i in range(0, len(X_num), batch):
        xn = torch.as_tensor(X_num[i:i + batch]).to(DEVICE)
        xc = torch.as_tensor(X_cat[i:i + batch]).to(DEVICE)
        out = model(xn, xc).squeeze(-1)              # (B, k)
        preds.append(torch.sigmoid(out.float()).mean(1).cpu())
    return torch.cat(preds).numpy() if preds else np.array([])


def main():
    TEST_PATH = "./data/test.csv"
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    OUT_PATH = "./output/submission.csv"

    print(f"Device: {DEVICE}")
    print("Load preprocessor & model...")
    prep = joblib.load("./model/prep_v1.pkl")
    model = load_model("./model/tabm_v1_best.pt")

    print("Load test data...")
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    sub = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼 불일치: {list(sub.columns)}")
    print(f" test={len(test)} submission={len(sub)}")

    print("Preprocess...")
    X_cat = prep["cat_enc"].transform(test[prep["cat_cols"]]).astype(np.int64) + 1
    ckpt_cfg = torch.load("./model/tabm_v1_best.pt",
                          map_location="cpu", weights_only=True)["config"]
    kept = ckpt_cfg.get("cat_cols_kept")
    if kept:  # 학습 때 일부 범주형을 제외한 경우 동일하게 선택
        idx = [prep["cat_cols"].index(c) for c in kept]
        X_cat = X_cat[:, idx]
    X_num = prep["num_pipe"].transform(test[prep["num_cols"]]).astype(np.float32)

    print("Inference...")
    preds = predict(model, X_num, X_cat)
    print(f" preds={len(preds)} mean={preds.mean():.4f}")

    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
