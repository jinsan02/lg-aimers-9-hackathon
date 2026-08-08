"""TabM 시드 체크포인트들의 개별/앙상블 검증 BSS 평가 + 검증 예측 저장.

실행(A100): ~/venv451/bin/python src/eval_tabm_ensemble.py out/tabm_v2s0_best.pt out/tabm_v2s1_best.pt ...
산출: out/tabm_val_preds.npz (각 시드 및 앙상블의 2024 검증 예측 — 블렌딩용)
"""

import sys

import numpy as np
import torch

from tabm_reference import Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA = "data/processed/train_v1.npz"
VAL_SEASON = 2024
CAT_NAMES = ["top_bottom", "game_type", "base_state", "pitcher_hand",
             "batter_hand", "pitcher_team_id", "batter_team_id",
             "pitcher_id", "batter_id"]


def bss(y, p):
    r = y.mean()
    return float(max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))))


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
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
    cat_idx = [CAT_NAMES.index(c) for c in kept]
    return model, cat_idx


@torch.no_grad()
def predict(model, X_num, X_cat, batch=16384):
    preds = []
    for i in range(0, len(X_num), batch):
        xn = torch.as_tensor(X_num[i:i + batch]).to(DEVICE)
        xc = torch.as_tensor(X_cat[i:i + batch]).to(DEVICE)
        out = model(xn, xc).squeeze(-1)
        preds.append(torch.sigmoid(out.float()).mean(1).cpu())
    return torch.cat(preds).numpy()


def main():
    ckpts = sys.argv[1:]
    assert ckpts, "체크포인트 경로를 인자로 전달"
    d = np.load(DATA)
    is_val = d["season"] == VAL_SEASON
    Xn, Xc = d["X_num"][is_val], d["X_cat"][is_val].astype(np.int64)
    y = d["y"][is_val].astype(np.float64)

    all_preds = {}
    for p in ckpts:
        model, cat_idx = load_model(p)
        pred = predict(model, Xn, Xc[:, cat_idx])
        all_preds[p] = pred
        print(f"{p}: BSS {bss(y, pred):.2f}")

    ens = np.mean(list(all_preds.values()), axis=0)
    print(f"ENSEMBLE({len(ckpts)}): BSS {bss(y, ens):.2f}")

    np.savez_compressed("out/tabm_val_preds.npz", y=y, ensemble=ens,
                        **{f"m{i}": v for i, v in enumerate(all_preds.values())})
    print("saved: out/tabm_val_preds.npz")


if __name__ == "__main__":
    main()
