"""TabM 학습 — A100(hsu-server, ~/venv451, torch 2.5.1+cu121, Python 3.10)용.

입력: data/processed/train_v1.npz (노트북에서 prep_v1.py로 생성)
출력: out/tabm_v1_best.pt (state_dict + config + 검증 점수), out/tabm_v1_log.json

실행 예:
  python src/train_tabm.py --tag v1 --num-emb ple
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import rtdl_num_embeddings
from tabm_reference import Model, make_parameter_groups

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# prep_v1.py의 CAT_COLS 순서와 동일해야 함
CAT_NAMES = ["top_bottom", "game_type", "base_state", "pitcher_hand",
             "batter_hand", "pitcher_team_id", "batter_team_id",
             "pitcher_id", "batter_id"]


def brier_skill_raw(y_true, y_pred):
    """클리핑 없는 skill score — 음수 구간에서도 개선 방향을 보기 위함.
    (리더보드 점수는 max(0, raw)지만, 모델 선택/early stopping은 raw 기준)"""
    r = y_true.mean()
    brier = ((y_pred - y_true) ** 2).mean()
    return float(100000 * (1 - brier / (r * (1 - r))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/train_v1.npz")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--arch", default="tabm-mini")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--num-emb", choices=["none", "ple"], default="ple")
    ap.add_argument("--d-embedding", type=int, default=16)
    ap.add_argument("--n-bins", type=int, default=48)
    ap.add_argument("--n-blocks", type=int, default=3)
    ap.add_argument("--d-block", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--eval-batch", type=int, default=16384)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val-season", type=int, default=2024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-cat", default="",
                    help="제외할 범주형 (쉼표 구분, 예: pitcher_id,batter_id)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.data)
    X_num, X_cat = d["X_num"], d["X_cat"].astype(np.int64)
    y, season = d["y"].astype(np.float32), d["season"]
    cat_keep = list(range(X_cat.shape[1]))
    if args.drop_cat:
        drop = set(args.drop_cat.split(","))
        cat_keep = [i for i, n in enumerate(CAT_NAMES) if n not in drop]
        X_cat = X_cat[:, cat_keep]
        print(f"cat 제외: {sorted(drop)} → 유지 {[CAT_NAMES[i] for i in cat_keep]}")
    is_val = season == args.val_season
    cardinalities = [int(X_cat[:, j].max()) + 1 for j in range(X_cat.shape[1])]
    print(f"data: {X_num.shape} num, {X_cat.shape} cat, card={cardinalities}")
    print(f"train {int((~is_val).sum())} / val {int(is_val.sum())} (season {args.val_season})")

    Xn_tr = torch.as_tensor(X_num[~is_val])
    Xc_tr = torch.as_tensor(X_cat[~is_val])
    y_tr = torch.as_tensor(y[~is_val])
    Xn_va = torch.as_tensor(X_num[is_val])
    Xc_va = torch.as_tensor(X_cat[is_val])
    y_va_np = y[is_val]

    bins = None
    num_embeddings = None
    if args.num_emb == "ple":
        t0 = time.time()
        bins = rtdl_num_embeddings.compute_bins(Xn_tr, n_bins=args.n_bins)
        print(f"compute_bins :: {time.time() - t0:.1f}s")
        num_embeddings = {
            "type": "PiecewiseLinearEmbeddings",
            "d_embedding": args.d_embedding,
            "activation": False,
            "version": "B",
        }

    model = Model(
        n_num_features=X_num.shape[1],
        cat_cardinalities=cardinalities,
        n_classes=None,          # binary → 1 logit
        backbone={"type": "MLP", "n_blocks": args.n_blocks,
                  "d_block": args.d_block, "dropout": args.dropout},
        bins=bins,
        num_embeddings=num_embeddings,
        arch_type=args.arch,
        k=args.k,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M | device={DEVICE}")

    opt = torch.optim.AdamW(make_parameter_groups(model),
                            lr=args.lr, weight_decay=args.weight_decay)

    n_tr = len(y_tr)
    steps = math.ceil(n_tr / args.batch)
    best = {"score": -math.inf, "epoch": -1}
    bad_epochs = 0
    os.makedirs("out", exist_ok=True)
    log = {"args": vars(args), "epochs": []}

    @torch.no_grad()
    def predict_val():
        model.eval()
        preds = []
        for i in range(0, len(y_va_np), args.eval_batch):
            xn = Xn_va[i:i + args.eval_batch].to(DEVICE)
            xc = Xc_va[i:i + args.eval_batch].to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16,
                                enabled=DEVICE == "cuda"):
                out = model(xn, xc).squeeze(-1)      # (B, k)
            preds.append(torch.sigmoid(out.float()).mean(1).cpu())
        return torch.cat(preds).numpy()

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_tr)
        t0, tot_loss = time.time(), 0.0
        for s in range(steps):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            xn = Xn_tr[idx].to(DEVICE, non_blocking=True)
            xc = Xc_tr[idx].to(DEVICE, non_blocking=True)
            yb = y_tr[idx].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE, dtype=torch.bfloat16,
                                enabled=DEVICE == "cuda"):
                out = model(xn, xc).squeeze(-1)                  # (B, k)
                loss = F.binary_cross_entropy_with_logits(
                    out, yb[:, None].expand_as(out))
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)

        val_pred = predict_val()
        score = brier_skill_raw(y_va_np, val_pred)
        el = time.time() - t0
        print(f"epoch {epoch:3d} | loss {tot_loss / n_tr:.5f} "
              f"| val BSS(raw) {score:9.2f} | {el:.0f}s", flush=True)
        log["epochs"].append({"epoch": epoch, "loss": tot_loss / n_tr,
                              "val_bss": score, "sec": el})

        if score > best["score"]:
            best = {"score": score, "epoch": epoch}
            bad_epochs = 0
            torch.save({"state_dict": model.state_dict(),
                        "config": {
                            "n_num_features": X_num.shape[1],
                            "cat_cardinalities": cardinalities,
                            "cat_cols_kept": [CAT_NAMES[i] for i in cat_keep],
                            "backbone": {"type": "MLP", "n_blocks": args.n_blocks,
                                         "d_block": args.d_block,
                                         "dropout": args.dropout},
                            "num_embeddings": num_embeddings,
                            "arch_type": args.arch, "k": args.k,
                        },
                        "bins": [b.cpu() for b in bins] if bins else None,
                        "val_bss": score, "epoch": epoch},
                       f"out/tabm_{args.tag}_best.pt")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop @ {epoch} (best {best['score']:.2f} "
                      f"@ epoch {best['epoch']})")
                break

    log["best"] = best
    with open(f"out/tabm_{args.tag}_log.json", "w") as f:
        json.dump(log, f, indent=1)
    print(f"BEST val BSS {best['score']:.2f} @ epoch {best['epoch']} "
          f"→ out/tabm_{args.tag}_best.pt")


if __name__ == "__main__":
    main()
