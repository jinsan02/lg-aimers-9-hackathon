"""이산 토큰 트랜스포머(DTT) — 47개 피처 토큰에 attention.

docs/discrete_tokenization.md 설계 구현.
  입력: tok_v1.npz의 [N,47] 토큰 ID (컬럼 순서 고정 → 위치가 곧 컬럼)
  구조: 값 임베딩 + 컬럼(위치) 임베딩 → TransformerEncoder → CLS → 확률

실행(A100/4070): python src/train_dtt.py --tag d1 --seed 0
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def bss_raw(y, p):
    r = y.mean()
    return float(100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r))))


class DTT(nn.Module):
    def __init__(self, vocab, n_cols, d_model=192, n_layers=4, n_heads=8,
                 dropout=0.1):
        super().__init__()
        self.val_emb = nn.Embedding(vocab, d_model)
        self.col_emb = nn.Embedding(n_cols, d_model)      # 컬럼 정체성
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.val_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.col_emb.weight, std=0.02)
        self.register_buffer("cols", torch.arange(n_cols), persistent=False)

    def forward(self, x):                       # x: (B, n_cols) int
        h = self.val_emb(x) + self.col_emb(self.cols)[None]
        h = torch.cat([self.cls.expand(len(x), -1, -1), h], dim=1)
        h = self.enc(h)
        return self.head(self.norm(h[:, 0])).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/tok_v1.npz")
    ap.add_argument("--tag", default="d1")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--eval-batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.data)
    X, y, season = d["X"].astype(np.int64), d["y"].astype(np.float32), d["season"]
    vocab, n_cols = int(X.max()) + 1, X.shape[1]
    is_val = season == 2024
    Xtr = torch.as_tensor(X[~is_val])
    ytr = torch.as_tensor(y[~is_val])
    Xva = torch.as_tensor(X[is_val])
    yva = y[is_val].astype(np.float64)
    print(f"vocab={vocab} cols={n_cols} | train {len(ytr)} val {len(yva)}")

    model = DTT(vocab, n_cols, args.d_model, args.layers, args.heads,
                args.dropout).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"params {n_p / 1e6:.2f}M | device {DEVICE}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps = math.ceil(len(ytr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps * args.epochs, pct_start=0.1)

    @torch.no_grad()
    def predict():
        model.eval()
        out = []
        for i in range(0, len(yva), args.eval_batch):
            xb = Xva[i:i + args.eval_batch].to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16,
                                enabled=DEVICE == "cuda"):
                out.append(torch.sigmoid(model(xb).float()).cpu())
        return torch.cat(out).numpy().astype(np.float64)

    best = {"score": -math.inf, "epoch": -1}
    bad, log = 0, []
    os.makedirs("out", exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        t0, tot = time.time(), 0.0
        for s in range(steps):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            xb, yb = Xtr[idx].to(DEVICE), ytr[idx].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE, dtype=torch.bfloat16,
                                enabled=DEVICE == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        p = predict()
        sc = bss_raw(yva, p)
        print(f"epoch {ep:3d} | loss {tot / len(ytr):.5f} | val BSS {sc:9.2f} "
              f"| {time.time() - t0:.0f}s", flush=True)
        log.append({"epoch": ep, "loss": tot / len(ytr), "bss": sc})
        if sc > best["score"]:
            best = {"score": sc, "epoch": ep}
            bad = 0
            torch.save({"state_dict": model.state_dict(),
                        "config": {"vocab": vocab, "n_cols": n_cols,
                                   "d_model": args.d_model, "layers": args.layers,
                                   "heads": args.heads, "dropout": args.dropout},
                        "val_bss": sc, "epoch": ep},
                       f"out/dtt_{args.tag}_best.pt")
            np.savez_compressed(f"out/dtt_{args.tag}_val_preds.npz", y=yva, pred=p)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ {ep} (best {best['score']:.2f} @ {best['epoch']})")
                break
    with open(f"out/dtt_{args.tag}_log.json", "w") as f:
        json.dump({"args": vars(args), "log": log, "best": best}, f, indent=1)
    print(f"BEST {best['score']:.2f} @ epoch {best['epoch']}")


if __name__ == "__main__":
    main()
