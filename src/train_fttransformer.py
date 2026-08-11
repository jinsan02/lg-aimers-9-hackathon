"""Compact FT-Transformer pilot on the exact CatBoost feature NPZ.

Unlike the existing dense MLP/TabM pilots, each numerical/categorical field is
a token and self-attention learns cross-field interactions explicitly.  The
split is fixed by the NPZ: train<=2022, val=2023, unseen test=2024.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer


def bss(y, p):
    r = float(np.mean(y))
    return 1e5*(1-float(np.mean((p-y)**2))/(r*(1-r)))


class FTTransformer(nn.Module):
    def __init__(self, n_num, cards, d=32, layers=2, heads=4, drop=.1):
        super().__init__()
        self.num_w = nn.Parameter(torch.empty(n_num, d))
        self.num_b = nn.Parameter(torch.zeros(n_num, d))
        nn.init.normal_(self.num_w, std=.02)
        self.cat = nn.ModuleList([nn.Embedding(c, d) for c in cards])
        for e in self.cat:
            nn.init.normal_(e.weight, std=.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        block = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d*4, dropout=drop,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, xn, xc):
        nt = xn.unsqueeze(-1)*self.num_w.unsqueeze(0) + self.num_b.unsqueeze(0)
        ct = torch.stack([e(xc[:, j]) for j, e in enumerate(self.cat)], 1)
        cls = self.cls.expand(len(xn), -1, -1)
        h = self.encoder(torch.cat((cls, nt, ct), 1))[:, 0]
        return self.head(h).squeeze(1)


@torch.inference_mode()
def predict(net, xn, xc, idx, dev, bs=2048):
    net.eval()
    out = []
    for s in range(0, len(idx), bs):
        j = idx[s:s+bs]
        a = torch.from_numpy(xn[j]).to(dev)
        b = torch.from_numpy(xc[j]).to(dev)
        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            out.append(torch.sigmoid(net(a, b)).float().cpu().numpy())
    return np.concatenate(out)


def train_epochs(net, xn, xc, y, idx, dev, epochs, bs, lr, wd, seed,
                 val_idx=None, patience=2):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    rng = np.random.default_rng(seed)
    best, best_epoch, stale = None, 0, 0
    hist = []
    for ep in range(1, epochs+1):
        net.train()
        order = rng.permutation(idx)
        total = 0.0
        for s in range(0, len(order), bs):
            j = order[s:s+bs]
            a = torch.from_numpy(xn[j]).to(dev)
            b = torch.from_numpy(xc[j]).to(dev)
            t = torch.from_numpy(y[j]).to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                loss = loss_fn(net(a, b), t)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss)*len(j)
        msg = f"epoch {ep}/{epochs} train_bce={total/len(idx):.6f}"
        if val_idx is not None:
            pv = predict(net, xn, xc, val_idx, dev)
            vb = bss(y[val_idx], pv)
            msg += f" val_bss={vb:.3f}"
            hist.append((ep, vb))
            if best is None or vb > max(x[1] for x in hist[:-1]):
                best = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
                best_epoch, stale = ep, 0
            else:
                stale += 1
        print(msg, flush=True)
        if val_idx is not None and stale >= patience:
            break
    if best is not None:
        net.load_state_dict(best)
    return best_epoch or epochs, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="out/mtnn_pitch_2324.npz")
    ap.add_argument("--tag", default="FTT1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--drop", type=float, default=.1)
    ap.add_argument("--refit-mult", type=float, default=1.5)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    z = np.load(args.npz, allow_pickle=True)
    xn = z["Xn"].astype(np.float32)
    xc = z["Xc"].astype(np.int64)
    y = z["y"].astype(np.float32)
    va, te = z["is_val"], z["is_test"]
    tr = ~(va | te)
    it, iv, ie = np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te)
    print(f"device={dev} rows={len(it):,}/{len(iv):,}/{len(ie):,} "
          f"features={xn.shape[1]}+{xc.shape[1]}", flush=True)

    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             subsample=300_000, random_state=0)
    xn = np.nan_to_num(xn, nan=0., posinf=0., neginf=0.)
    qt.fit(xn[it])
    xn = qt.transform(xn).astype(np.float32)
    cards = [int(xc[:, j].max())+2 for j in range(xc.shape[1])]

    def make():
        return FTTransformer(xn.shape[1], cards, args.dim, args.layers,
                             args.heads, args.drop).to(dev)

    net = make()
    best_ep, hist = train_epochs(net, xn, xc, y, it, dev, args.epochs,
                                 args.batch_size, args.lr, args.wd, args.seed,
                                 val_idx=iv)
    pv = predict(net, xn, xc, iv, dev)
    print(f"best_epoch={best_ep} val_bss={bss(y[iv],pv):.3f}", flush=True)

    # Same frozen preprocessing, model refit through validation.
    del net
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    torch.manual_seed(args.seed)
    final = make()
    refit_ep = max(1, int(round(best_ep*args.refit_mult)))
    train_epochs(final, xn, xc, y, np.concatenate((it, iv)), dev, refit_ep,
                 args.batch_size, args.lr, args.wd, args.seed+10000)
    pt = predict(final, xn, xc, ie, dev)
    print(f"test_bss={bss(y[ie],pt):.3f} refit_epochs={refit_ep} "
          f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)

    os.makedirs("out", exist_ok=True)
    np.savez_compressed(f"out/ft_{args.tag}_s{args.seed}_val_preds.npz",
                        y=y[iv], pred=pv, row_id=z["row_id"][iv])
    np.savez_compressed(f"out/ft_{args.tag}_s{args.seed}_test_preds.npz",
                        y=y[ie], pred=pt, row_id=z["row_id"][ie])
    torch.save({"state_dict": {k:v.detach().cpu() for k,v in final.state_dict().items()},
                "args": vars(args), "cards": cards, "best_epoch": best_ep,
                "refit_epochs": refit_ep}, f"model/ft_{args.tag}_s{args.seed}.pt")


if __name__ == "__main__":
    main()
