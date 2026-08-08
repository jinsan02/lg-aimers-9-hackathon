"""Content-only pitcher/batter two-tower pilot for cold-start-safe matchups."""

import argparse
import time

import joblib
import numpy as np
import torch
import torch.nn as nn


def bss(y, p):
    r = y.mean()
    return float(1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / (r * (1-r))))


class Tower(nn.Module):
    def __init__(self, n_num, cards, pidx, bidx, dim=16, drop=.15):
        super().__init__()
        self.register_buffer("pidx", torch.tensor(pidx, dtype=torch.long))
        self.register_buffer("bidx", torch.tensor(bidx, dtype=torch.long))
        used = set(pidx) | set(bidx)
        cidx = [i for i in range(n_num) if i not in used]
        self.register_buffer("cidx", torch.tensor(cidx, dtype=torch.long))
        self.p = nn.Sequential(nn.Linear(len(pidx), 64), nn.SiLU(), nn.Dropout(drop),
                               nn.Linear(64, dim))
        self.b = nn.Sequential(nn.Linear(len(bidx), 64), nn.SiLU(), nn.Dropout(drop),
                               nn.Linear(64, dim))
        self.embs = nn.ModuleList([nn.Embedding(c, min(8, max(2, c // 2)))
                                   for c in cards])
        cw = len(cidx) + sum(e.embedding_dim for e in self.embs)
        self.ctx = nn.Sequential(nn.Linear(cw, 128), nn.BatchNorm1d(128), nn.SiLU(),
                                 nn.Dropout(drop), nn.Linear(128, 64), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(64 + dim * 4 + 1, 128), nn.SiLU(),
                                  nn.Dropout(drop), nn.Linear(128, 1))

    def forward(self, xn, xc):
        p = self.p(xn.index_select(1, self.pidx))
        b = self.b(xn.index_select(1, self.bidx))
        c = [xn.index_select(1, self.cidx)] + [e(xc[:, i]) for i, e in enumerate(self.embs)]
        c = self.ctx(torch.cat(c, 1))
        dot = (p * b).sum(1, keepdim=True) / np.sqrt(p.shape[1])
        z = torch.cat([c, p, b, p * b, torch.abs(p - b), dot], 1)
        return self.head(z).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--tag", default="TW_CONTENT")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    meta = joblib.load(args.meta)
    names = list(meta["num"])
    Xn, Xc = z["Xn"].astype(np.float32), z["Xc"].astype(np.int64)
    y, va = z["y"].astype(np.float32), z["is_val"].astype(bool)
    te = z["is_test"].astype(bool) if "is_test" in z else np.zeros_like(va)
    tr = ~(va | te)
    pidx = [i for i, n in enumerate(names) if "pitcher" in n]
    bidx = [i for i, n in enumerate(names) if "batter" in n]
    if not pidx or not bidx:
        raise ValueError(f"tower inputs missing: pitcher={len(pidx)} batter={len(bidx)}")

    med = np.nanmedian(Xn[tr], axis=0)
    Xn = np.where(np.isfinite(Xn), Xn, med)
    mu, sd = Xn[tr].mean(0), Xn[tr].std(0)
    Xn = ((Xn - mu) / np.where(sd > 1e-6, sd, 1)).astype(np.float32)
    cards = [int(Xc[tr, i].max()) + 2 for i in range(Xc.shape[1])]
    for i, c in enumerate(cards):
        Xc[:, i] = np.where(Xc[:, i] < c, Xc[:, i], c - 1)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    net = Tower(Xn.shape[1], cards, pidx, bidx).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    idx = np.flatnonzero(tr)
    steps = args.epochs * (len(idx) // args.bs + 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    xn = torch.from_numpy(Xn[tr]).to(dev)
    xc = torch.from_numpy(Xc[tr]).to(dev)
    yt = torch.from_numpy(y[tr]).to(dev)
    xnv = torch.from_numpy(Xn[va]).to(dev)
    xcv = torch.from_numpy(Xc[va]).to(dev)
    lossfn = nn.BCEWithLogitsLoss()

    def predict():
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(xnv), 65536):
                out.append(torch.sigmoid(net(xnv[i:i+65536], xcv[i:i+65536])).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    best, bestp, best_ep, t0 = -1e9, None, 1, time.time()
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(len(idx), device=dev)
        for i in range(0, len(idx), args.bs):
            j = perm[i:i+args.bs]
            opt.zero_grad(); loss = lossfn(net(xn[j], xc[j]), yt[j])
            loss.backward(); opt.step(); sch.step()
        p = predict(); score = bss(y[va].astype(float), p)
        print(f"ep{ep+1:02d} BSS {score:.2f}", flush=True)
        if score > best:
            best, bestp, best_ep = score, p, ep + 1
            torch.save({"state": net.state_dict(), "names": names, "pidx": pidx,
                        "bidx": bidx, "cards": cards, "mu": mu, "sd": sd, "med": med},
                       f"model/tower_{args.tag}_s{args.seed}.pt")
    np.savez_compressed(f"out/tower_{args.tag}_s{args.seed}_val_preds.npz",
                        y=y[va].astype(float), pred=bestp)
    print(f"[tower {args.tag}] best {best:.2f} | {time.time()-t0:.0f}s", flush=True)

    if te.any():
        # Same rolling-origin protocol as the other NN pilots: choose epoch on
        # validation, refit from scratch including it, then score unseen season.
        refit = tr | va
        nepoch = max(1, int(round(best_ep * 1.5)))
        torch.manual_seed(args.seed)
        final = Tower(Xn.shape[1], cards, pidx, bidx).to(dev)
        opt = torch.optim.AdamW(final.parameters(), lr=args.lr, weight_decay=1e-5)
        ridx = np.flatnonzero(refit)
        steps = nepoch * (len(ridx) // args.bs + 1)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
        xnr = torch.from_numpy(Xn[refit]).to(dev)
        xcr = torch.from_numpy(Xc[refit]).to(dev)
        yr = torch.from_numpy(y[refit]).to(dev)
        final.train()
        for _ in range(nepoch):
            perm = torch.randperm(len(ridx), device=dev)
            for i in range(0, len(ridx), args.bs):
                j = perm[i:i+args.bs]
                opt.zero_grad(); loss = lossfn(final(xnr[j], xcr[j]), yr[j])
                loss.backward(); opt.step(); sch.step()
        xnt = torch.from_numpy(Xn[te]).to(dev)
        xct = torch.from_numpy(Xc[te]).to(dev)
        final.eval(); pred = []
        with torch.no_grad():
            for i in range(0, len(xnt), 65536):
                pred.append(torch.sigmoid(final(xnt[i:i+65536], xct[i:i+65536]))
                            .cpu().numpy())
        pred = np.concatenate(pred).astype(np.float64)
        ytst = y[te].astype(np.float64)
        np.savez_compressed(f"out/tower_{args.tag}_s{args.seed}_test_preds.npz",
                            y=ytst, pred=pred)
        print(f"  -> unseen test BSS {bss(ytst, pred):.2f} | refit {nepoch} epochs",
              flush=True)


if __name__ == "__main__":
    main()
