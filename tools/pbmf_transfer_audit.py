"""Rolling transfer audit for outcome-aware pitcher-batter factorization.

K=0 reproduces the frozen recent-middle + exact-PB route.  K=4 adds a
rank-four SVD of the source-year shrunk pair residual matrix, then refits the
exact-pair residual.  Only source-year labels are used.
"""

from glob import glob
import argparse
import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W_CELL, SLOPE, SHIFT = .55, 1.0416, .0052
MID = "asof_pitcher_prev5_game_middle_rate"
PAIR_K, RANK = 500.0, 4


def ensemble(tag, kind):
    fs = sorted(glob(os.path.join(ROOT, "out", f"cat_{tag}_s*_{kind}_preds.npz")))
    plain = os.path.join(ROOT, "out", f"cat_{tag}_{kind}_preds.npz")
    if not fs and os.path.exists(plain):
        fs = [plain]
    if not fs:
        raise FileNotFoundError(f"{tag} {kind}")
    zs = [np.load(f) for f in fs]
    y = zs[0]["y"].astype(np.float64)
    if any(not np.array_equal(z["y"], y) for z in zs[1:]):
        raise ValueError(f"target mismatch within {tag}/{kind}")
    return np.mean([z["pred"].astype(np.float64) for z in zs], axis=0), y, len(zs)


def post(p):
    q = np.clip(p, 1e-6, 1-1e-6)
    return np.clip(1/(1+np.exp(-SLOPE*np.log(q/(1-q))))-SHIFT, 0, 1)


def bss(y, p):
    r = float(np.mean(y))
    return 1e5*(1-float(np.mean((np.clip(p, 0, 1)-y)**2))/(r*(1-r)))


def fit_map(ks, kt, resid, k):
    tab = pd.DataFrame({"key": ks, "r": resid}).groupby("key").r.agg(["sum", "size"])
    mp = (tab["sum"]/(tab["size"]+k)).to_dict()
    src = pd.Series(ks).map(mp).fillna(0).to_numpy(float)
    tgt = pd.Series(kt).map(mp).fillna(0).to_numpy(float)
    mean = float(src.mean())
    return src-mean, tgt-mean


def middle_keys(xs, xt):
    good = xs[np.isfinite(xs)]
    edges = np.unique(np.quantile(good, np.linspace(0, 1, 9)))
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.searchsorted(edges[1:-1], xs, side="right")
    b = np.searchsorted(edges[1:-1], xt, side="right")
    a[~np.isfinite(xs)], b[~np.isfinite(xt)] = -1, -1
    return a, b


def pair_keys(d):
    return (d.pitcher_id.astype(str)+"|"+d.batter_id.astype(str)).to_numpy()


def lowrank_pair(src, tgt, resid):
    pids = pd.Index(src.pitcher_id.unique())
    bids = pd.Index(src.batter_id.unique())
    pi = pids.get_indexer(src.pitcher_id)
    bi = bids.get_indexer(src.batter_id)
    tab = pd.DataFrame({"pi": pi, "bi": bi, "r": resid}).groupby(["pi", "bi"]).r.agg(["sum", "size"]).reset_index()
    val = tab["sum"].to_numpy(float)/(tab["size"].to_numpy(float)+PAIR_K)
    mat = coo_matrix((val, (tab.pi, tab.bi)), shape=(len(pids), len(bids))).tocsr()
    u, s, vt = svds(mat, k=RANK, which="LM", random_state=0)
    order = np.argsort(s)[::-1]
    u, s, vt = u[:, order], s[order], vt[order]
    left, right = u*np.sqrt(s), vt.T*np.sqrt(s)

    def apply(d):
        ip, ib = pids.get_indexer(d.pitcher_id), bids.get_indexer(d.batter_id)
        ok = (ip >= 0) & (ib >= 0)
        out = np.zeros(len(d), float)
        out[ok] = np.sum(left[ip[ok]]*right[ib[ok]], axis=1)
        return out, ok

    a, _ = apply(src)
    b, known = apply(tgt)
    mean = float(a.mean())
    return a-mean, b-mean, known


def report(label, frame, y, before, after):
    masks = {
        "all": np.ones(len(frame), bool),
        "R": frame.game_type.to_numpy() == "R",
        "F": frame.game_type.to_numpy() == "F",
        "early": frame.game_month.to_numpy() <= 6,
        "late": frame.game_month.to_numpy() > 6,
    }
    vals = []
    for name, m in masks.items():
        if m.sum() < 100:
            continue
        vals.append(f"{name}={bss(y[m],after[m])-bss(y[m],before[m]):+.3f}")
    print(f"{label:<18} " + " ".join(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--source", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--drop-f-pre", type=int, default=0)
    args = ap.parse_args()

    bv, ys, nb = ensemble(args.base, "val")
    cv, ysc, nc = ensemble(args.cell, "val")
    bt, yt, _ = ensemble(args.base, "test")
    ct, ytc, _ = ensemble(args.cell, "test")
    if not (np.array_equal(ys, ysc) and np.array_equal(yt, ytc)):
        raise ValueError("base/cell target mismatch")
    ps, pt = post((1-W_CELL)*bv+W_CELL*cv), post((1-W_CELL)*bt+W_CELL*ct)

    cols = ["season", "game_month", "game_type", "pitcher_id", "batter_id", MID]
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    if args.drop_f_pre:
        d = d[~((d.game_type == "F") & (d.season <= args.drop_f_pre))]
    src = d[d.season == args.source].reset_index(drop=True)
    tgt = d[d.season == args.target].reset_index(drop=True)
    if len(src) != len(ys) or len(tgt) != len(yt):
        raise ValueError(f"row mismatch data={len(src)}/{len(tgt)} pred={len(ys)}/{len(yt)}")

    ms, mt = middle_keys(src[MID].to_numpy(float), tgt[MID].to_numpy(float))
    r0 = ys-ps
    r0 -= r0.mean()
    ma, mb = fit_map(ms, mt, r0, 500.)
    r1 = ys-(ps+ma)
    r1 -= r1.mean()
    pk_s, pk_t = pair_keys(src), pair_keys(tgt)
    ea, eb = fit_map(pk_s, pk_t, r1, PAIR_K)
    k0s, k0t = ps+ma+ea, pt+mb+eb

    la, lb, known_nodes = lowrank_pair(src, tgt, r1)
    r2 = ys-(ps+ma+la)
    r2 -= r2.mean()
    xa, xb = fit_map(pk_s, pk_t, r2, PAIR_K)
    k4s, k4t = ps+ma+la+xa, pt+mb+lb+xb

    exact = pd.Series(pk_t).isin(set(pk_s)).to_numpy()
    src_p, src_b = set(src.pitcher_id), set(src.batter_id)
    cold = (~tgt.pitcher_id.isin(src_p) | ~tgt.batter_id.isin(src_b)).to_numpy()
    new_pair = known_nodes & ~exact
    print(f"surface {args.source}->{args.target} base={args.base}({nb}) cell={args.cell}({nc}) "
          f"rows={len(src):,}/{len(tgt):,}")
    print(f"coverage exact={exact.mean():.1%} known-new-pair={new_pair.mean():.1%} cold={cold.mean():.1%}")
    report("K0 vs base", tgt, yt, pt, k0t)
    report("K4 vs K0", tgt, yt, k0t, k4t)
    for name, m in (("exact", exact), ("known-new", new_pair), ("cold", cold)):
        if m.sum() >= 100:
            print(f"  K4-K0 {name:<9} n={m.sum():>7,} delta={bss(yt[m],k4t[m])-bss(yt[m],k0t[m]):+.3f}")
    print(f"source diagnostic K0={bss(ys,k0s)-bss(ys,ps):+.3f} "
          f"K4-K0={bss(ys,k4s)-bss(ys,k0s):+.3f} lowrank_sd={la.std():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
