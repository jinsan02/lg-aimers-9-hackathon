"""Rolling transfer gate for adding the masked-pitch NN before v11 corrections.

For each route, recent-middle and exact pitcher-batter residual maps are fitted
only on the source validation season and applied unchanged to the target
season.  This avoids judging the NN against corrections self-fitted on the
target whose apparent source gain is intentionally much larger than transfer.
"""

from __future__ import annotations

import argparse
from glob import glob

import numpy as np
import pandas as pd


SLOPE, SHIFT = 1.0416, 0.0052
MID = "asof_pitcher_prev5_game_middle_rate"


def load_one(path):
    z = np.load(path, allow_pickle=True)
    return z["pred"].astype(float), z["y"].astype(float)


def load_many(pattern):
    fs = sorted(glob(pattern))
    if not fs:
        raise FileNotFoundError(pattern)
    zs = [np.load(f) for f in fs]
    y = zs[0]["y"].astype(float)
    if any(not np.array_equal(z["y"], y) for z in zs[1:]):
        raise ValueError(f"target mismatch: {pattern}")
    return np.mean([z["pred"].astype(float) for z in zs], 0), y, len(fs)


def bss(y, p):
    r = y.mean()
    return 1e5*(1-np.mean((np.clip(p, 0, 1)-y)**2)/(r*(1-r)))


def post(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.clip(1/(1+np.exp(-SLOPE*np.log(p/(1-p))))-SHIFT, 0, 1)


def fit_apply(keys_s, keys_t, resid, k=500.0):
    tab = pd.DataFrame({"key": keys_s, "r": resid}).groupby("key").r.agg(
        ["sum", "size"])
    mp = (tab["sum"]/(tab["size"]+k)).to_dict()
    a = pd.Series(keys_s).map(mp).fillna(0).to_numpy(float)
    b = pd.Series(keys_t).map(mp).fillna(0).to_numpy(float)
    mean = float(a.mean())
    return a-mean, b-mean


def route(ps, pt, ys, src, tgt):
    ps, pt = post(ps), post(pt)
    xs, xt = src[MID].to_numpy(float), tgt[MID].to_numpy(float)
    good = xs[np.isfinite(xs)]
    edges = np.unique(np.quantile(good, np.linspace(0, 1, 9)))
    edges[0], edges[-1] = -np.inf, np.inf
    ms = np.searchsorted(edges[1:-1], xs, side="right")
    mt = np.searchsorted(edges[1:-1], xt, side="right")
    ms[~np.isfinite(xs)], mt[~np.isfinite(xt)] = -1, -1
    a_s, a_t = fit_apply(ms, mt, ys-ps)
    ps = np.clip(ps+a_s, 0, 1)
    pt = np.clip(pt+a_t, 0, 1)
    pair_s = (src.pitcher_id.astype(str)+"|"+src.batter_id.astype(str)).to_numpy()
    pair_t = (tgt.pitcher_id.astype(str)+"|"+tgt.batter_id.astype(str)).to_numpy()
    b_s, b_t = fit_apply(pair_s, pair_t, ys-ps)
    return np.clip(ps+b_s, 0, 1), np.clip(pt+b_t, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-val", required=True)
    ap.add_argument("--base-test", required=True)
    ap.add_argument("--nn-val", required=True)
    ap.add_argument("--nn-test", required=True)
    ap.add_argument("--source", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--drop-f-pre", type=int, default=0)
    ap.add_argument("--weight", type=float, default=0.10)
    args = ap.parse_args()

    bs, ys = load_one(args.base_val)
    bt, yt = load_one(args.base_test)
    ns, yns, n1 = load_many(args.nn_val)
    nt, ynt, n2 = load_many(args.nn_test)
    if not (np.array_equal(ys, yns) and np.array_equal(yt, ynt)):
        raise ValueError("base/NN target mismatch")
    cols = ["season", "game_month", "game_type", "pitcher_id", "batter_id", MID]
    d = pd.read_csv("data/train.csv", usecols=cols)
    if args.drop_f_pre:
        d = d[~((d.game_type == "F") & (d.season <= args.drop_f_pre))]
    src = d[d.season == args.source].reset_index(drop=True)
    tgt = d[d.season == args.target].reset_index(drop=True)
    if len(src) != len(ys) or len(tgt) != len(yt):
        raise ValueError(f"row mismatch {len(src)}/{len(tgt)} vs {len(ys)}/{len(yt)}")

    _, q0 = route(bs, bt, ys, src, tgt)
    w = args.weight
    _, q1 = route((1-w)*bs+w*ns, (1-w)*bt+w*nt, ys, src, tgt)
    masks = {
        "all": np.ones(len(tgt), bool),
        "R": tgt.game_type.eq("R").to_numpy(),
        "F": tgt.game_type.eq("F").to_numpy(),
        "early": tgt.game_month.le(6).to_numpy(),
        "late": tgt.game_month.gt(6).to_numpy(),
    }
    print(f"source={args.source} target={args.target} w={w:.3f} nn={n1}/{n2}")
    print(f"core base={bss(yt,bt):.3f} blend={bss(yt,(1-w)*bt+w*nt):.3f} "
          f"delta={bss(yt,(1-w)*bt+w*nt)-bss(yt,bt):+.3f}")
    for name, m in masks.items():
        before, after = bss(yt[m], q0[m]), bss(yt[m], q1[m])
        print(f"{name:<6} route {before:9.3f}->{after:9.3f} {after-before:+8.3f}")


if __name__ == "__main__":
    main()
