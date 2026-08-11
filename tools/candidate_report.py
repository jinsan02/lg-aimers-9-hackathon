"""Generic same-surface candidate, segment, and fixed-blend report."""

import argparse
import glob
import os

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    z = [np.load(p) for p in paths]
    y = z[0]["y"].astype(float)
    if any(not np.array_equal(y, q["y"]) for q in z[1:]):
        raise ValueError(f"target mismatch: {pattern}")
    return np.mean([q["pred"].astype(float) for q in z], axis=0), y, len(paths)


def bss(y, p, center=False):
    p = np.asarray(p, float).copy()
    if center:
        p += y.mean()-p.mean()
    den = y.mean()*(1-y.mean())
    return 1e5*(1-np.mean((np.clip(p, 0, 1)-y)**2)/den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--cell", default="")
    ap.add_argument("--cell-weight", type=float, default=.55)
    args = ap.parse_args()
    pb, y, nb = load(args.base)
    pa, ya, na = load(args.alt)
    if not np.array_equal(y, ya):
        raise ValueError("base/alt target mismatch")
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                    usecols=["season", "game_month", "game_type"])
    d = d[d.season == args.season].reset_index(drop=True)
    if len(d) != len(y):
        raise ValueError(f"row mismatch {len(d)} vs {len(y)}")
    print(f"base n_models={nb} alt n_models={na} rows={len(y):,} "
          f"rms={np.sqrt(np.mean((pa-pb)**2)):.7f}")
    masks = {"all": np.ones(len(y), bool), "R": d.game_type.eq("R").to_numpy(),
             "F": d.game_type.eq("F").to_numpy(),
             "early": d.game_month.le(6).to_numpy(),
             "late": d.game_month.gt(6).to_numpy()}
    for name, m in masks.items():
        if m.sum() < 100:
            continue
        rb, ra = bss(y[m], pb[m]), bss(y[m], pa[m])
        cb, ca = bss(y[m], pb[m], True), bss(y[m], pa[m], True)
        print(f"{name:<6} raw {rb:9.3f}->{ra:9.3f} {ra-rb:+8.3f} | "
              f"centered {cb:9.3f}->{ca:9.3f} {ca-cb:+8.3f}")
    if args.cell:
        pc, yc, nc = load(args.cell)
        if not np.array_equal(y, yc):
            raise ValueError("cell target mismatch")
        w = args.cell_weight
        before, after = (1-w)*pb+w*pc, (1-w)*pa+w*pc
        print(f"\nfixed blend cell_models={nc} w={w:.3f}")
        for name, m in masks.items():
            if m.sum() < 100:
                continue
            rb, ra = bss(y[m], before[m], True), bss(y[m], after[m], True)
            print(f"{name:<6} centered {rb:9.3f}->{ra:9.3f} {ra-rb:+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
