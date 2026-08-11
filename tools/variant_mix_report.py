"""Report a fixed equal mixture of base variants under the current cell blend."""

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
    zs = [np.load(p) for p in paths]
    y = zs[0]["y"].astype(float)
    if any(not np.array_equal(y, z["y"]) for z in zs[1:]):
        raise ValueError(f"target mismatch within {pattern}")
    return np.mean([z["pred"].astype(float) for z in zs], axis=0), y


def bss(y, p):
    p = np.asarray(p, float).copy()
    p += y.mean() - p.mean()
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2)
                  / (y.mean() * (1 - y.mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--variants", required=True, nargs="+")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--cell-weight", type=float, default=.55)
    args = ap.parse_args()
    base, y = load(args.base)
    variants = []
    for pattern in args.variants:
        p, yy = load(pattern)
        if not np.array_equal(y, yy):
            raise ValueError(f"target mismatch: {pattern}")
        variants.append(p)
    cell, yy = load(args.cell)
    if not np.array_equal(y, yy):
        raise ValueError("cell target mismatch")
    mixed_base = np.mean([base] + variants, axis=0)
    w = args.cell_weight
    before = (1-w)*base + w*cell
    after = (1-w)*mixed_base + w*cell
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                    usecols=["season", "game_month", "game_type"])
    d = d[d.season == args.season].reset_index(drop=True)
    masks = {"all": np.ones(len(y), bool), "R": d.game_type.eq("R").to_numpy(),
             "F": d.game_type.eq("F").to_numpy(),
             "early": d.game_month.le(6).to_numpy(),
             "late": d.game_month.gt(6).to_numpy()}
    print(f"variants={len(variants)} base_mix_rms={np.sqrt(np.mean((mixed_base-base)**2)):.7f}")
    for name, mask in masks.items():
        s0, s1 = bss(y[mask], before[mask]), bss(y[mask], after[mask])
        print(f"{name:<6} {s0:9.3f}->{s1:9.3f} {s1-s0:+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
