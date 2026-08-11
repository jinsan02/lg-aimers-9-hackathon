"""Paired and fixed-blend report for an equal per-seed core-model ensemble."""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_RE = re.compile(r"_s(\d+)_(?:val|test)_preds\.npz$")


def load_tag(tag, kind):
    out, y = {}, None
    for path in glob.glob(os.path.join(ROOT, "out", f"cat_{tag}_s*_{kind}_preds.npz")):
        match = SEED_RE.search(path)
        if not match:
            continue
        z = np.load(path)
        yy = z["y"].astype(float)
        if y is None:
            y = yy
        elif not np.array_equal(y, yy):
            raise ValueError(f"target mismatch: {path}")
        out[int(match.group(1))] = z["pred"].astype(float)
    if not out:
        raise FileNotFoundError(f"{tag} {kind}")
    return out, y


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
    ap.add_argument("--kind", choices=["val", "test"], required=True)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--cell-weight", type=float, default=.55)
    args = ap.parse_args()
    base, y = load_tag(args.base, args.kind)
    variants = [load_tag(tag, args.kind)[0] for tag in args.variants]
    common = sorted(set(base).intersection(*(set(v) for v in variants)))
    delta = []
    mixed = {}
    for seed in common:
        mixed[seed] = np.mean([base[seed]] + [v[seed] for v in variants], axis=0)
        delta.append(bss(y, mixed[seed]) - bss(y, base[seed]))
    delta = np.asarray(delta)
    se = delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else np.nan
    print(f"paired seeds={common} delta=" + ",".join(f"{d:+.3f}" for d in delta))
    print(f"mean={delta.mean():+.3f} SE={se:.3f} t={delta.mean()/se:+.3f}")

    cell, yc = load_tag(args.cell, args.kind)
    if not np.array_equal(y, yc):
        raise ValueError("cell target mismatch")
    pb = np.mean([base[seed] for seed in common], axis=0)
    pm = np.mean(list(mixed.values()), axis=0)
    pc = np.mean(list(cell.values()), axis=0)
    w = args.cell_weight
    before, after = (1-w)*pb+w*pc, (1-w)*pm+w*pc
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                    usecols=["season", "game_month", "game_type"])
    d = d[d.season == args.season].reset_index(drop=True)
    masks = {"all": np.ones(len(y), bool), "R": d.game_type.eq("R").to_numpy(),
             "F": d.game_type.eq("F").to_numpy(),
             "early": d.game_month.le(6).to_numpy(),
             "late": d.game_month.gt(6).to_numpy()}
    print(f"ensemble base={len(base)} mixed={len(mixed)} cell={len(cell)} "
          f"rms={np.sqrt(np.mean((after-before)**2)):.7f}")
    for name, mask in masks.items():
        s0, s1 = bss(y[mask], before[mask]), bss(y[mask], after[mask])
        print(f"{name:<6} {s0:9.3f}->{s1:9.3f} {s1-s0:+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
