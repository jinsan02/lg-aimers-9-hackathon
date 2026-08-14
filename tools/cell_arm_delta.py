"""Paired cell-arm comparison with the base family held fixed.

`--fm-legacy-shift` and `--fm-noise-rate` are consumed inside
`if args.failmode_cells`, so neither can reach a depth-8 binary run. Re-running
the base arm for such a candidate therefore measures nothing but GPU
nondeterminism -- and then folds it into `core`.

That is not hypothetical. LEGLBL's base arm is an identical command with a
flag that is a no-op for it, and it still moved:

    seed      3      4      5      6      8     13
    base  -0.02  +2.62  -0.28  +0.95  -0.03  +0.19      mean +0.57
    cell  +0.30 +10.73  +3.11 +13.40  -0.60  +2.13      mean +4.85

so the +3.97 core headline carries a base contribution that has nothing to do
with labels -- and the two seeds that carry the cell effect are the same two
where the base drifted most (r = .75, n = 6).

Holding one base family fixed removes that term exactly: the same base
predictions enter both sides of every pair, so the difference is the cell arm's
alone.

  python tools/cell_arm_delta.py --base B1S_base --ref B1S_cell --cand LEGLBL_cell
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

W_CELL = 0.55
SHIFT, SLOPE = 0.0052, 1.0416


def seed_of(path):
    return path.rsplit("_s", 1)[1].split("_", 1)[0]


def load(tag, split="val"):
    out = {}
    for p in sorted(glob.glob(f"./out/*_{tag}_s*_{split}_preds.npz")):
        z = np.load(p, allow_pickle=True)
        out[seed_of(p)] = (z["pred"].astype(np.float64), z["y"].astype(np.float64),
                           z["row_id"] if "row_id" in z.files else None)
    return out


def bss(p, y):
    r = y.mean()
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def debias(p):
    """The shipped post-processing, so the number is on the LB coordinate."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.clip(1 / (1 + np.exp(-SLOPE * np.log(q / (1 - q)))) - SHIFT, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()

    B, R, C = (load(t, a.split) for t in (a.base, a.ref, a.cand))
    seeds = sorted(set(B) & set(R) & set(C), key=int)
    if len(seeds) < 2:
        raise SystemExit(f"need paired seeds; got base {sorted(B)} ref {sorted(R)} "
                         f"cand {sorted(C)}")

    # Same rows, same targets, or the pairing is fiction.
    y0 = B[seeds[0]][1]
    for tag, d in (("base", B), ("ref", R), ("cand", C)):
        for s in seeds:
            if len(d[s][1]) != len(y0) or not np.array_equal(d[s][1], y0):
                raise SystemExit(f"{tag} seed {s} disagrees with the base on y -- "
                                 f"different rows, the comparison is void")

    print(f"base {a.base} held fixed | {len(seeds)} paired seeds {seeds} | "
          f"{len(y0):,} rows")
    print(f"\n{'seed':>5} {'ref':>10} {'cand':>10} {'delta':>9}   "
          f"{'cell-only ref':>13} {'cand':>10} {'delta':>9}")
    core_d, cell_d = [], []
    for s in seeds:
        b = B[s][0]
        cr = (1 - W_CELL) * b + W_CELL * R[s][0]
        cc = (1 - W_CELL) * b + W_CELL * C[s][0]
        r_, c_ = bss(debias(cr), y0), bss(debias(cc), y0)
        rc, cc_ = bss(debias(R[s][0]), y0), bss(debias(C[s][0]), y0)
        core_d.append(c_ - r_)
        cell_d.append(cc_ - rc)
        print(f"{s:>5} {r_:10.2f} {c_:10.2f} {c_ - r_:+9.2f}   "
              f"{rc:13.2f} {cc_:10.2f} {cc_ - rc:+9.2f}")

    for name, d in (("core (0.45 base + 0.55 cell)", core_d), ("cell arm alone", cell_d)):
        d = np.asarray(d)
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se else float("nan")
        lo, hi = d.mean() - 2.571 * se, d.mean() + 2.571 * se     # t(.975, 5)
        print(f"\n{name}")
        print(f"  mean {d.mean():+.3f}  SE {se:.3f}  t {t:+.2f}  "
              f"95% CI [{lo:+.2f}, {hi:+.2f}]")
        # Heavy tails were what made the legacy headline hard to read, so the
        # stability diagnostics print alongside -- they do not change the rule.
        print(f"  median {np.median(d):+.3f}  trimmed {np.sort(d)[1:-1].mean():+.3f}  "
              f"positive {int((d > 0).sum())}/{len(d)}")
        print(f"  adoption: delta>=+3 {'Y' if d.mean() >= 3 else 'N'}  "
              f"t>=2.4 {'Y' if t >= 2.4 else 'N'}  "
              f"95% upper>=+3 {'Y' if hi >= 3 else 'N (reject)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
