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


# Keyed by n, not by degrees of freedom: T975[6] is t(.975, df=5) = 2.571.
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def stats(name, d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = d.mean() / se if se else float("nan")
    crit = T975.get(n, 1.96)
    lo, hi = d.mean() - crit * se, d.mean() + crit * se
    print(f"\n{name}")
    print(f"  per-seed {np.round(d, 2)}")
    print(f"  mean {d.mean():+.3f}  SE {se:.3f}  t {t:+.2f}  "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]")
    # Heavy tails were what made the legacy headline hard to read, so the
    # stability diagnostics print alongside -- they do not change the rule.
    print(f"  median {np.median(d):+.3f}  trimmed {np.sort(d)[1:-1].mean():+.3f}  "
          f"positive {int((d > 0).sum())}/{n}")
    ok = d.mean() >= 3 and t >= 2.4 and n >= 6
    print(f"  adoption: delta>=+3 {'Y' if d.mean() >= 3 else 'N'}  "
          f"t>=2.4 {'Y' if t >= 2.4 else 'N'}  n>=6 {'Y' if n >= 6 else 'N'}  "
          f"95% upper>=+3 {'Y' if hi >= 3 else 'N (reject)'}  "
          f"-> {'KEEP' if ok else ('DROP' if hi < 3 else 'PARK')}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ref", required=True, help="the champion's cell arm")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--control", default=None,
                    help="a re-run of --ref under a different tag. When given, "
                         "cand-vs-control is reported too and is the cleaner "
                         "mechanism estimate: both sides were trained in the "
                         "same session, so anything that drifted between the "
                         "champion's run and today cancels.")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()

    tags = {"base": a.base, "ref": a.ref, "cand": a.cand}
    if a.control:
        tags["control"] = a.control
    got = {k: load(v, a.split) for k, v in tags.items()}
    seeds = sorted(set.intersection(*[set(v) for v in got.values()]), key=int)
    if len(seeds) < 2:
        raise SystemExit("need paired seeds; got "
                         + "  ".join(f"{k} {sorted(v)}" for k, v in got.items()))

    # Same rows, same targets, or the pairing is fiction.
    y0 = got["base"][seeds[0]][1]
    for k, d in got.items():
        for s in seeds:
            if len(d[s][1]) != len(y0) or not np.array_equal(d[s][1], y0):
                raise SystemExit(f"{k} seed {s} disagrees with the base on y -- "
                                 f"different rows, the comparison is void")

    print(f"base {a.base} held fixed | {len(seeds)} paired seeds {seeds} | "
          f"{len(y0):,} rows")

    def core(tag, s):
        return bss(debias((1 - W_CELL) * got["base"][s][0]
                          + W_CELL * got[tag][s][0]), y0)

    def cell(tag, s):
        return bss(debias(got[tag][s][0]), y0)

    cols = ["ref", "control", "cand"] if a.control else ["ref", "cand"]
    hdr = {"ref": a.ref, "control": a.control, "cand": a.cand}
    print("\ncell arm alone")
    print("  " + "".join(f"{'seed':>5}" if i == 0 else "" for i in [0])
          + "".join(f"{hdr[c][:12]:>13}" for c in cols))
    for s in seeds:
        print(f"  {s:>5}" + "".join(f"{cell(c, s):>13.2f}" for c in cols))
    print("\ncore (0.45 base + 0.55 cell)")
    print("   seed" + "".join(f"{hdr[c][:12]:>13}" for c in cols))
    for s in seeds:
        print(f"  {s:>5}" + "".join(f"{core(c, s):>13.2f}" for c in cols))

    print("\n" + "=" * 70)
    print(f"A.  {a.cand} vs {a.ref}   (challenger vs the champion's own run)")
    print("=" * 70)
    stats("core", [core("cand", s) - core("ref", s) for s in seeds])
    stats("cell arm alone", [cell("cand", s) - cell("ref", s) for s in seeds])

    if a.control:
        print("\n" + "=" * 70)
        print(f"B.  {a.cand} vs {a.control}   (challenger vs a same-session re-run)")
        print("    the mechanism estimate: run-to-run drift cancels on both sides")
        print("=" * 70)
        stats("core", [core("cand", s) - core("control", s) for s in seeds])
        stats("cell arm alone", [cell("cand", s) - cell("control", s) for s in seeds])

        print("\n" + "=" * 70)
        print(f"C.  {a.control} vs {a.ref}   (re-run variation -- NOT a result)")
        print("    same command, same seeds, different invocation. This is the")
        print("    scale against which A and B have to be read, and it is")
        print("    specific to this machine, surface, cell family and seed set.")
        print("=" * 70)
        stats("core", [core("control", s) - core("ref", s) for s in seeds])
        stats("cell arm alone", [cell("control", s) - cell("ref", s) for s in seeds])
    return 0


if __name__ == "__main__":
    sys.exit(main())
