"""EXP-C -- does the shipped recent-middle lookup reproduce on new OOF?

The champion adds, after slope/shift, an 8-bin offset keyed on
`asof_pitcher_prev5_game_middle_rate`, hard-coded inline in
`src/script_blend_v11.py`. The lookup is **non-monotone** (bin 1 +0.00913 ->
bin 2 +0.00146, sign flip at 5->6) and v12 lost LB -18.271 by replacing it with
a career-middle version.

C-1 recomputes the bin residuals from OOF and compares them to the shipped
numbers. C-2 refits the lookup with the same shrinkage the generator uses
(`MID_K = 500`) and -- the part that decides -- freezes it onto a season it was
not fitted on. EXP-A established the prior: a map fitted on 2023 and frozen
onto 2024 lost 19.93 while the legacy constants lost 1.22.

  python tools/exp_c_middle.py B1J6           # fit val2023, apply to 2024
  python tools/exp_c_middle.py B1S --no-target
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from calib_refit import bss, legacy, load_core                  # noqa: E402

MID = "asof_pitcher_prev5_game_middle_rate"
MID_K = 500.0
# Shipped inline in script_blend_v11.py. Bin 1 is the lowest recent-middle rate.
LEGACY_OFF = np.array([+0.00912860, +0.00145823, +0.00166458, +0.00313052,
                       +0.00028883, -0.00547501, -0.00413855, -0.00507436])
LEGACY_NAN = -0.00810269


def bins_of(x, edges=None):
    """8 quantile bins, NaN -> -1. Edges are frozen when supplied."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if edges is None:
        edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, 9)))
        edges[0], edges[-1] = -np.inf, np.inf
    b = np.searchsorted(edges[1:-1], x, side="right")
    b[~ok] = -1
    return b, edges


def fit_lookup(y, p, b, k=MID_K):
    """Shrunk mean residual per bin, re-centred to leave the mean untouched."""
    r = y - p
    r = r - r.mean()
    t = pd.DataFrame({"b": b, "r": r}).groupby("b").r.agg(["sum", "size"])
    t["off"] = t["sum"] / (t["size"] + k)
    t["off"] -= float(np.average(t.off, weights=t["size"]))
    return t


def apply_lookup(p, b, tab):
    return np.clip(p + pd.Series(b).map(tab.off).fillna(0.0).to_numpy(), 0, 1)


def apply_legacy(p, b):
    off = np.where(b < 0, LEGACY_NAN, LEGACY_OFF[np.clip(b, 0, 7)])
    return np.clip(p + off, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--source", default="val")
    ap.add_argument("--target", default="test")
    ap.add_argument("--no-target", action="store_true")
    ap.add_argument("--k", type=float, default=MID_K)
    a = ap.parse_args()

    cols = ["row_id", MID]
    d = pd.read_csv("./data/train.csv", usecols=cols).set_index("row_id")

    def prep(split):
        p, y = load_core(a.tag, split)
        z = np.load(f"./out/cat_{a.tag}_base_s3_{split}_preds.npz", allow_pickle=True)
        x = d.reindex(z["row_id"])[MID].to_numpy(float)
        return legacy(p), y, x           # legacy() = the slope/shift the lookup sits after

    ps, ys, xs = prep(a.source)
    bs, edges = bins_of(xs)
    print(f"=== {a.tag} | C-1 legacy lookup vs OOF, source '{a.source}' "
          f"{len(ys):,} rows ===")
    print(f"  NaN share {float((bs < 0).mean()) * 100:.2f}%")
    tab = fit_lookup(ys, ps, bs, a.k)
    print(f"  {'bin':>4s} {'n':>9s} {'OOF offset':>12s} {'shipped':>12s} "
          f"{'diff':>10s}  sign")
    for i in range(-1, 8):
        if i not in tab.index:
            continue
        o = float(tab.off[i])
        s = LEGACY_NAN if i < 0 else float(LEGACY_OFF[i])
        agree = "same" if np.sign(o) == np.sign(s) else "**FLIP**"
        print(f"  {('NaN' if i < 0 else str(i + 1)):>4s} {int(tab['size'][i]):9,d} "
              f"{o:+12.8f} {s:+12.8f} {o - s:+10.8f}  {agree}")
    same = sum(1 for i in range(8) if i in tab.index
               and np.sign(tab.off[i]) == np.sign(LEGACY_OFF[i]))
    print(f"  sign agreement on the 8 bins: {same}/8")
    print(f"  monotone in OOF? {bool(np.all(np.diff([tab.off[i] for i in range(8) if i in tab.index]) <= 0)) or bool(np.all(np.diff([tab.off[i] for i in range(8) if i in tab.index]) >= 0))}")

    print(f"\n  source scores")
    print(f"    slope/shift only          {bss(ys, ps):8.2f}")
    print(f"    + shipped lookup          {bss(ys, apply_legacy(ps, bs)):8.2f}")
    print(f"    + OOF lookup (in-sample)  {bss(ys, apply_lookup(ps, bs, tab)):8.2f}   <- cannot lose")

    if a.no_target:
        print("\n  no target split: transfer untestable here.")
        return 0

    pt, yt, xt = prep(a.target)
    bt, _ = bins_of(xt, edges)                 # frozen edges, frozen table
    base = bss(yt, pt)
    print(f"\n=== C-2 frozen onto '{a.target}' {len(yt):,} rows ===")
    print(f"  slope/shift only            {base:8.2f}")
    print(f"  + shipped lookup            {bss(yt, apply_legacy(pt, bt)):8.2f}   "
          f"delta {bss(yt, apply_legacy(pt, bt)) - base:+7.2f}")
    q = apply_lookup(pt, bt, tab)
    print(f"  + OOF lookup frozen         {bss(yt, q):8.2f}   delta {bss(yt, q) - base:+7.2f}")
    to = fit_lookup(yt, pt, bt, a.k)
    print(f"  + oracle lookup on target   {bss(yt, apply_lookup(pt, bt, to)):8.2f}   "
          f"delta {bss(yt, apply_lookup(pt, bt, to)) - base:+7.2f}   <- ceiling, never a candidate")
    print("\n  target-season offsets vs source-season offsets")
    for i in range(-1, 8):
        if i in tab.index and i in to.index:
            f = "**FLIP**" if np.sign(tab.off[i]) != np.sign(to.off[i]) else "same"
            print(f"    {('NaN' if i < 0 else str(i + 1)):>4s} source {tab.off[i]:+.8f}  "
                  f"target {to.off[i]:+.8f}   {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
