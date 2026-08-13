"""Dacon's actual row-independence test: does a row's prediction depend on
which OTHER rows happen to be in test.csv?

The builder already reverses `test.csv` and checks nothing moves. That is
necessary but NOT sufficient, and the gap matters: an aggregate over all test
rows -- `test.groupby("pitcher_id")["x"].mean()`, a global quantile, a frequency
count -- is **invariant to row order**. It passes the reversal check and is a
disqualifying feature under the 2026-08-13 notice, which states the criterion
as:

    the prediction for a row must be identical whether test.csv holds that row
    alone or the whole evaluation set

So this tool varies the *set*, not the order. It runs the packaged script's
`blend()` on a full frame, then on subsets of it, and on single rows, and
compares the predictions for the rows they have in common.

The public `data/test.csv` is a 5-row sample, which is too small to expose a
per-player aggregate. So the frame is synthesised from train rows with the
target dropped -- same 48 columns, deliberately stacked so single pitchers and
batters appear many times. Nothing is fitted here; the rows are inputs only.

  python tools/audit_subset_independence.py --package out/_pkg
  python tools/audit_subset_independence.py --package <unzipped submission dir>

Exit 2 on any drift above the tolerance.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-12


def load_script(pkg):
    """Import the packaged script.py with its own directory on sys.path.

    The script resolves its model/ paths from its own __file__, so it must be
    imported from where it actually sits -- importing a copy would test a
    different artifact than the one that ships.
    """
    sys.path.insert(0, pkg)
    spec = importlib.util.spec_from_file_location(
        "shipped_script", os.path.join(pkg, "script.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_frame(n_pitchers, per_pitcher, seed=0):
    """Rows from train, target dropped, stacked by pitcher.

    Repetition is the point. A per-row transform gives the same answer however
    the rows are grouped; a groupby over the test frame does not, and it only
    diverges when a key appears a different number of times in the subset than
    in the whole. So every pitcher gets several rows, and the subsets below
    deliberately split those groups.
    """
    cols = pd.read_csv(os.path.join(ROOT, "data", "test.csv"),
                       encoding="utf-8-sig", nrows=1).columns.tolist()
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                     encoding="utf-8-sig")
    tr = tr[tr["season"] == tr["season"].max()]
    rng = np.random.default_rng(seed)
    counts = tr["pitcher_id"].value_counts()
    pool = counts[counts >= per_pitcher].index.to_numpy()
    picked = rng.choice(pool, size=min(n_pitchers, len(pool)), replace=False)
    parts = [tr[tr["pitcher_id"] == p].head(per_pitcher) for p in picked]
    df = pd.concat(parts, ignore_index=True)[cols].reset_index(drop=True)
    # The real evaluation frame is 2025. Keep that so the same asof/prior
    # branches run as on the server.
    df["season"] = 2025
    df["row_id"] = [f"TEST_{i:06d}" for i in range(len(df))]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True,
                    help="directory holding the packaged script.py and model/")
    ap.add_argument("--pitchers", type=int, default=300)
    ap.add_argument("--per-pitcher", type=int, default=12)
    a = ap.parse_args()

    pkg = os.path.abspath(a.package)
    mod = load_script(pkg)

    # blend() reloads every pkl on each call. Cache so the five frames below
    # cost one model load, not five. This changes nothing the script computes.
    import joblib
    mod.joblib.load = functools.lru_cache(maxsize=None)(joblib.load)

    full = make_frame(a.pitchers, a.per_pitcher)
    print(f"frame: {len(full):,} rows | {full['pitcher_id'].nunique():,} pitchers "
          f"| {full['batter_id'].nunique():,} batters")

    print("\n[1/2] full frame")
    ref = dict(zip(full["row_id"], mod.blend(full)))

    # Each case changes WHICH rows are present, never their order. Splitting a
    # pitcher's rows across cases is what makes a per-player aggregate differ.
    cases = {
        "half (every 2nd row)":      full.iloc[::2],
        "one pitcher's rows only":   full[full["pitcher_id"] == full["pitcher_id"].iloc[0]],
        "16 scattered rows":         full.iloc[[0, 5, 11, 100, 250, 999, 1500, 1777,
                                                2000, 2222, 2500, 2600, 3000, 3100,
                                                3333, 3400]],
        "single row (first)":        full.iloc[[0]],
        "single row (last)":         full.iloc[[-1]],
    }

    print("\n[2/2] subsets")
    worst, bad = 0.0, []
    for name, sub in cases.items():
        sub = sub.reset_index(drop=True)
        p = mod.blend(sub)
        d = max(abs(float(v) - ref[r]) for r, v in zip(sub["row_id"], p))
        worst = max(worst, d)
        flag = "OK " if d <= TOL else "DRIFT"
        print(f"  {flag} {name:<26} {len(sub):>6,} rows   max |delta| {d:.3e}")
        if d > TOL:
            bad.append(name)

    print(f"\nworst drift {worst:.3e} (tolerance {TOL:.0e})")
    if bad:
        print("NOT subset independent -- a prediction depends on which other "
              "rows are in test.csv:")
        for n in bad:
            print(f"  - {n}")
        return 2
    print("subset independent: every row's prediction is the same alone as in "
          "the full frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
