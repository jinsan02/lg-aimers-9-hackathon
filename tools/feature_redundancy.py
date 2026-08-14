"""How much of the 121 is information, and how much is a precomputed shortcut.

The transfer map permutes one family at a time and leaves the rest intact. That
is the right scheme, but it means a family that is an exact function of columns
still sitting in the frame cannot show its information leaving -- the
information never left. Its number measures only how much the fitted trees
happened to route through the convenience column.

Spot checks on the 2024 frame already found four such constructions, all exact:

    asof_*_shr            = (rate*n + p*k) / (n + k), k = 200   resid 1e-15
    std_*_delta           = std_rate - asof_rate                resid 0
    te_*_ratio_dev        = te_level / te_pitcher_ratio         resid 0
    skill_pc_hat_vs_std   = skill_pc_hat - std_asof_..._rate    resid 0

This finds the rest by brute force rather than by eye: every numeric feature is
regressed on all the other numeric features (ridge, standardised, on a
subsample), and R^2 is reported. R^2 ~ 1 means the column adds no numeric
information to the frame it sits in.

That is emphatically **not** an argument for deleting anything -- an actual
retrain that dropped the delta columns cost -16.22, and a shortcut the tree
cannot rebuild with axis-parallel splits is worth its slot. The point is only to
mark which rows of the transfer map may be read as "this information stopped
transferring" and which may not.

  python tools/feature_redundancy.py --season 2024
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fpipe                                                     # noqa: E402
import feature_families as ff                                    # noqa: E402

TARGET = "control_success"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/cat_B1S_base_s3.pkl")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--sample", type=int, default=60_000)
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--out", default="out/feature_redundancy.csv")
    args = ap.parse_args()

    pack = joblib.load(os.path.join(ROOT, args.model))
    feats = pack["features"]
    ff.validate(feats)
    header = list(pd.read_csv(os.path.join(ROOT, "data", "test.csv"), nrows=0,
                              encoding="utf-8-sig").columns)
    full = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                       encoding="utf-8-sig", usecols=header + [TARGET])
    fr = fpipe.transform(full[full["season"] == args.season]
                         .reset_index(drop=True), pack["fpipe"])

    num = [c for c in feats if c not in pack["cat_cols"]
           and pd.api.types.is_numeric_dtype(fr[c])]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(fr), size=min(args.sample, len(fr)), replace=False)
    M = fr.loc[idx, num].astype(np.float64).to_numpy()
    # A missing value is itself information a tree reads natively; filling with
    # the column median is the conservative choice here, because it can only
    # make a column look *less* predictable, never more.
    med = np.nanmedian(M, axis=0)
    M = np.where(np.isfinite(M), M, med)
    mu, sd = M.mean(0), M.std(0)
    sd[sd == 0] = 1.0
    Z = (M - mu) / sd
    n, p = Z.shape
    print(f"{args.season}: {n:,} sampled rows, {p} numeric features "
          f"({len(feats) - p} categorical/held out)")

    # One Gram matrix, then leave-one-column-out by solving the reduced system.
    G = Z.T @ Z + args.ridge * n * np.eye(p)
    rows = []
    for j in range(p):
        ss = float(Z[:, j] @ Z[:, j])
        if ss == 0.0:
            # constant inside this season -- `season` itself. Nothing to
            # predict, and the transfer map already notes it contributes
            # nothing to a within-season permutation.
            rows.append(dict(feature=num[j], family=ff.FAMILY_OF[num[j]],
                             axis=ff.axis_of(num[j]), kind=ff.kind_of(num[j]),
                             r2=np.nan, rmse_sd=0.0))
            continue
        k = [i for i in range(p) if i != j]
        beta = np.linalg.solve(G[np.ix_(k, k)], Z[:, k].T @ Z[:, j])
        resid = Z[:, j] - Z[:, k] @ beta
        r2 = 1.0 - float(resid @ resid) / ss
        rows.append(dict(feature=num[j], family=ff.FAMILY_OF[num[j]],
                         axis=ff.axis_of(num[j]), kind=ff.kind_of(num[j]),
                         r2=round(r2, 6),
                         rmse_sd=round(float(np.sqrt(resid @ resid / n)), 6)))
    t = pd.DataFrame(rows).sort_values("r2", ascending=False)

    t["class"] = np.where(t.r2.isna(), "CONSTANT",
                  np.where(t.r2 >= 0.9999, "EXACT",
                  np.where(t.r2 >= 0.99, "NEAR_EXACT",
                  np.where(t.r2 >= 0.90, "MOSTLY", "INFORMATIVE"))))
    print("\nper feature, most redundant first (top 30):")
    print(t.head(30).to_string(index=False))
    print("\nby family:")
    g = (t.groupby("family")
          .agg(n=("r2", "size"), median_r2=("r2", "median"),
               exact=("class", lambda s: (s == "EXACT").sum()),
               near=("class", lambda s: (s == "NEAR_EXACT").sum()),
               informative=("class", lambda s: (s == "INFORMATIVE").sum()))
          .sort_values("median_r2", ascending=False))
    print(g.round(4).to_string())
    print(f"\nEXACT {int((t['class'] == 'EXACT').sum())} / NEAR_EXACT "
          f"{int((t['class'] == 'NEAR_EXACT').sum())} / MOSTLY "
          f"{int((t['class'] == 'MOSTLY').sum())} / INFORMATIVE "
          f"{int((t['class'] == 'INFORMATIVE').sum())} of {p}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
