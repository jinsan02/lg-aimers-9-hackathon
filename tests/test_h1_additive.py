"""Contract for the additive H1 arm.

The original H1 arm replaced `std_asof_pitcher_success_rate_delta` with
`h1_hand_delta` and parked at core +0.63 (base +3.88 SE 2.82, cell -1.55). That
arm changed two things at once: it dropped a column that was already a champion
feature and it added a new one. `--h1-additive` keeps the old column, so the
question becomes only "does the composed hand prior add anything".

Asserted before any GPU time, because "one mechanism" is a claim about the
feature list and the feature list is easy to get wrong:

  * additive really is additive -- nothing is dropped
  * the replacement path still drops, so the old arm is unchanged
  * the new column is not constant and not a copy of what it used to replace
  * the value of a row depends on that row alone, which is what makes it legal
    on test.csv

Run: python tests/test_h1_additive.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import fpipe                                                     # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def frame(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    seasons = rng.choice([2019, 2020, 2021, 2022, 2023, 2024], size=n)
    df = pd.DataFrame({
        "row_id": [f"TRAIN_{i:07d}" for i in range(n)],
        "season": seasons,
        "te_pitcher_batter_hand_ratio_dev": rng.normal(1.0, 0.08, n),
        "std_pitcher_n": rng.integers(0, 900, n).astype(float),
        fpipe.H1_DROP: rng.normal(0.0, 0.02, n),
    })
    # A few rows with no left/right history at all.
    df.loc[rng.choice(n, size=n // 50, replace=False),
           "te_pitcher_batter_hand_ratio_dev"] = np.nan
    prior = pd.DataFrame({fpipe.H1_PRIOR_COL: {s: 0.55 - 0.01 * (s - 2019)
                                               for s in range(2019, 2026)}})
    return df, prior


def main():
    print("H1 additive")
    df, prior = frame()

    add_df, add_cols, add_drop = fpipe._apply_h1(df, prior, additive=True)
    rep_df, rep_cols, rep_drop = fpipe._apply_h1(df, prior, additive=False)

    check("additive drops nothing", add_drop == [], f"dropped {add_drop}")
    check("replacement still drops the old delta", rep_drop == [fpipe.H1_DROP],
          f"dropped {rep_drop}")
    check("both add exactly one column",
          add_cols == ["h1_hand_delta"] == rep_cols, f"{add_cols} / {rep_cols}")
    check("the old delta survives additively",
          fpipe.H1_DROP in add_df.columns and
          np.allclose(add_df[fpipe.H1_DROP], df[fpipe.H1_DROP]))
    check("the new column is identical either way",
          np.allclose(add_df["h1_hand_delta"], rep_df["h1_hand_delta"]),
          "additive must change the feature list, not the values")

    v = add_df["h1_hand_delta"].to_numpy(np.float64)
    check("new column is finite everywhere", np.isfinite(v).all())
    check("new column is not constant", v.std() > 0, f"sd {v.std():.6g}")
    check("new column is not a copy of the one it replaces",
          not np.allclose(v, df[fpipe.H1_DROP].to_numpy(np.float64)),
          f"pearson {np.corrcoef(v, df[fpipe.H1_DROP])[0, 1]:+.4f}")

    # No hand history must mean no adjustment, not an invented one.
    miss = df["te_pitcher_batter_hand_ratio_dev"].isna().to_numpy()
    check("rows with no left/right history get exactly 0",
          np.all(v[miss] == 0.0), f"{miss.sum():,} such rows")

    # Row independence. The value uses the row's own dev and n plus a frozen
    # per-season constant, so a row scored alone must match a row scored in
    # company -- the property test.csv is judged on.
    sub = np.arange(0, len(df), 7)
    one, _, _ = fpipe._apply_h1(df.iloc[sub].reset_index(drop=True), prior,
                                additive=True)
    check("a subset scores identically",
          np.allclose(one["h1_hand_delta"].to_numpy(), v[sub], equal_nan=True))
    single, _, _ = fpipe._apply_h1(df.iloc[[0]].copy(), prior, additive=True)
    check("a single row scores identically",
          float(single["h1_hand_delta"].iloc[0]) == v[0])

    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
