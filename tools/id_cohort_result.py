"""Segment a candidate-vs-base prediction delta by row-local player cohorts."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def load(path: str) -> pd.DataFrame:
    z = np.load(path, allow_pickle=True)
    return pd.DataFrame({"row_id": z["row_id"].astype(str), "y": z["y"],
                         "pred": z["pred"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("candidate")
    args = ap.parse_args()
    a = load(args.base).rename(columns={"pred": "base", "y": "y_base"})
    b = load(args.candidate).rename(columns={"pred": "cand", "y": "y_cand"})
    p = a.merge(b, on="row_id", validate="one_to_one")
    d = pd.read_csv("data/train.csv", usecols=["row_id", "season", "pitcher_id",
                                                   "batter_id", "asof_pitcher_n",
                                                   "asof_batter_n"])
    d["row_id"] = d["row_id"].astype(str)
    p = p.merge(d, on="row_id", validate="one_to_one")
    assert np.array_equal(p.y_base, p.y_cand)
    p["delta_mse"] = (p.base - p.y_base) ** 2 - (p.cand - p.y_base) ** 2
    p["pcohort"] = p.pitcher_id // 100
    p["bcohort"] = p.batter_id // 100
    p["pitcher_n_bin"] = pd.cut(p.asof_pitcher_n, [-1, 0, 9, 49, 199, 999, np.inf])
    print(f"rows={len(p)} mean_delta_mse={p.delta_mse.mean():.9f}")
    for col in ("pcohort", "bcohort", "pitcher_n_bin"):
        z = p.groupby(col, observed=True).agg(rows=("delta_mse", "size"),
                                               delta_mse=("delta_mse", "mean"),
                                               pred_delta=("cand", lambda x: 0.0))
        # Compute prediction delta separately to avoid a groupby lambda closing over p.
        z["pred_delta"] = p.groupby(col, observed=True).apply(
            lambda g: float((g.cand - g.base).mean()), include_groups=False)
        print(f"\n[{col}]\n{z.to_string()}")


if __name__ == "__main__":
    main()
