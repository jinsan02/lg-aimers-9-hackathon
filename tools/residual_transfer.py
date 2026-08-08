"""Test whether a residual group correction transfers across seasons.

The source-season ensemble residual (prediction - target) is estimated per
group, shrunk toward the source global residual, and frozen before it is
applied to the target season.  This is a screening gate, not a fitted result
on the target season.

Example:
  python tools/residual_transfer.py BI2023 BI2024 pitcher_id,batter_hand
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd


def load(tag: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"./out/*{tag}_s*_test_preds.npz"))
    if not files:
        raise FileNotFoundError(f"no test predictions for tag {tag}")
    arrays = [np.load(path, allow_pickle=True) for path in files]
    row_id = arrays[0]["row_id"]
    if any(not np.array_equal(row_id, z["row_id"]) for z in arrays[1:]):
        raise ValueError(f"row_id mismatch inside {tag}")
    return pd.DataFrame(
        {
            "row_id": row_id,
            "y": arrays[0]["y"].astype(np.float64),
            "pred": np.mean([z["pred"] for z in arrays], axis=0).astype(np.float64),
        }
    )


def bss(y: np.ndarray, pred: np.ndarray) -> float:
    rate = float(y.mean())
    return 1e5 * (1.0 - np.mean((np.clip(pred, 0.0, 1.0) - y) ** 2)
                  / (rate * (1.0 - rate)))


def key_series(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    if len(keys) == 1:
        return df[keys[0]]
    return pd.Series(list(map(tuple, df[keys].to_numpy())), index=df.index)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_tag")
    parser.add_argument("target_tag")
    parser.add_argument("keys", help="comma-separated columns")
    parser.add_argument("--k", type=float, nargs="+", default=[30.0, 100.0, 300.0])
    args = parser.parse_args()

    keys = [x.strip() for x in args.keys.split(",") if x.strip()]
    source, target = load(args.source_tag), load(args.target_tag)
    meta = pd.read_csv("./data/train.csv", usecols=["row_id", *keys])
    source = source.merge(meta, on="row_id", how="left", validate="one_to_one")
    target = target.merge(meta, on="row_id", how="left", validate="one_to_one")
    if source[keys].isna().any().any() or target[keys].isna().any().any():
        raise ValueError("missing group metadata after row_id merge")

    source["err"] = source.pred - source.y
    target["err"] = target.pred - target.y
    source_global = float(source.err.mean())
    target_global = float(target.err.mean())
    source_group = source.groupby(keys, observed=True).err.agg(["mean", "size"])
    target_group = target.groupby(keys, observed=True).err.agg(["mean", "size"])
    target_key = key_series(target, keys)

    common = source_group.join(target_group, lsuffix="_src", rsuffix="_tgt", how="inner")
    corr = (float(common.mean_src.corr(common.mean_tgt))
            if len(common) >= 2 else float("nan"))
    hit = float(target_key.isin(source_group.index).mean())
    base = bss(target.y.to_numpy(), target.pred.to_numpy())
    centered = bss(target.y.to_numpy(), target.pred.to_numpy() - target_global)

    print(f"source={args.source_tag} n={len(source):,} global_err={source_global:+.6f}")
    print(f"target={args.target_tag} n={len(target):,} global_err={target_global:+.6f}")
    print(f"keys={','.join(keys)} source_groups={len(source_group):,} "
          f"common_groups={len(common):,} target_row_hit={hit:.1%} "
          f"unweighted_group_corr={corr:+.4f}")
    print(f"target raw BSS={base:.3f} | centered-oracle={centered:.3f}")
    print("k       adj_sd      raw_gain   centered_gain")
    for k in args.k:
        shrunk = ((source_group["mean"] * source_group["size"] + source_global * k)
                  / (source_group["size"] + k))
        # Remove the source global component: the gate measures only whether the
        # group structure transfers. Global calibration is handled separately.
        offsets = shrunk - source_global
        adj = target_key.map(offsets).fillna(0.0).to_numpy(np.float64)
        raw_gain = bss(target.y.to_numpy(), target.pred.to_numpy() - adj) - base
        centered_gain = (bss(target.y.to_numpy(),
                             target.pred.to_numpy() - target_global - adj)
                         - centered)
        print(f"{k:<7g} {adj.std():>10.6f} {raw_gain:>+12.3f} {centered_gain:>+15.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
