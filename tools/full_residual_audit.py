"""Audit every released column against a fixed champion-like residual.

A smoothed correction is learned from a model's source-season validation
residual and applied to that model's separately refit next-season prediction.
This asks a stricter question than target correlation: what structure remains
after the current 121-feature CatBoost has already acted?
"""

from __future__ import annotations

import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd

from full_dataset_audit import _feature_code, raw_bss


OUT = "out/full_audit"
TARGET = "control_success"


def load_frames(tag="MVA_native", source_year=2023, target_year=2024):
    cols = list(pd.read_csv("data/test.csv", nrows=0,
                            encoding="utf-8-sig").columns)
    feats = [c for c in cols if c != "row_id"]
    d = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                    usecols=cols + [TARGET])
    src = d[d.season == source_year].reset_index(drop=True)
    tgt = d[d.season == target_year].reset_index(drop=True)
    zv = np.load(f"out/cat_{tag}_val_preds.npz")
    zt = np.load(f"out/cat_{tag}_test_preds.npz", allow_pickle=True)
    if len(zv["pred"]) != len(src):
        raise RuntimeError(f"source order/length mismatch {len(zv['pred'])} != {len(src)}")
    if "row_id" in zt:
        pos = pd.Series(np.arange(len(tgt)), index=tgt.row_id)
        order = pos.reindex(zt["row_id"]).to_numpy()
        if np.isnan(order).any():
            raise RuntimeError("2024 row_id mismatch")
        tgt = tgt.iloc[order.astype(int)].reset_index(drop=True)
    if len(zt["pred"]) != len(tgt):
        raise RuntimeError("2024 prediction length mismatch")
    return src, tgt, feats, zv["pred"].astype(float), zt["pred"].astype(float)


def correction(resid, cs, ct, n_groups, k=500.0, zero_mean=True):
    n = np.bincount(cs, minlength=n_groups).astype(float)
    s = np.bincount(cs, weights=resid, minlength=n_groups).astype(float)
    q = s / (n + k)
    if zero_mean:
        q -= np.average(q[cs])
    return q[ct]


def gains(tgt, p0, add):
    y = tgt[TARGET].to_numpy(float)
    base = raw_bss(y, p0)
    masks = {
        "all": np.ones(len(tgt), bool),
        "early": tgt.game_month.to_numpy() <= 6,
        "late": tgt.game_month.to_numpy() > 6,
        "R": tgt.game_type.astype(str).to_numpy() == "R",
        "F": tgt.game_type.astype(str).to_numpy() == "F",
    }
    out = {}
    for name, m in masks.items():
        out[name] = raw_bss(y[m], np.clip(p0[m] + add[m], .01, .99)) - raw_bss(y[m], p0[m])
    out["rms"] = float(np.sqrt(np.mean(add ** 2)))
    return out


def audit(stage="single", tag="MVA_native", source_year=2023, target_year=2024):
    os.makedirs(OUT, exist_ok=True)
    src, tgt, feats, ps, pt = load_frames(tag, source_year, target_year)
    suffix = f"{source_year}_{target_year}_{tag}"
    resid = src[TARGET].to_numpy(float) - ps
    codes = {c: _feature_code(src[c], tgt[c]) for c in feats}
    uni = {}
    rows = []
    for c in feats:
        a, b, n, kind = codes[c]
        add = correction(resid, a, b, n)
        g = gains(tgt, pt, add)
        uni[c] = g["all"]
        rows.append({"column": c, "kind": kind, "groups": n, **g})
    pd.DataFrame(rows).sort_values("all", ascending=False).to_csv(
        f"{OUT}/residual_single_{suffix}.csv", index=False)
    if stage == "single":
        return

    rows = []
    for ix, (a, b) in enumerate(combinations(feats, 2), 1):
        sa, ta, na, _ = codes[a]
        sb, tb, nb, _ = codes[b]
        cs = sa.astype(np.int64) * nb + sb
        ct = ta.astype(np.int64) * nb + tb
        add = correction(resid, cs, ct, na * nb)
        g = gains(tgt, pt, add)
        rows.append({"a": a, "b": b, "groups": na * nb,
                     "gain_over_best_single": g["all"] - max(uni[a], uni[b]), **g})
        if ix % 100 == 0:
            print(f"residual pairs {ix}/1081", flush=True)
    pd.DataFrame(rows).sort_values("all", ascending=False).to_csv(
        f"{OUT}/residual_pair_{suffix}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["single", "pairs"], default="single", nargs="?")
    ap.add_argument("--tag", default="MVA_native")
    ap.add_argument("--source-year", type=int, default=2023)
    ap.add_argument("--target-year", type=int, default=2024)
    args = ap.parse_args()
    audit(args.stage, args.tag, args.source_year, args.target_year)


if __name__ == "__main__":
    main()
