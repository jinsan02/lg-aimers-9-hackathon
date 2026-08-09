"""Audit one predeclared PB familiarity shrinkage rule.

Pairs seen before the source season use k=250; other source-season pairs keep
the champion k=500.  This uses only pre-source history and source residuals.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbmf_transfer_audit as core


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def adaptive_pair(src, tgt, history, resid):
    prior = set(core.pair_keys(history))
    keys = core.pair_keys(src)
    tab = pd.DataFrame({"key": keys, "r": resid}).groupby("key").r.agg(["sum", "size"])
    tab["familiar"] = tab.index.isin(prior)
    tab["k"] = np.where(tab.familiar, 250.0, 500.0)
    mp = (tab["sum"]/(tab["size"]+tab["k"])).to_dict()
    a = pd.Series(keys).map(mp).fillna(0).to_numpy(float)
    b = pd.Series(core.pair_keys(tgt)).map(mp).fillna(0).to_numpy(float)
    mean = float(a.mean())
    return a-mean, b-mean, int(tab.familiar.sum()), len(tab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--source", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--drop-f-pre", type=int, default=0)
    args = ap.parse_args()

    bv, ys, _ = core.ensemble(args.base, "val")
    cv, ysc, _ = core.ensemble(args.cell, "val")
    bt, yt, _ = core.ensemble(args.base, "test")
    ct, ytc, _ = core.ensemble(args.cell, "test")
    if not (np.array_equal(ys, ysc) and np.array_equal(yt, ytc)):
        raise ValueError("target mismatch")
    ps = core.post((1-core.W_CELL)*bv+core.W_CELL*cv)
    pt = core.post((1-core.W_CELL)*bt+core.W_CELL*ct)

    cols = ["season", "game_month", "game_type", "pitcher_id", "batter_id", core.MID]
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    if args.drop_f_pre:
        d = d[~((d.game_type == "F") & (d.season <= args.drop_f_pre))]
    src = d[d.season == args.source].reset_index(drop=True)
    tgt = d[d.season == args.target].reset_index(drop=True)
    hist = d[d.season < args.source].reset_index(drop=True)
    if len(src) != len(ys) or len(tgt) != len(yt):
        raise ValueError("row mismatch")

    ms, mt = core.middle_keys(src[core.MID].to_numpy(float), tgt[core.MID].to_numpy(float))
    r = ys-ps; r -= r.mean()
    ma, mb = core.fit_map(ms, mt, r, 500.)
    r = ys-(ps+ma); r -= r.mean()
    ea, eb = core.fit_map(core.pair_keys(src), core.pair_keys(tgt), r, 500.)
    aa, ab, familiar, groups = adaptive_pair(src, tgt, hist, r)
    k0t, adt = pt+mb+eb, pt+mb+ab
    print(f"surface {args.source}->{args.target} familiar_groups={familiar:,}/{groups:,}")
    core.report("adaptivePB-K0", tgt, yt, k0t, adt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
