"""Recover train batter_id <-> Trackman batter ID using shared pitch contexts.

The confident pitcher map is used inside the context key.  This makes the
batter fingerprint substantially sharper than matching only calendar/count
distributions.  The output is an identity linkage only; no current-pitch
Trackman measurement is joined to train/test rows.
"""

import os
import sys

import numpy as np
import pandas as pd

DATA = "./data"
OUT = f"{DATA}/processed/batter_map2.csv"
PITCHER_MAP = f"{DATA}/processed/pitcher_map2.csv"
KEYS = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
        "balls_before", "strikes_before", "outs_before"]


def _norm_train_hand(s):
    # The majority code is right-handed in the released train data.
    major = s.value_counts().index[0]
    return np.where(s == major, "R", "L")


def _norm_tm_hand(s):
    x = s.astype(str).str.upper().str[0]
    return x.where(x.isin(["R", "L"]), "S")


def _best(sc):
    sc = sc.sort_values(["batter_id", "score"], ascending=[True, False])
    top = sc.groupby("batter_id").head(2).copy()
    top["rk"] = top.groupby("batter_id").cumcount()
    r0 = top[top.rk == 0].set_index("batter_id")
    r1 = top[top.rk == 1].set_index("batter_id")
    out = pd.DataFrame({
        "tm_batter_id": r0["tm_batter_id"],
        "score": r0["score"],
        "ratio": r0["ratio"],
        "score2": r1["score"].reindex(r0.index).fillna(0.0),
    }).reset_index()
    out["margin"] = out["score"] / np.maximum(out["score2"], 1.0)
    out = out.sort_values("score", ascending=False)
    out["dup"] = out.duplicated("tm_batter_id")
    return out


def _score(m, a, b):
    sc = m.groupby(["batter_id", "tm_batter_id"])["ov"].sum() \
        .rename("score").reset_index()
    ta = a.groupby("batter_id")["na"].sum().rename("n_train")
    tb = b.groupby("tm_batter_id")["nb"].sum().rename("n_tm")
    sc = sc.join(ta, on="batter_id").join(tb, on="tm_batter_id")
    sc["ratio"] = sc["score"] / np.minimum(sc["n_train"], sc["n_tm"])
    return sc


def main():
    pmap = pd.read_csv(PITCHER_MAP)
    tm_to_pitcher = pmap.set_index("tm_id")["pitcher_id"]

    tr_cols = KEYS + ["pitcher_id", "batter_id", "batter_hand"]
    tm_cols = KEYS + ["pitcher_trackman_id", "batter_trackman_id", "batter_hand"]
    tr = pd.read_csv(f"{DATA}/train.csv", usecols=tr_cols)
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", usecols=tm_cols)
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(tm_to_pitcher)
    before = len(tm)
    tm = tm.dropna(subset=["pitcher_id"]).copy()
    tm["pitcher_id"] = tm["pitcher_id"].astype(tr.pitcher_id.dtype)
    # train uses compact codes while Trackman can use Top/Bottom strings.
    for frame in (tr, tm):
        frame["top_bottom"] = frame["top_bottom"].astype(str).str[0].str.upper()
    tr["hand"] = _norm_train_hand(tr.batter_hand)
    tm["hand"] = _norm_tm_hand(tm.batter_hand)
    print(f"train {len(tr):,} | tm pitcher-linked {len(tm):,}/{before:,} "
          f"({len(tm) / before:.1%})")

    join = KEYS + ["pitcher_id", "hand"]
    a = tr.groupby(join + ["batter_id"]).size().rename("na").reset_index()
    b = tm.groupby(join + ["batter_trackman_id"]).size().rename("nb").reset_index() \
        .rename(columns={"batter_trackman_id": "tm_batter_id"})
    m = a.merge(b, on=join, how="inner")
    m["ov"] = np.minimum(m.na, m.nb)
    print(f"grouped train={len(a):,} tm={len(b):,} candidate-contexts={len(m):,}")

    best = _best(_score(m, a, b))
    conf = best[(~best.dup) & (best.margin >= 1.5) &
                (best.ratio >= 0.50) & (best.score >= 20)].copy()

    # Independent era agreement is a correctness audit, not a selection input.
    era = []
    for name, mask in (("early", m.season <= 2021), ("late", m.season >= 2022)):
        ma = m.loc[mask]
        aa = a[a.season <= 2021] if name == "early" else a[a.season >= 2022]
        bb = b[b.season <= 2021] if name == "early" else b[b.season >= 2022]
        era.append(_best(_score(ma, aa, bb))[["batter_id", "tm_batter_id"]]
                   .rename(columns={"tm_batter_id": name}))
    chk = era[0].merge(era[1], on="batter_id")
    agree = (chk.early == chk.late).mean() if len(chk) else np.nan

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    conf[["batter_id", "tm_batter_id", "score", "ratio", "margin"]] \
        .sort_values("batter_id").to_csv(OUT, index=False)
    total = best.batter_id.nunique()
    if total == 0:
        raise RuntimeError("no candidate batter pairs after normalized context join")
    print(f"confident {len(conf):,}/{total:,} "
          f"({len(conf) / total:.1%}) | duplicate rejected {best.dup.sum():,}")
    print(f"ratio median={conf.ratio.median():.3f} | margin median={conf.margin.median():.2f}")
    print(f"independent early/late agreement={agree:.1%} (common {len(chk):,})")
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
