"""Audit whether the frozen Trackman pitcher linkage can recover more IDs.

The production linker omitted ``batter_hand`` even though it is the only
unused row-local field shared by train.csv and trackman_history.csv.  This
script repeats the histogram-overlap identity match with that field included.
It only reports aggregate coverage and agreement; it never joins Trackman
measurements to evaluation rows and does not write a mapping by default.
"""

import argparse
import os

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
KEYS = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
        "balls_before", "strikes_before", "outs_before"]


def hand_code(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.map({1: 0, 2: 1}).fillna(-1).astype(np.int8)
    z = s.astype(str).str.upper().str[0]
    return z.map({"L": 0, "R": 1}).fillna(-1).astype(np.int8)


def key_code(df):
    tb = (df["top_bottom"].astype(str).str[0].str.upper() == "T").astype(np.int64)
    x = df["season"].astype(np.int64) - 2018
    for value, width in (
        (df["game_month"].astype(np.int64), 13),
        (df["game_dayofweek"].astype(np.int64), 8),
        (df["inning"].clip(1, 15).astype(np.int64), 16),
        (tb, 2),
        (df["balls_before"].clip(0, 3).astype(np.int64), 4),
        (df["strikes_before"].clip(0, 2).astype(np.int64), 3),
        (df["outs_before"].clip(0, 2).astype(np.int64), 3),
        (hand_code(df["pitcher_hand"]), 2),
        (hand_code(df["batter_hand"]), 2),
    ):
        x = x * width + value
    return x


def candidate_table():
    shared = KEYS + ["pitcher_hand", "batter_hand"]
    tr = pd.read_csv(os.path.join(DATA, "train.csv"),
                     usecols=shared + ["pitcher_id"])
    tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"),
                     usecols=shared + ["pitcher_trackman_id"])
    tr["k"], tm["k"] = key_code(tr), key_code(tm)
    a = tr.groupby(["k", "pitcher_id"]).size().rename("na").reset_index()
    b = (tm.groupby(["k", "pitcher_trackman_id"]).size().rename("nb")
         .reset_index().rename(columns={"pitcher_trackman_id": "tm_id"}))
    m = a.merge(b, on="k", how="inner")
    m["ov"] = np.minimum(m.na, m.nb)
    sc = m.groupby(["pitcher_id", "tm_id"]).ov.sum().rename("score").reset_index()
    ta = a.groupby("pitcher_id").na.sum().rename("n_train")
    tb = b.groupby("tm_id").nb.sum().rename("n_tm")
    sc = sc.join(ta, on="pitcher_id").join(tb, on="tm_id")
    sc["ratio"] = sc.score / np.minimum(sc.n_train, sc.n_tm)
    sc = sc.sort_values(["pitcher_id", "score"], ascending=[True, False])
    top = sc.groupby("pitcher_id").head(2).copy()
    top["rank"] = top.groupby("pitcher_id").cumcount()
    first = top[top["rank"] == 0].set_index("pitcher_id")
    second = top[top["rank"] == 1].set_index("pitcher_id")
    best = first[["tm_id", "score", "ratio", "n_train"]].copy()
    best["score2"] = second.score.reindex(first.index).fillna(0)
    best["margin"] = best.score / np.maximum(best.score2, 1)
    best = best.reset_index().sort_values("score", ascending=False)
    best["duplicate_tm"] = best.duplicated("tm_id")
    return tr, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the confident audit mapping as pitcher_map3.csv")
    args = ap.parse_args()
    tr, best = candidate_table()
    conf = best[(best.margin >= 1.5) & ~best.duplicate_tm].copy()
    old = pd.read_csv(os.path.join(DATA, "processed", "pitcher_map2.csv"))
    joined = conf.merge(old[["pitcher_id", "tm_id"]], on="pitcher_id",
                        suffixes=("_new", "_old"))
    agree = np.mean(joined.tm_id_new == joined.tm_id_old) if len(joined) else np.nan
    old_rows = tr.pitcher_id.isin(old.pitcher_id).mean()
    new_rows = tr.pitcher_id.isin(conf.pitcher_id).mean()
    added = conf[~conf.pitcher_id.isin(old.pitcher_id)]
    lost = old[~old.pitcher_id.isin(conf.pitcher_id)]
    print(f"existing players={old.pitcher_id.nunique():,} row_coverage={old_rows:.4%}")
    print(f"+batter_hand players={conf.pitcher_id.nunique():,} row_coverage={new_rows:.4%}")
    print(f"common={len(joined):,} identity_agreement={agree:.4%} "
          f"added={len(added):,} lost={len(lost):,}")
    print(f"coverage_gain={(new_rows-old_rows):+.4%} "
          f"remaining_rows={(1-new_rows):.4%}")
    if len(added):
        print("added aggregate: " +
              f"rows={int(added.n_train.sum()):,} median_n={added.n_train.median():.1f} "
              f"median_ratio={added.ratio.median():.3f} median_margin={added.margin.median():.2f}")
    if args.write:
        out = os.path.join(DATA, "processed", "pitcher_map3.csv")
        conf[["pitcher_id", "tm_id", "score", "ratio", "margin"]] \
            .sort_values("pitcher_id").to_csv(out, index=False)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
