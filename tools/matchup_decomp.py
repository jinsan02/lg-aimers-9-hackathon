"""Inside MATCHUP: is it the platoon, or is it this pitcher's record by hand?

The block map says MATCHUP holds across every season boundary measured -- 6 of 6
cells STABLE_POSITIVE, retention 1.05 to 1.18, 14-17% of the base arm's unseen
importance. Read carelessly that reopens the hand axis, which is exactly the
axis `--feat-h1` just failed on and `skill --axis hand` has never run.

But MATCHUP holds two different things:

  HAND_RAW   pitcher_hand, batter_hand and the flags derived from them.
             Left-on-left is a fact about biomechanics; it does not need a
             season to re-learn, and the model already has it.

  HAND_TE    te_pitcher_batter_hand_{ratio, rate, n, ratio_dev} -- *this
             pitcher's* record against that handedness. This is the quantity
             H1 composes and the one `skill --axis hand` would fit.

Only the second is new information a candidate could exploit. So the question
is not "does MATCHUP transfer" but "does the pitcher-specific part transfer once
the platoon itself is held fixed".

Both subgroups turn out to be closed -- the check runs first and prints its
verdict, because the worry was real: the hand-keyed TE columns are keyed *by*
batter_hand and could have stood in for it. At the 1e-4 linear tolerance they
do not, so both numbers may be read as information rather than as routing.

  python tools/matchup_decomp.py
"""

from __future__ import annotations

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import feature_blocks as fb                                      # noqa: E402
from season_transfer_map import TARGET, bss, frames, predict     # noqa: E402

GROUPS = {
    "HAND_RAW": ["pitcher_hand", "batter_hand", "matchup_same_hand",
                 "dom_same_hand", "dom_same_ball", "dom_same_rev"],
    "HAND_TE": ["te_pitcher_batter_hand_ratio", "te_pitcher_batter_hand_rate",
                "te_pitcher_batter_hand_n",
                "te_pitcher_batter_hand_ratio_dev"],
    "MATCHUP_ALL": fb.BLOCKS["MATCHUP"],
}

# (model, arm, val season it was refit on, unseen season)
RUNS = [
    ("cat_MVN3_s3.pkl", "base", 2023, 2024),
    ("cat_MVN3_s4.pkl", "base", 2023, 2024),
    ("cat_MVN3_s5.pkl", "base", 2023, 2024),
    ("cat_MVCELL_s42.pkl", "cell", 2023, 2024),
    ("cat_MVB22_native.pkl", "base", 2022, 2023),
    ("cat_MVCELL22_s42.pkl", "cell", 2022, 2023),
    ("cat_MV21_base.pkl", "base", 2021, 2022),
    ("cat_MV21_cell.pkl", "cell", 2021, 2022),
]
REPEATS = 5


def main():
    header = list(pd.read_csv(os.path.join(ROOT, "data", "test.csv"), nrows=0,
                              encoding="utf-8-sig").columns)
    full = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                       encoding="utf-8-sig", usecols=header + [TARGET])

    # closure: nothing in a subgroup may be rebuilt from outside it
    pack0 = joblib.load(os.path.join(ROOT, "model", "cat_B1S_base_s3.pkl"))
    fr0 = frames(full, pack0, 2024)[0]
    fr0 = fr0.assign(**{TARGET: 0.0})
    saved = fb.BLOCK_OF, fb.ORDER
    for g in ("HAND_RAW", "HAND_TE"):
        fb.BLOCK_OF = {c: (g if c in GROUPS[g] else "REST")
                       for c in pack0["features"]}
        fb.ORDER = [g, "REST"]
        leaks = [l for l in fb.closure_report(fr0, pack0["features"],
                                              pack0["cat_cols"]) if l[0] == g]
        print(f"closure {g:12} : " +
              ("CLOSED" if not leaks
               else "NOT CLOSED -> " + ", ".join(f"{c} r2={r:.5f}"
                                                 for _, c, r in leaks)))
    fb.BLOCK_OF, fb.ORDER = saved

    rows = []
    for pkl, arm, val, unseen in RUNS:
        pack = joblib.load(os.path.join(ROOT, "model", pkl))
        print(f"\n=== {pkl} ({arm}) refit<={val}, unseen {unseen}")
        for season, horizon in ((val, "in_sample_val"),
                                (unseen, "next_season")):
            X, y, league = frames(full, pack, season)
            m = league == "R"
            Xl, yl = X[m].reset_index(drop=True), y[m]
            intact = bss(yl, predict(pack, Xl))
            line = []
            for g, cols in GROUPS.items():
                t0 = time.time()
                ds = []
                for r in range(REPEATS):
                    rng = np.random.default_rng(1000 * r + 7)
                    idx = rng.permutation(len(Xl))
                    Xp = Xl.copy()
                    for c in cols:
                        Xp[c] = Xp[c].to_numpy()[idx]
                    ds.append(intact - bss(yl, predict(pack, Xp)))
                rows.append(dict(model=pkl, arm=arm, season=season,
                                 horizon=horizon, group=g,
                                 base_bss=round(intact, 2),
                                 dbss=round(float(np.mean(ds)), 3),
                                 sd=round(float(np.std(ds, ddof=1)), 3)))
                line.append(f"{g} {np.mean(ds):+7.1f}")
            print(f"  {season} [{horizon:13}] R n={int(m.sum()):,} "
                  f"BSS {intact:8.2f}   " + "   ".join(line))
            del X

    t = pd.DataFrame(rows)
    t.to_csv("out/matchup_decomp.csv", index=False)
    print("\nretention (next / same), R league")
    p = t.pivot_table(index=["model", "arm"], columns=["group", "horizon"],
                      values="dbss")
    for g in GROUPS:
        p[(g, "retain")] = p[(g, "next_season")] / p[(g, "in_sample_val")]
    print(p[[(g, "retain") for g in GROUPS]].round(3).to_string())
    print("\nwrote out/matchup_decomp.csv")


if __name__ == "__main__":
    main()
