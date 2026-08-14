"""Which feature families survive a season boundary — fixed-model diagnostic.

The question is not "what would happen if we removed this family" — an earlier
pruning experiment removed provably redundant columns and got *worse*, because
CatBoost's split scaffold routes information through columns that carry none of
their own. This measures something narrower and answerable: **inside a model
that is already fitted, how much of what it leans on still pays off one season
later.**

  Why this needs no new training
  ------------------------------
  The shipped champion cannot answer it. `train_gbdt2` refits the deployment
  model on fit+val, so `cat_B1S_*.pkl` has seen every season in the file and has
  no unseen surface at all. But three older runs of the *same 121-feature
  recipe* were trained with `--test-season`, which splits the test season out
  before the refit:

      MV21_base / MV21_cell        fit <= 2021   unseen 2022
      MVB22_native / MVCELL22_s42  fit <= 2022   unseen 2023
      MVN3_s3,4,5 / MVCELL_s42     fit <= 2023   unseen 2024

  Their feature lists are byte-identical to B1S8's (asserted below), so the
  representation under test is the champion's. Their hyperparameters are not
  (`--feat-k`, `--te-k`, `--p1`, `--fm-modes` differ), and the MV cell taxonomy
  has 14 cells against B1S8's 12. So this maps the *representation*, not the
  champion's exact fitted surface, and no number here is comparable to a B1S8
  score. All three were trained on the same host (`rohjinsan`).

  The three surfaces per model
  ----------------------------
  Each model is scored on three seasons drawn from its own artifact, so the
  feature tables never extrapolate: two seasons inside its fit era and the one
  season it has never seen. Retention is read *within* a model — same trees,
  same tables, only the season of the scored rows changes.

  The in-sample side is inflated by memorisation, and unequally so: a family
  keyed to pitcher identity can be recalled, a family of situational flags
  cannot. That is why the older in-sample season is also scored — a family that
  is strong on both in-sample seasons and collapses only on the unseen one has
  lost transfer, while one that rises monotonically toward the recent season was
  only ever measuring recency.

  Permutation scheme — one, fixed, not swept
  ------------------------------------------
  For family G: draw a single permutation of the slice's rows and apply it to
  every column of G at once.

    * jointly, so the family's internal structure survives and what is destroyed
      is only its link to the target and to the other families;
    * within the slice, which is one season and one league, so no value from
      another era is injected;
    * categoricals are permuted as their own strings, so every value stays in
      vocabulary. Nothing is recoded to a number.

  `season` is constant inside a slice, so RAW_CONTEXT's figure is about its
  other 23 columns.

Run (CPU, ~35 min for the full grid):

  python tools/season_transfer_map.py --out out/season_transfer_map.csv
  python tools/season_transfer_map.py --quick        # one model, one season
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fpipe                                                     # noqa: E402
import feature_families as ff                                    # noqa: E402
import feature_blocks as fb                                      # noqa: E402

TARGET = "control_success"
REPEATS = 5

# model file, arm, fit era (last season in the refit), unseen season, seed.
# The two in-sample seasons scored are (unseen - 2) and (unseen - 1).
MODELS = [
    ("cat_MV21_base.pkl",     "base", 2021, 2022, 42),
    ("cat_MVB22_native.pkl",  "base", 2022, 2023, 42),
    ("cat_MVN3_s3.pkl",       "base", 2023, 2024, 3),
    ("cat_MVN3_s4.pkl",       "base", 2023, 2024, 4),
    ("cat_MVN3_s5.pkl",       "base", 2023, 2024, 5),
    ("cat_MV21_cell.pkl",     "cell", 2021, 2022, 42),
    ("cat_MVCELL22_s42.pkl",  "cell", 2022, 2023, 42),
    ("cat_MVCELL_s42.pkl",    "cell", 2023, 2024, 42),
    # Diagnostic calibration, 2026-08-15. The B1S8 recipe itself on the judging
    # surface, seed 3, 12-cell taxonomy -- the packs the MV surrogates were only
    # standing in for. Trained on DESKTOP-053T952, not on this laptop like every
    # row above, so its block *shares* may be compared with the surrogates' but
    # no score of any kind may be. Select it with `--only B1SMOKE`.
    ("cat_B1SMOKE_base.pkl",  "base", 2023, 2024, 3),
    ("cat_B1SMOKE_cell.pkl",  "cell", 2023, 2024, 3),
]

# test npz that pins each model's reconstruction. The val npz cannot be used:
# it holds the *selection* model's predictions, and the pickled model is the
# refit one, which has seen the validation season.
REPLAY = {
    "cat_MV21_base.pkl": "cat_MV21_base_test_preds.npz",
    "cat_MVB22_native.pkl": "cat_MVB22_native_test_preds.npz",
    "cat_MVN3_s3.pkl": "cat_MVN3_s3_test_preds.npz",
    "cat_MVN3_s4.pkl": "cat_MVN3_s4_test_preds.npz",
    "cat_MVN3_s5.pkl": "cat_MVN3_s5_test_preds.npz",
    "cat_MV21_cell.pkl": "cat_MV21_cell_test_preds.npz",
    "cat_MVCELL22_s42.pkl": "cat_MVCELL22_s42_test_preds.npz",
    "cat_MVCELL_s42.pkl": "cat_MVCELL_s42_test_preds.npz",
    "cat_B1SMOKE_base.pkl": "cat_B1SMOKE_base_test_preds.npz",
    "cat_B1SMOKE_cell.pkl": "cat_B1SMOKE_cell_test_preds.npz",
}


def bss(y, p):
    """Competition metric without the max(0, .) floor.

    The floor exists to stop a submission scoring below zero; here a permuted
    model is *supposed* to go negative and clipping would collapse every strong
    family onto the same number.
    """
    y = np.asarray(y, np.float64)
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    r = float(y.mean())
    return float(1e5 * (1.0 - np.mean((p - y) ** 2) / (r * (1.0 - r))))


def predict(pack, X):
    m = pack["model"]
    succ = pack.get("fm_success")
    pr = m.predict_proba(X)
    if succ is None:
        return pr[:, 1]
    return pr[:, np.asarray(succ)].sum(axis=1)


def frames(full, pack, season):
    raw = full[full["season"] == season].reset_index(drop=True)
    fr = fpipe.transform(raw, pack["fpipe"])
    X = fr[pack["features"]].copy()
    for c in pack["cat_cols"]:
        X[c] = X[c].astype(str)
    return X, fr[TARGET].to_numpy(np.float64), fr["game_type"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/season_transfer_map.csv")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--quick", action="store_true",
                    help="one model, unseen season only — a wiring smoke test")
    ap.add_argument("--groups", choices=("family", "block"), default="family",
                    help="family = 12 representation groups, reads as routing; "
                         "block = 6 closed information groups, reads as "
                         "information (see tools/feature_blocks.py)")
    ap.add_argument("--only", default="",
                    help="run only models whose filename contains this")
    args = ap.parse_args()
    grouping = ff if args.groups == "family" else fb
    key = "FAMILY_OF" if args.groups == "family" else "BLOCK_OF"

    header = list(pd.read_csv(os.path.join(ROOT, "data", "test.csv"), nrows=0,
                              encoding="utf-8-sig").columns)
    full = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                       encoding="utf-8-sig", usecols=header + [TARGET])
    print(f"train {len(full):,} rows, seasons {sorted(full.season.unique())}")

    models = MODELS[2:3] if args.quick else MODELS
    if args.only:
        models = [m for m in models if args.only in m[0]]
        if not models:
            raise SystemExit(f"--only {args.only!r} matched no model")
    rows = []
    t_start = time.time()

    for pkl, arm, fit_era, unseen, seed in models:
        pack = joblib.load(os.path.join(ROOT, "model", pkl))
        grouping.validate(pack["features"])    # refuses an unmapped column
        member = getattr(grouping, key)
        fams = {g: [c for c in pack["features"] if member[c] == g]
                for g in grouping.ORDER}

        seasons = ([unseen] if args.quick
                   else [unseen - 2, unseen - 1, unseen])
        print(f"\n=== {pkl}  arm={arm}  fit<={fit_era}  unseen={unseen}  "
              f"seed={seed}  trees={pack['model'].tree_count_}")

        for season in seasons:
            if season not in set(full["season"]):
                print(f"  season {season}: absent from train.csv, skipped")
                continue
            horizon = ("next_season" if season == unseen
                       else "in_sample_val" if season == fit_era
                       else "in_sample_old")
            X, y, league = frames(full, pack, season)

            if season == unseen:                # pin the reconstruction
                z = np.load(os.path.join(ROOT, "out", REPLAY[pkl]),
                            allow_pickle=True)
                d = float(np.max(np.abs(predict(pack, X)
                                        - z["pred"].astype(np.float64))))
                if d > 1e-12:
                    raise SystemExit(f"{pkl} @{season}: replay mismatch {d:.3g}")
                print(f"  season {season} [{horizon}] replay_max_abs_diff=0")

            for lg in ("R", "F"):
                m = league == lg
                if m.sum() < 5000:
                    continue
                Xl, yl = X[m].reset_index(drop=True), y[m]
                intact = bss(yl, predict(pack, Xl))
                t0 = time.time()
                per_fam = {}
                for g, cols in fams.items():
                    ds = []
                    for r in range(args.repeats):
                        rng = np.random.default_rng(1000 * r + 7)
                        idx = rng.permutation(len(Xl))
                        Xp = Xl.copy()
                        for c in cols:
                            Xp[c] = Xp[c].to_numpy()[idx]
                        ds.append(intact - bss(yl, predict(pack, Xp)))
                    per_fam[g] = (float(np.mean(ds)), float(np.std(ds, ddof=1)))
                tot = sum(max(v[0], 0.0) for v in per_fam.values()) or 1.0
                for g, (mu, sd) in per_fam.items():
                    rows.append(dict(
                        model=pkl.replace("cat_", "").replace(".pkl", ""),
                        arm=arm, fit_era=fit_era, seed=seed, season=season,
                        horizon=horizon, league=lg, n_rows=int(m.sum()),
                        base_bss=round(intact, 3), family=g,
                        n_cols=len(fams[g]), dbss=round(mu, 4),
                        dbss_sd=round(sd, 4),
                        share=round(100.0 * max(mu, 0.0) / tot, 3)))
                print(f"    {lg} n={int(m.sum()):>7,} BSS {intact:9.2f}  "
                      f"({time.time() - t0:.0f}s)  " +
                      "  ".join(f"{g[:7]} {per_fam[g][0]:+.0f}"
                                for g in sorted(per_fam,
                                                key=lambda k: -per_fam[k][0])[:4]))
            del X

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} rows, "
          f"{time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
