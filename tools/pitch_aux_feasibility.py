"""Measure whether SBS-style next-pitch prediction can transfer to this contest.

Two 3-class pitch-type models are trained on Trackman seasons <= 2023 and
evaluated on 2024:

* row_local: only fields that have a row-local analogue in train/test.csv.
* sequence: row_local plus pitch_of_pa and the previous two pitch types.

The second arm approximates the public description of the 2017 SBS system,
whereas only the first arm could legally create an inference-time auxiliary
feature for the competition.  No Trackman measurement from the target pitch is
used as an input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score


DATA = Path("data")
OUT = Path("out")
TYPES = ["fastball", "breaking", "offspeed"]


def load() -> pd.DataFrame:
    cols = [
        "season", "trackman_game_id", "pitch_no", "game_month",
        "game_dayofweek", "inning", "top_bottom", "balls_before",
        "strikes_before", "outs_before", "pitch_of_pa",
        "pitcher_trackman_id", "batter_trackman_id", "pitcher_hand",
        "batter_hand", "pitch_type_group",
    ]
    d = pd.read_csv(DATA / "trackman_history.csv", encoding="utf-8-sig",
                    usecols=cols)
    d = d[d.pitch_type_group.isin(TYPES)].copy()

    # Existing mappings recover anonymous identity only.  They do not attach
    # the current pitch's post-release measurements to a contest row.
    pm = pd.read_csv(DATA / "processed/pitcher_map2.csv")
    bm = pd.read_csv(DATA / "processed/batter_map2.csv")
    pmap = pm.drop_duplicates("tm_id").set_index("tm_id").pitcher_id
    bmap = bm.drop_duplicates("tm_batter_id").set_index("tm_batter_id").batter_id
    d["pitcher_id"] = d.pitcher_trackman_id.map(pmap).fillna(-1).astype(int).astype(str)
    d["batter_id"] = d.batter_trackman_id.map(bmap).fillna(-1).astype(int).astype(str)

    # Reconstruct only Trackman's own within-game sequence for the SBS arm.
    d = d.sort_values(["trackman_game_id", "pitch_no"]).reset_index(drop=True)
    by_game = d.groupby("trackman_game_id", sort=False)
    for lag in (1, 2):
        prev_type = by_game.pitch_type_group.shift(lag)
        prev_pa = by_game.pitch_of_pa.shift(lag)
        valid = d.pitch_of_pa.eq(prev_pa + lag)
        d[f"prev_type_{lag}"] = prev_type.where(valid, "NONE").fillna("NONE")
    return d


def score(y: np.ndarray, proba: np.ndarray, train_prior: np.ndarray) -> dict:
    pred = np.asarray(TYPES)[np.argmax(proba, axis=1)]
    majority = TYPES[int(np.argmax(train_prior))]
    target_index = np.fromiter((TYPES.index(v) for v in y), dtype=np.int8,
                               count=len(y))
    eps = 1e-15
    model_ll = -np.log(np.clip(proba[np.arange(len(y)), target_index],
                               eps, 1.0)).mean()
    prior_ll = -np.log(np.clip(train_prior[target_index], eps, 1.0)).mean()
    per_class = {}
    for typ in TYPES:
        m = y == typ
        per_class[typ] = float(np.mean(pred[m] == typ))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "logloss": float(model_ll),
        "majority_accuracy": float(np.mean(y == majority)),
        "prior_logloss": float(prior_ll),
        "recall": per_class,
    }


def fit_arm(d: pd.DataFrame, name: str, features: list[str], iterations: int):
    tr = d.season <= 2023
    va = d.season == 2024
    cats = [c for c in features if c in {
        "game_dayofweek", "top_bottom", "pitcher_id", "batter_id",
        "pitcher_hand", "batter_hand", "prev_type_1", "prev_type_2",
    }]
    xtr, xva = d.loc[tr, features].copy(), d.loc[va, features].copy()
    for c in cats:
        xtr[c] = xtr[c].astype(str)
        xva[c] = xva[c].astype(str)
    ytr = d.loc[tr, "pitch_type_group"].astype(str).to_numpy()
    yva = d.loc[va, "pitch_type_group"].astype(str).to_numpy()
    train_prior = pd.Series(ytr).value_counts(normalize=True).reindex(TYPES).to_numpy()

    model = CatBoostClassifier(
        loss_function="MultiClass", eval_metric="MultiClass",
        iterations=iterations, depth=7, learning_rate=0.08, l2_leaf_reg=10,
        random_seed=42, task_type="GPU", devices="0",
        max_ctr_complexity=1, allow_writing_files=False, verbose=100,
    )
    model.fit(Pool(xtr, ytr, cat_features=cats),
              eval_set=Pool(xva, yva, cat_features=cats),
              early_stopping_rounds=60)
    raw = model.predict_proba(xva)
    # CatBoost class order is explicit and may differ from TYPES.
    order = [list(model.classes_).index(c) for c in TYPES]
    proba = raw[:, order]
    result = score(yva, proba, train_prior)
    result.update({
        "name": name,
        "best_iteration": int(model.get_best_iteration()),
        "features": features,
        "feature_importance": dict(zip(
            features, map(float, model.get_feature_importance()))),
    })
    np.savez_compressed(OUT / f"pitch_aux_{name}_2024.npz",
                        y=yva, proba=proba, classes=np.asarray(TYPES))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=500)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    d = load()
    common = [
        "season", "game_month", "game_dayofweek", "inning", "top_bottom",
        "balls_before", "strikes_before", "outs_before", "pitcher_id",
        "batter_id", "pitcher_hand", "batter_hand",
    ]
    print(f"rows={len(d):,} train={(d.season <= 2023).sum():,} "
          f"val2024={(d.season == 2024).sum():,}", flush=True)
    results = [
        fit_arm(d, "row_local", common, args.iterations),
        fit_arm(d, "sequence", common + ["pitch_of_pa", "prev_type_1",
                                          "prev_type_2"], args.iterations),
    ]
    path = OUT / "pitch_aux_feasibility.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
