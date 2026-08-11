"""Screen fitted CatBoost features on a future-season Brier surface.

This is deliberately a screening tool, not an adoption test.  CatBoost's
LossFunctionChange importance is computed for log loss, whereas the contest
uses Brier score.  A feature is therefore only a candidate for a real paired
retrain when its sign is stable across rolling transitions and it belongs to a
small, coherent feature family.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

import fpipe


TARGET = "control_success"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pack = joblib.load(args.model)
    z = np.load(args.preds, allow_pickle=True)
    header = list(pd.read_csv("data/test.csv", nrows=0,
                              encoding="utf-8-sig").columns)
    raw = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                      usecols=header + [TARGET])
    raw = raw.loc[raw["season"] == args.season].reset_index(drop=True)

    if "row_id" in z:
        pos = pd.Series(np.arange(len(raw)), index=raw["row_id"])
        order = pos.reindex(z["row_id"]).to_numpy()
        if pd.isna(order).any():
            raise RuntimeError("prediction row_id is not a permutation of target rows")
        raw = raw.iloc[order.astype(int)].reset_index(drop=True)
    if len(raw) != len(z["pred"]):
        raise RuntimeError(f"row mismatch: {len(raw)} != {len(z['pred'])}")

    frame = fpipe.transform(raw, pack["fpipe"])
    features = pack["features"]
    cat_cols = pack["cat_cols"]
    x = frame[features]
    replay = pack["model"].predict_proba(x)[:, 1]
    max_diff = float(np.max(np.abs(replay - z["pred"])))
    if max_diff > 1e-12:
        raise RuntimeError(f"model replay mismatch: max_abs_diff={max_diff:.3g}")

    pool = Pool(x, label=raw[TARGET].to_numpy(float),
                cat_features=[features.index(c) for c in cat_cols])
    values = pack["model"].get_feature_importance(
        pool, type="LossFunctionChange")
    out = pd.DataFrame({
        "feature": features,
        "loss_change": values,
        "sign": np.where(values < 0, "negative", "positive"),
        "season": args.season,
        "model": Path(args.model).name,
    }).sort_values("loss_change")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"replay_max_abs_diff={max_diff:.3g}")
    print(f"features={len(out)} negative={(out.loss_change < 0).sum()} "
          f"positive={(out.loss_change > 0).sum()}")
    print(out.head(20).to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
