"""Load the two clean champion-like transfer frames used by CPU audits."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import fpipe  # noqa: E402
import tm2command_supervised_audit as core  # noqa: E402


BOUNDARIES = (
    (2022, 2023, "BND22_base", "BND22_cell", "BND22"),
    (2023, 2024, "B1J6_base", "B1J6_cell", "B1J6"),
)


def load_train():
    cols = list(pd.read_csv(ROOT / "data/test.csv", nrows=0).columns)
    return pd.read_csv(ROOT / "data/train.csv",
                       usecols=list(dict.fromkeys(cols + ["control_success"])))


def boundary(data, source, target, base, cell, tag):
    ps, ys = core.core_predictions(base, cell, "val")
    pt, yt = core.core_predictions(base, cell, "test")
    s = data[data.season == source].reset_index(drop=True)
    t = data[data.season == target].reset_index(drop=True)
    if len(s) != len(ys) or len(t) != len(yt):
        raise RuntimeError(f"{tag}: prediction/frame length mismatch")
    if not np.array_equal(s.control_success.to_numpy(), ys):
        raise RuntimeError(f"{tag}: source target mismatch")
    if not np.array_equal(t.control_success.to_numpy(), yt):
        raise RuntimeError(f"{tag}: target target mismatch")
    pack = joblib.load(ROOT / f"model/cat_{tag}_base_s3.pkl")
    sx = fpipe.transform(s.drop(columns="control_success").copy(), pack["fpipe"])
    tx = fpipe.transform(t.drop(columns="control_success").copy(), pack["fpipe"])
    nums = [c for c in pack["features"] if c not in pack["cat_cols"]]
    return {
        "source": source, "target": target, "tag": tag,
        "s": s, "t": t, "ps": ps, "pt": pt, "ys": ys, "yt": yt,
        "rs": ys - ps, "rt": yt - pt,
        "sx": sx, "tx": tx, "numeric_features": nums,
    }


def champion_profiles(b, entity):
    sp = (b["sx"][b["numeric_features"]]
          .assign(**{entity: b["s"][entity].to_numpy()})
          .groupby(entity).mean())
    tp = (b["tx"][b["numeric_features"]]
          .assign(**{entity: b["t"][entity].to_numpy()})
          .groupby(entity).mean())
    return sp, tp
