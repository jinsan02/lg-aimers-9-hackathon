"""v9 champion plus a frozen recent-middle residual correction.

The thresholds and offsets were estimated only from the 2024 submission-
equivalent validation surface.  Every test row is adjusted independently;
no statistic is computed from the evaluation batch.
"""

import os

import joblib
import numpy as np
import pandas as pd

import fpipe

ID_COL = "row_id"
TARGET_COL = "control_success"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHIFT = 0.0052
SLOPE = 1.0416
_W_CELL = 0.55
_F = [42, 7, 13, 3, 4, 5, 6, 8]
_C = [42, 7, 13, 3, 4, 5]
WEIGHTS = ([(os.path.join(SCRIPT_DIR, "model", f"cat_v14f_s{s}.pkl"),
             (1 - _W_CELL) / len(_F)) for s in _F]
           + [(os.path.join(SCRIPT_DIR, "model", f"cat_ZD5_s{s}.pkl"),
               _W_CELL / len(_C)) for s in _C])
COL = "asof_pitcher_prev5_game_middle_rate"
THRESHOLDS = np.asarray([
    0.114286, 0.141026, 0.159091, 0.174312,
    0.188889, 0.203837, 0.227273,
], dtype=np.float64)
OFFSETS = np.asarray([
    0.009128596327376929,
    0.0014582307357119428,
    0.0016645796882829116,
    0.0031305197564435523,
    0.00028882666473504875,
    -0.005475006685108483,
    -0.00413854957909932,
    -0.005074360453772326,
], dtype=np.float64)
NAN_OFFSET = -0.008102692247033697


def middle_adjustment(test):
    x = pd.to_numeric(test[COL], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(x)
    adj = np.full(len(test), NAN_OFFSET, dtype=np.float64)
    bins = np.searchsorted(THRESHOLDS, x[finite], side="right")
    adj[finite] = OFFSETS[bins]
    return adj


def base_blend(test):
    total = sum(w for _, w in WEIGHTS)
    preds = np.zeros(len(test), dtype=np.float64)
    for path, w in WEIGHTS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"model missing: {path}")
        p = fpipe.predict(joblib.load(path), test)
        preds += (w / total) * p
        print(f"  {os.path.basename(path)} (w={w:.4f}): mean={p.mean():.4f}")
    raw = preds.mean()
    z = np.log(np.clip(preds, 1e-6, 1 - 1e-6) /
               (1 - np.clip(preds, 1e-6, 1 - 1e-6)))
    preds = 1.0 / (1.0 + np.exp(-SLOPE * z))
    print(f"BLEND base mean={raw:.4f} -> slope x{SLOPE} -> {preds.mean():.4f} "
          f"-> SHIFT -{SHIFT:.4f}")
    return np.clip(preds - SHIFT, 0.0, 1.0)


def blend(test):
    preds = base_blend(test)
    adj = middle_adjustment(test)
    print(f"recent-middle correction: mean={adj.mean():+.6f} "
          f"min={adj.min():+.6f} max={adj.max():+.6f}")
    return np.clip(preds + adj, 0.0, 1.0)


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission columns mismatch: {list(sub.columns)}")
    print(f"test={len(test)} submission={len(sub)}")
    preds = blend(test)
    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)}) mean={preds.mean():.4f}")


if __name__ == "__main__":
    main()
