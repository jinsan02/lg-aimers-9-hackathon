"""B1-S submission: the post-audit rebuild, with v11's post-processing kept.

Members are the `B1S` family — `--p1` (two-stage artifact, fit-only TE prior,
neutral first-season skill), 8 base + 6 cell seeds, trained on the submission
surface (val2024, no `--drop-f-pre`).

Post-processing is v11's, unchanged, and that is a measured decision rather than
inertia. Every constant was re-derived from new OOF on 2026-08-13 and every
refit lost to the constant already here:

    calibration   refit fitted on 2023, frozen onto 2024   -19.93  vs legacy -1.22
    recent-middle refit fitted on 2023, frozen onto 2024    +5.72  vs legacy +8.03
    blend weight  optimum fitted on 2023, frozen            -0.81  vs .55

On the submission surface the shipped middle offsets reproduce from new OOF to
about 1e-4 with 8/8 sign agreement, so there was nothing to correct.

Two things this script does that `script_blend_v11.py` did not:
  * it refuses a submission whose ids do not match the sample exactly, instead
    of silently leaving the sample's own value in place for a missing row;
  * it names its member seeds and fails if one is absent, rather than blending
    whatever happens to be on disk.
"""

import os

import joblib
import numpy as np
import pandas as pd

import fpipe

ID_COL = "row_id"
TARGET_COL = "control_success"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SHIFT, SLOPE = 0.0052, 1.0416
_W_CELL = 0.55
# 8 base / 6 cell. Adding 42 and 7 to the base family is worth +0.89 on the
# debiased local score at the stage the +139.03 LB offset was measured; the
# same two seeds on the cell family are worth -0.13, so the cell stays at 6.
_BASE_SEEDS = [3, 4, 5, 6, 8, 13, 42, 7]
_CELL_SEEDS = [3, 4, 5, 6, 8, 13]
WEIGHTS = ([(os.path.join(SCRIPT_DIR, "model", f"cat_B1S_base_s{s}.pkl"),
             (1 - _W_CELL) / len(_BASE_SEEDS)) for s in _BASE_SEEDS]
           + [(os.path.join(SCRIPT_DIR, "model", f"cat_B1S_cell_s{s}.pkl"),
               _W_CELL / len(_CELL_SEEDS)) for s in _CELL_SEEDS])

MID_COL = "asof_pitcher_prev5_game_middle_rate"
THRESHOLDS = np.asarray([0.114286, 0.141026, 0.159091, 0.174312,
                         0.188889, 0.203837, 0.227273], dtype=np.float64)
OFFSETS = np.asarray([0.009128596327376929, 0.0014582307357119428,
                      0.0016645796882829116, 0.0031305197564435523,
                      0.00028882666473504875, -0.005475006685108483,
                      -0.00413854957909932, -0.005074360453772326])
NAN_OFFSET = -0.008102692247033697
MATCHUP_PATH = os.path.join(SCRIPT_DIR, "model", "matchup_constants_2024.npz")


def middle_adjustment(test):
    x = pd.to_numeric(test[MID_COL], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(x)
    adj = np.full(len(test), NAN_OFFSET, dtype=np.float64)
    adj[finite] = OFFSETS[np.searchsorted(THRESHOLDS, x[finite], side="right")]
    return adj


def pb_adjustment(test):
    """Frozen pitcher x batter residuals. Measured worth out of sample: +3.80.

    The generator's own `pb_raw_increment: 425.73` is an in-sample figure --
    fitted on val2023 and frozen onto 2024 the step gives +3.80, because only
    38.1% of rows a season later match a pair the table has seen. Kept because
    +3.80 is real and positive; recorded here so nobody reads 425.73 as headroom.
    """
    z = np.load(MATCHUP_PATH)
    table = {(int(p), int(b)): float(v) for p, b, v in
             zip(z["pb0_pitcher"], z["pb0_batter"], z["pb0_offset"])}
    pitcher = pd.to_numeric(test["pitcher_id"], errors="coerce").fillna(-1).astype(np.int64)
    batter = pd.to_numeric(test["batter_id"], errors="coerce").fillna(-1).astype(np.int64)
    return np.fromiter((table.get((int(p), int(b)), 0.) for p, b in zip(pitcher, batter)),
                       dtype=np.float64, count=len(test))


def base_blend(test):
    total = sum(w for _, w in WEIGHTS)
    preds = np.zeros(len(test), dtype=np.float64)
    for path, w in WEIGHTS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"model missing: {path}")
        p = fpipe.predict(joblib.load(path), test)
        if not np.isfinite(p).all():
            raise ValueError(f"non-finite predictions from {os.path.basename(path)}")
        preds += (w / total) * p
        print(f"  {os.path.basename(path)} (w={w:.4f}): mean={p.mean():.4f}")
    raw = preds.mean()
    q = np.clip(preds, 1e-6, 1 - 1e-6)
    preds = 1 / (1 + np.exp(-SLOPE * np.log(q / (1 - q))))
    print(f"BLEND base mean={raw:.4f} -> slope x{SLOPE} -> {preds.mean():.4f} "
          f"-> SHIFT -{SHIFT:.4f}")
    return np.clip(preds - SHIFT, 0, 1)


def blend(test):
    p = base_blend(test)
    mid = middle_adjustment(test)
    pb = pb_adjustment(test)
    print(f"recent-middle mean={mid.mean():+.6f}; "
          f"pitcher-batter coverage={(pb != 0).mean() * 100:.1f}% mean={pb.mean():+.6f}")
    return np.clip(p + mid + pb, 0, 1)


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission columns mismatch: {list(sub.columns)}")

    # v11 mapped predictions onto the sample and left the sample's own value
    # wherever a row was missing. That turns a broken join into a plausible
    # submission. Fail instead.
    tid, sid = test[ID_COL], sub[ID_COL]
    if tid.duplicated().any():
        raise ValueError(f"test.csv has {int(tid.duplicated().sum())} duplicate {ID_COL}")
    if sid.duplicated().any():
        raise ValueError(f"sample has {int(sid.duplicated().sum())} duplicate {ID_COL}")
    if set(tid) != set(sid):
        raise ValueError(f"id sets differ: {len(set(sid) - set(tid))} sample ids absent "
                         f"from test, {len(set(tid) - set(sid))} test ids absent from sample")

    p = blend(test)
    mp = dict(zip(tid, p))
    out = np.asarray([mp[r] for r in sid], dtype=np.float64)
    if not np.isfinite(out).all() or out.min() < 0 or out.max() > 1:
        raise ValueError(f"predictions out of range: min {out.min()} max {out.max()}")
    sub[TARGET_COL] = out
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved ./output/submission.csv rows={len(sub)} mean={out.mean():.4f}")


if __name__ == "__main__":
    main()
