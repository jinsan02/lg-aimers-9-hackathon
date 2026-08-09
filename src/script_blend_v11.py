"""Final candidate: career-middle calibration plus frozen pitcher-batter residuals."""

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
_F = [42, 7, 13, 3, 4, 5, 6, 8]
_C = [42, 7, 13, 3, 4, 5]
WEIGHTS = ([(os.path.join(SCRIPT_DIR, "model", f"cat_v14f_s{s}.pkl"),
             (1-_W_CELL)/len(_F)) for s in _F]
           + [(os.path.join(SCRIPT_DIR, "model", f"cat_ZD5_s{s}.pkl"),
               _W_CELL/len(_C)) for s in _C])

MID_COL = "asof_pitcher_middle_rate"
FINAL_PATH = os.path.join(SCRIPT_DIR, "model", "final_constants_2024.npz")


def middle_adjustment(test):
    z = np.load(FINAL_PATH)
    x = pd.to_numeric(test[MID_COL], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(x)
    adj = np.full(len(test), float(z["nan_offset"][0]), dtype=np.float64)
    adj[finite] = z["offsets"][np.searchsorted(z["thresholds"], x[finite], side="right")]
    return adj


def pb_adjustment(test):
    z = np.load(FINAL_PATH)
    table = {(int(p), int(b)): float(v) for p, b, v in
             zip(z["pb_pitcher"], z["pb_batter"], z["pb_offset"])}
    pitcher = pd.to_numeric(test["pitcher_id"], errors="coerce").fillna(-1).astype(np.int64)
    batter = pd.to_numeric(test["batter_id"], errors="coerce").fillna(-1).astype(np.int64)
    adj = np.fromiter((table.get((int(p), int(b)), 0.) for p, b in zip(pitcher, batter)),
                      dtype=np.float64, count=len(test))
    return adj


def base_blend(test):
    total = sum(w for _, w in WEIGHTS)
    preds = np.zeros(len(test), dtype=np.float64)
    for path, w in WEIGHTS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"model missing: {path}")
        p = fpipe.predict(joblib.load(path), test)
        preds += (w/total)*p
        print(f"  {os.path.basename(path)} (w={w:.4f}): mean={p.mean():.4f}")
    raw = preds.mean()
    q = np.clip(preds, 1e-6, 1-1e-6)
    preds = 1/(1+np.exp(-SLOPE*np.log(q/(1-q))))
    print(f"BLEND base mean={raw:.4f} -> slope x{SLOPE} -> {preds.mean():.4f} "
          f"-> SHIFT -{SHIFT:.4f}")
    return np.clip(preds-SHIFT, 0, 1)


def blend(test):
    p = base_blend(test)
    mid = middle_adjustment(test)
    pb = pb_adjustment(test)
    print(f"career-middle mean={mid.mean():+.6f}; "
          f"pitcher-batter coverage={(pb != 0).mean()*100:.1f}% mean={pb.mean():+.6f}")
    return np.clip(p+mid+pb, 0, 1)


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission columns mismatch: {list(sub.columns)}")
    p = blend(test)
    mp = dict(zip(test[ID_COL], p))
    sub[TARGET_COL] = [mp.get(r, v) for r, v in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved ./output/submission.csv rows={len(sub)} mean={p.mean():.4f}")


if __name__ == "__main__":
    main()
