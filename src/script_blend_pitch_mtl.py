"""v11 champion + legal masked-pitch auxiliary NN (10% fixed blend).

Pitch type was an auxiliary label only during training.  This inference script
never loads Trackman data and predicts each evaluation row independently from
the competition test columns and frozen training artifacts.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import torch

import fpipe


ID_COL = "row_id"
TARGET_COL = "control_success"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
PITCH_DIR = os.path.join(MODEL_DIR, "pitch_mtl")
_W_CELL = 0.55
SLOPE, SHIFT = 1.0416, 0.0052  # exposed for the standard row-independence audit
_F = [42, 7, 13, 3, 4, 5, 6, 8]
_C = [42, 7, 13, 3, 4, 5]
WEIGHTS = ([(os.path.join(MODEL_DIR, f"cat_v14f_s{s}.pkl"),
             (1 - _W_CELL) / len(_F)) for s in _F]
           + [(os.path.join(MODEL_DIR, f"cat_ZD5_s{s}.pkl"),
               _W_CELL / len(_C)) for s in _C])
NN_SEEDS = [3, 4, 5, 6, 8, 13]
NN_BATCH = 4096  # fixed/padded shape removes batch-size-dependent GEMM roundoff
CORRECTIONS = os.path.join(PITCH_DIR, "corrections_2024.npz")
PREPROCESS = os.path.join(PITCH_DIR, "preprocess.pkl")
MID_COL = "asof_pitcher_prev5_game_middle_rate"


def encode_categories(df, columns, vocab):
    arrays = []
    for col in columns:
        cats = vocab[col]
        unknown = len(cats)
        mapping = {value: i for i, value in enumerate(cats)}
        arrays.append(df[col].astype(str).map(mapping).fillna(unknown)
                      .to_numpy(np.int64))
    return (np.stack(arrays, 1) if arrays else
            np.zeros((len(df), 0), dtype=np.int64))


def cat_core(test):
    pred = np.zeros(len(test), dtype=np.float64)
    total = sum(weight for _, weight in WEIGHTS)
    for path, weight in WEIGHTS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"model missing: {path}")
        pred += weight / total * fpipe.predict(joblib.load(path), test)
    return pred


def pitch_aux_predict(test):
    pack = joblib.load(PREPROCESS)
    meta, qt, qbins = pack["feature_meta"], pack["qt"], int(pack["qbins"])
    frame = fpipe.transform(test.copy(), meta["fpipe"])
    xn = np.nan_to_num(frame[meta["num"]].to_numpy(np.float32),
                       nan=0.0, posinf=0.0, neginf=0.0)
    xn = qt.transform(xn).astype(np.float32)
    xq = np.clip((xn * 4 + qbins / 2).astype(np.int64), 0, qbins - 1)
    xc = encode_categories(frame, meta["cat_cols"], meta["vocab"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = np.zeros(len(test), dtype=np.float64)
    with torch.inference_mode():
        for seed in NN_SEEDS:
            path = os.path.join(PITCH_DIR, f"pitch_aux_s{seed}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"model missing: {path}")
            net = torch.jit.load(path, map_location=device).eval()
            member = []
            for i in range(0, len(test), NN_BATCH):
                take = min(NN_BATCH, len(test) - i)
                # Always execute the same matrix shape.  Otherwise cuBLAS may
                # choose a different kernel for a one-row audit and differ by
                # a few float32 ulps even though no row information is shared.
                bn = np.zeros((NN_BATCH, xn.shape[1]), dtype=np.float32)
                bc = np.zeros((NN_BATCH, xc.shape[1]), dtype=np.int64)
                bq = np.zeros((NN_BATCH, xq.shape[1]), dtype=np.int64)
                bn[:take], bc[:take], bq[:take] = (xn[i:i + take],
                                                   xc[i:i + take],
                                                   xq[i:i + take])
                output = net(torch.from_numpy(bn).to(device),
                             torch.from_numpy(bc).to(device),
                             torch.from_numpy(bq).to(device))
                member.append(torch.sigmoid(output[0][:take]).cpu().numpy())
            preds += np.concatenate(member).astype(np.float64) / len(NN_SEEDS)
            del net
    return preds


def corrections(test, constants):
    x = pd.to_numeric(test[MID_COL], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(x)
    mid = np.full(len(test), float(constants["middle_nan_offset"]),
                  dtype=np.float64)
    mid[finite] = constants["middle_offsets"][np.searchsorted(
        constants["middle_thresholds"], x[finite], side="right")]
    table = {(int(p), int(b)): float(v) for p, b, v in zip(
        constants["pb_pitcher"], constants["pb_batter"],
        constants["pb_offset"])}
    pitcher = pd.to_numeric(test["pitcher_id"], errors="coerce").fillna(-1)
    batter = pd.to_numeric(test["batter_id"], errors="coerce").fillna(-1)
    pb = np.fromiter((table.get((int(p), int(b)), 0.0)
                      for p, b in zip(pitcher, batter)),
                     dtype=np.float64, count=len(test))
    return mid, pb


def blend(test):
    constants = np.load(CORRECTIONS)
    cat = cat_core(test)
    nn = pitch_aux_predict(test)
    weight = float(constants["nn_weight"])
    raw = np.clip((1 - weight) * cat + weight * nn, 1e-6, 1 - 1e-6)
    slope, shift = float(constants["slope"]), float(constants["shift"])
    pred = 1 / (1 + np.exp(-slope * np.log(raw / (1 - raw)))) - shift
    mid, pb = corrections(test, constants)
    return np.clip(pred + mid + pb, 0, 1)


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission columns mismatch: {list(sub.columns)}")
    pred = blend(test)
    mapping = dict(zip(test[ID_COL], pred))
    sub[TARGET_COL] = [mapping.get(row, old) for row, old in
                       zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"saved ./output/submission.csv rows={len(sub)} mean={pred.mean():.6f}")


if __name__ == "__main__":
    main()
