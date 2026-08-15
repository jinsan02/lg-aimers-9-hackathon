"""Pre-registered F1 offset-logistic adapter and untouched-season gate."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fpipe  # noqa: E402
import invalidated  # noqa: E402
from segment_resolution import honest_rank_resolution  # noqa: E402

FEATURES = [
    "std_asof_pitcher_ball_rate",
    "std_asof_pitcher_ball_rate_delta",
    "std_asof_pitcher_middle_rate",
    "std_asof_pitcher_middle_rate_delta",
    "balls_before",
    "strikes_before",
]
L2 = 100.0
W_CELL = 0.55


def read_npz(tag, split):
    path = os.path.join(ROOT, "out", f"cat_{tag}_{split}_preds.npz")
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k].copy() for k in z.files}


def require_same(a, b, name):
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        raise SystemExit(f"elementwise mismatch: {name}")


def expit(x):
    x = np.asarray(x, np.float64)
    out = np.empty_like(x)
    m = x >= 0
    out[m] = 1.0 / (1.0 + np.exp(-x[m]))
    e = np.exp(x[~m])
    out[~m] = e / (1.0 + e)
    return out


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def fit_adapter(frame, p0, y):
    x0 = frame[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(x0, axis=0)
    if not np.isfinite(med).all():
        raise SystemExit("F1 source median is non-finite")
    x0 = np.where(np.isfinite(x0), x0, med)
    mean, scale = x0.mean(axis=0), x0.std(axis=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise SystemExit(f"F1 zero/nonfinite scale: {scale}")
    x = (x0 - mean) / scale
    off = logit(p0)

    def fun(beta):
        q = expit(off + x @ beta)
        eps = 1e-12
        loss = -np.sum(y * np.log(q + eps) + (1-y) * np.log(1-q + eps))
        return float(loss + 0.5 * L2 * np.dot(beta, beta))

    def jac(beta):
        q = expit(off + x @ beta)
        return x.T @ (q - y) + L2 * beta

    opt = minimize(fun, np.zeros(x.shape[1]), jac=jac, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    if not opt.success or not np.isfinite(opt.x).all():
        raise SystemExit(f"F1 optimizer failed: {opt.message}")
    return {"features": FEATURES, "median": med, "mean": mean,
            "scale": scale, "beta": opt.x, "l2": L2,
            "intercept": 0.0, "optimizer": str(opt.message),
            "iterations": int(opt.nit)}


def apply_adapter(artifact, frame, p0):
    x = frame[artifact["features"]].apply(
        pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.asarray(artifact["median"], float)
    x = np.where(np.isfinite(x), x, med)
    x = (x - np.asarray(artifact["mean"], float)) / np.asarray(
        artifact["scale"], float)
    return expit(logit(p0) + x @ np.asarray(artifact["beta"], float))


def bss(y, p, denom=None):
    denom = float(y.mean() * (1-y.mean())) if denom is None else denom
    return float(1e5 * (1 - np.mean((p-y)**2) / denom))


def gain(y, p0, p1, denom):
    return float(1e5 * (np.mean((p0-y)**2)-np.mean((p1-y)**2)) / denom)


def murphy(y, p, bins=20):
    order = np.argsort(p)
    base = float(y.mean())
    rel = res = 0.0
    for idx in np.array_split(order, bins):
        w = len(idx)/len(y)
        rel += w * (float(p[idx].mean())-float(y[idx].mean()))**2
        res += w * (float(y[idx].mean())-base)**2
    return float(rel), float(res)


def transformed_frame(raw, pack, ids):
    lookup = raw.set_index("row_id", drop=False)
    try:
        frame = lookup.loc[list(ids)].reset_index(drop=True)
    except KeyError as e:
        raise SystemExit(f"raw row_id alignment failed: {e}")
    require_same(frame["row_id"].to_numpy(), ids, "raw row order")
    return fpipe.transform(frame.copy(), pack["fpipe"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="B1SMOKE_base")
    ap.add_argument("--cell", default="B1SMOKE_cell")
    ap.add_argument("--artifact", default="model/f1_adapter.pkl")
    ap.add_argument("--report", default="out/f1_gate.json")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    invalidated.guard([args.base, args.cell])
    os.chdir(ROOT)

    if args.audit_only:
        art = joblib.load(args.artifact)
        pack = joblib.load(f"model/cat_{args.base}.pkl")
        raw = pd.read_csv("data/train.csv", encoding="utf-8-sig")
        zb, zc = read_npz(args.base, "test"), read_npz(args.cell, "test")
        require_same(zb["row_id"], zc["row_id"], "audit row_id")
        frame = transformed_frame(raw, pack, zb["row_id"])
        mask = frame["game_type"].astype(str).to_numpy() == "F"
        p0 = (1-W_CELL)*zb["pred"].astype(float) + W_CELL*zc["pred"].astype(float)
        pred = p0.copy()
        pred[mask] = apply_adapter(art, frame.loc[mask], p0[mask])
        with np.load("out/f1_candidate_test.npz", allow_pickle=True) as z:
            require_same(z["row_id"], zb["row_id"], "saved candidate row_id")
            maxdiff = float(np.max(np.abs(pred-z["pred"])))
        if maxdiff != 0.0:
            raise SystemExit(f"fresh-process full maxdiff {maxdiff:.3e}")
        probe = np.flatnonzero(mask)[:800]
        full = apply_adapter(art, frame.iloc[probe], p0[probe])
        rev = apply_adapter(art, frame.iloc[probe[::-1]], p0[probe[::-1]])[::-1]
        half = apply_adapter(art, frame.iloc[probe[:400]], p0[probe[:400]])
        one = apply_adapter(art, frame.iloc[[probe[0]]], p0[[probe[0]]])
        drift = max(float(np.max(np.abs(full-rev))),
                    float(np.max(np.abs(full[:400]-half))),
                    float(abs(full[0]-one[0])))
        if drift != 0.0:
            raise SystemExit(f"subset drift {drift:.3e}")
        print(f"F1 fresh-process full maxdiff {maxdiff:.3e} "
              f"subset drift {drift:.3e}")
        return 0

    base_pack = joblib.load(f"model/cat_{args.base}.pkl")
    cell_pack = joblib.load(f"model/cat_{args.cell}.pkl")
    if base_pack["features"] != cell_pack["features"] or len(base_pack["features"]) != 121:
        raise SystemExit("B1SMOKE feature contract mismatch")
    if any(c not in base_pack["features"] for c in FEATURES):
        raise SystemExit("F1 frozen feature missing from B1SMOKE")

    zvb, zvc = read_npz(args.base, "val"), read_npz(args.cell, "val")
    ztb, ztc = read_npz(args.base, "test"), read_npz(args.cell, "test")
    for x, y, split in ((zvb, zvc, "val"), (ztb, ztc, "test")):
        require_same(x["row_id"], y["row_id"], f"{split} row_id")
        require_same(x["y"], y["y"], f"{split} target")
        require_same(x["host"], y["host"], f"{split} host")
        require_same(x["seed"], y["seed"], f"{split} seed")
    if str(ztb["host"].item()) != "DESKTOP-053T952" or int(ztb["seed"]) != 3:
        raise SystemExit("F1 must use 5070 seed-3 B1SMOKE")

    raw = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    fv = transformed_frame(raw, base_pack, zvb["row_id"])
    ft = transformed_frame(raw, base_pack, ztb["row_id"])
    yv, yt = zvb["y"].astype(float), ztb["y"].astype(float)
    require_same(fv["control_success"].to_numpy(float), yv, "val raw target")
    require_same(ft["control_success"].to_numpy(float), yt, "test raw target")
    pv = (1-W_CELL)*zvb["pred"].astype(float) + W_CELL*zvc["pred"].astype(float)
    pt = (1-W_CELL)*ztb["pred"].astype(float) + W_CELL*ztc["pred"].astype(float)
    mv = fv["game_type"].astype(str).to_numpy() == "F"
    mt = ft["game_type"].astype(str).to_numpy() == "F"

    art = fit_adapter(fv.loc[mv], pv[mv], yv[mv])
    art.update({"base_tag": args.base, "cell_tag": args.cell,
                "source_season": 2023, "target_season": 2024,
                "host": socket.gethostname(), "success_set": None})
    joblib.dump(art, args.artifact)
    qv, qt = pv.copy(), pt.copy()
    qv[mv] = apply_adapter(art, fv.loc[mv], pv[mv])
    qt[mt] = apply_adapter(art, ft.loc[mt], pt[mt])
    if not np.array_equal(qt[~mt], pt[~mt]):
        raise SystemExit("F1 changed an R prediction")
    if not np.isfinite(qt).all() or qt.min() < 0 or qt.max() > 1:
        raise SystemExit("F1 predictions outside probability contract")

    denom = float(yt.mean()*(1-yt.mean()))
    n = len(yt)
    first = np.arange(n) < n//2
    early = ft["game_month"].to_numpy() <= 6
    segments = {"first": first, "second": ~first, "early": early,
                "late": ~early, "R": ~mt, "F": mt}
    seg = {k: gain(yt[m], pt[m], qt[m], denom) for k, m in segments.items()}
    rng0 = np.random.default_rng(20260808)
    rng1 = np.random.default_rng(20260808)
    # `honest_rank_resolution` already returns the within-segment gain on the
    # global Brier denominator.  This is the table's `hon/norm` quantity;
    # multiplying by segment share would turn it into full-score contribution.
    h0 = honest_rank_resolution(pt[mt], yt[mt], denom, rng0)
    h1 = honest_rank_resolution(qt[mt], yt[mt], denom, rng1)
    rel0, res0 = murphy(yt, pt)
    rel1, res1 = murphy(yt, qt)
    report = {
        "source_delta": gain(yv, pv, qv, float(yv.mean()*(1-yv.mean()))),
        "target_control_bss": bss(yt, pt), "target_candidate_bss": bss(yt, qt),
        "target_delta": gain(yt, pt, qt, denom), "segments": seg,
        "F_honest_norm_control": h0, "F_honest_norm_candidate": h1,
        "F_honest_norm_gain": h1-h0, "reliability": [rel0, rel1],
        "resolution": [res0, res1],
        "rms": float(np.sqrt(np.mean((qt-pt)**2))),
        "pearson": float(np.corrcoef(pt, qt)[0,1]),
        "beta": np.asarray(art["beta"]).tolist(),
        "correction_sd_F": float(np.std(logit(qt[mt])-logit(pt[mt]))),
        "R_max_change": float(np.max(np.abs(qt[~mt]-pt[~mt]))),
        "n": n, "n_F": int(mt.sum()), "host": socket.gethostname(),
    }
    halves_ok = all(seg[k] >= 0 for k in ("first", "second", "early", "late"))
    passed = (report["target_delta"] >= 10 and report["F_honest_norm_gain"] >= 100
              and halves_ok)
    report["verdict"] = "PASS" if passed else "FAIL"
    np.savez_compressed("out/f1_candidate_test.npz",
                        row_id=ztb["row_id"], y=yt, pred=qt)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    cp = subprocess.run([sys.executable, os.path.abspath(__file__),
                         "--base", args.base, "--cell", args.cell,
                         "--artifact", args.artifact, "--audit-only"],
                        cwd=ROOT)
    if cp.returncode:
        raise SystemExit("F1 fresh-process audit failed")
    print(f"VERDICT: {report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
