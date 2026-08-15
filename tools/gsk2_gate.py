"""Six-seed untouched-season gate for GENERAL_SKILL_ADD on the cell arm."""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fpipe  # noqa: E402
import invalidated  # noqa: E402

SEEDS = [3, 4, 5, 6, 8, 13]
W_CELL = 0.55
ADDED = ["skill_hat", "skill_hat_vs_std"]


def read_npz(tag, seed, split):
    path = os.path.join(ROOT, "out", f"cat_{tag}_s{seed}_{split}_preds.npz")
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k].copy() for k in z.files}


def same(a, b, label):
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        raise SystemExit(f"elementwise mismatch: {label}")


def gain(y, p0, p1, denom):
    return float(1e5 * (np.mean((p0-y)**2)-np.mean((p1-y)**2))/denom)


def bss(y, p):
    d = float(y.mean()*(1-y.mean()))
    return float(1e5*(1-np.mean((p-y)**2)/d))


def murphy(y, p, bins=20):
    order = np.argsort(p)
    base = float(y.mean())
    rel = res = 0.0
    for idx in np.array_split(order, bins):
        w = len(idx)/len(y)
        rel += w*(float(p[idx].mean())-float(y[idx].mean()))**2
        res += w*(float(y[idx].mean())-base)**2
    return float(rel), float(res)


def check_feature_delta(control, candidate):
    """Require the candidate to add exactly the two pre-registered columns."""
    c0, c1 = list(control), list(candidate)
    if len(c1) != len(c0) + 2:
        raise ValueError(f"feature count {len(c0)} -> {len(c1)}, expected +2")
    if set(c1)-set(c0) != set(ADDED) or set(c0)-set(c1):
        raise ValueError("feature set delta is not exactly GENERAL_SKILL_ADD")
    if [x for x in c1 if x not in ADDED] != c0:
        raise ValueError("existing feature order changed")
    return {"control_n": len(c0), "candidate_n": len(c1), "added": ADDED}


def gate_verdict(mean, tval, ensemble_delta, segments, ci_upper):
    if (mean >= 3 and tval >= 2.4 and ensemble_delta > 0
            and all(segments[k] >= 0 for k in ("first", "second", "R", "F"))):
        return "KEEP"
    if ci_upper < 3:
        return "DROP"
    return "HOLD"


def artifact_parity(tag, seed, raw):
    pack_path = os.path.join(ROOT, "model", f"cat_{tag}_s{seed}.pkl")
    pack = joblib.load(pack_path)
    if [int(x) for x in pack.get("fm_success") or []] != [9, 10, 11]:
        raise SystemExit(f"{tag} s{seed}: fm_success drift")
    if pack.get("fm_multilabel"):
        raise SystemExit(f"{tag} s{seed}: unexpected multilabel route")
    if [int(x) for x in np.asarray(pack["model"].classes_)] != list(range(12)):
        raise SystemExit(f"{tag} s{seed}: class order drift")
    z = read_npz(tag, seed, "test")
    lookup = raw.set_index("row_id", drop=False)
    frame = lookup.loc[list(z["row_id"])].reset_index(drop=True)
    same(frame["row_id"].to_numpy(), z["row_id"], f"{tag} s{seed} raw order")
    pred = np.asarray(fpipe.predict(pack, frame), float)
    diff = float(np.max(np.abs(pred-z["pred"])))
    probe = frame.iloc[:800]
    full = np.asarray(fpipe.predict(pack, probe), float)
    rev = np.asarray(fpipe.predict(pack, probe.iloc[::-1]), float)[::-1]
    half = np.asarray(fpipe.predict(pack, probe.iloc[:400]), float)
    one = np.asarray(fpipe.predict(pack, probe.iloc[[0]]), float)
    drift = max(float(np.max(np.abs(full-rev))),
                float(np.max(np.abs(full[:400]-half))),
                float(abs(full[0]-one[0])))
    if diff != 0.0 or drift != 0.0:
        raise SystemExit(f"{tag} s{seed} parity diff={diff} drift={drift}")
    lin = pack.get("lineage") or {}
    return {"seed": seed, "maxdiff": diff, "subset_drift": drift,
            "best_iteration": int(pack["best_iteration"]),
            "fit_rowid_sha": lin.get("fit_rowid_sha"),
            "features_sha": lin.get("features_sha"),
            "features": list(pack["features"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="GSK2CTL_cell")
    ap.add_argument("--candidate", default="GSK2CAND_cell")
    ap.add_argument("--base", default="RMSE2CTL_base")
    ap.add_argument("--report", default="out/gsk2_gate.json")
    args = ap.parse_args()
    invalidated.guard([args.control, args.candidate, args.base])
    os.chdir(ROOT)

    rows, ctl_test, cand_test, base_test = [], [], [], []
    ctl_val, cand_val, base_val = [], [], []
    ref_ids = ref_y = None
    for seed in SEEDS:
        ct, ca, ba = (read_npz(args.control, seed, "test"),
                      read_npz(args.candidate, seed, "test"),
                      read_npz(args.base, seed, "test"))
        cv, av, bv = (read_npz(args.control, seed, "val"),
                      read_npz(args.candidate, seed, "val"),
                      read_npz(args.base, seed, "val"))
        for z, label in ((ca, "candidate"), (ba, "base")):
            same(ct["row_id"], z["row_id"], f"test {label} s{seed}")
            same(ct["y"], z["y"], f"test target {label} s{seed}")
        for z, label in ((av, "candidate"), (bv, "base")):
            same(cv["row_id"], z["row_id"], f"val {label} s{seed}")
            same(cv["y"], z["y"], f"val target {label} s{seed}")
        for z in (ct, ca, ba):
            if str(z["host"].item()) != "DESKTOP-053T952" or int(z["seed"]) != seed:
                raise SystemExit(f"host/seed mismatch s{seed}")
        if ref_ids is None:
            ref_ids, ref_y = ct["row_id"], ct["y"].astype(float)
        else:
            same(ref_ids, ct["row_id"], f"cross-seed row_id s{seed}")
            same(ref_y, ct["y"], f"cross-seed target s{seed}")
        y = ct["y"].astype(float); den = float(y.mean()*(1-y.mean()))
        pctl = (1-W_CELL)*ba["pred"].astype(float) + W_CELL*ct["pred"].astype(float)
        pcand = (1-W_CELL)*ba["pred"].astype(float) + W_CELL*ca["pred"].astype(float)
        vctl = (1-W_CELL)*bv["pred"].astype(float) + W_CELL*cv["pred"].astype(float)
        vcand = (1-W_CELL)*bv["pred"].astype(float) + W_CELL*av["pred"].astype(float)
        rows.append({"seed": seed,
                     "cell_delta": gain(y, ct["pred"], ca["pred"], den),
                     "core_delta": gain(y, pctl, pcand, den),
                     "source_core_delta": gain(cv["y"].astype(float), vctl, vcand,
                         float(cv["y"].mean()*(1-cv["y"].mean())))})
        ctl_test.append(ct["pred"].astype(float)); cand_test.append(ca["pred"].astype(float))
        base_test.append(ba["pred"].astype(float))
        ctl_val.append(cv["pred"].astype(float)); cand_val.append(av["pred"].astype(float))
        base_val.append(bv["pred"].astype(float))

    raw = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    ctl_parity = [artifact_parity(args.control, s, raw) for s in SEEDS]
    cand_parity = [artifact_parity(args.candidate, s, raw) for s in SEEDS]
    base_packs = [joblib.load(os.path.join(ROOT, "model", f"cat_{args.base}_s{s}.pkl"))
                  for s in SEEDS]
    hashes = {p["fit_rowid_sha"] for p in ctl_parity+cand_parity}
    hashes |= {(p.get("lineage") or {}).get("fit_rowid_sha") for p in base_packs}
    if len(hashes) != 1 or None in hashes:
        raise SystemExit(f"fit-row lineage mismatch: {hashes}")
    deltas = [check_feature_delta(c["features"], a["features"])
              for c, a in zip(ctl_parity, cand_parity)]
    if any(d != deltas[0] for d in deltas) or deltas[0]["control_n"] != 121:
        raise SystemExit(f"feature delta mismatch: {deltas}")
    ctl_sha = {p["features_sha"] for p in ctl_parity}
    base_sha = {(p.get("lineage") or {}).get("features_sha") for p in base_packs}
    cand_sha = {p["features_sha"] for p in cand_parity}
    if len(ctl_sha) != 1 or ctl_sha != base_sha or len(cand_sha) != 1 or ctl_sha == cand_sha:
        raise SystemExit("feature lineage mismatch")

    delta = np.array([r["core_delta"] for r in rows])
    mean = float(delta.mean()); se = float(delta.std(ddof=1)/np.sqrt(len(delta)))
    tval = mean/se if se else float("inf")
    crit = float(student_t.ppf(.975, len(delta)-1)); ci = [mean-crit*se, mean+crit*se]
    y = ref_y; den = float(y.mean()*(1-y.mean()))
    cell0, cell1, base = map(lambda x: np.mean(x, axis=0),
                             (ctl_test, cand_test, base_test))
    p0 = (1-W_CELL)*base + W_CELL*cell0
    p1 = (1-W_CELL)*base + W_CELL*cell1
    ensemble_delta = gain(y, p0, p1, den)
    gt = pd.Series(raw.game_type.to_numpy(), index=raw.row_id).reindex(ref_ids)
    if gt.isna().any():
        raise SystemExit("game_type alignment failed")
    n = len(y); first = np.arange(n) < n//2
    masks = {"first": first, "second": ~first,
             "R": gt.to_numpy() == "R", "F": gt.to_numpy() == "F"}
    segments = {k: gain(y[m], p0[m], p1[m], den) for k, m in masks.items()}
    rel0, res0 = murphy(y, p0); rel1, res1 = murphy(y, p1)
    verdict = gate_verdict(mean, tval, ensemble_delta, segments, ci[1])
    report = {"rows": rows, "mean": mean, "se": se, "t": tval,
              "ci95_student": ci, "median": float(np.median(delta)),
              "positive": int((delta > 0).sum()), "ensemble_delta": ensemble_delta,
              "segments": segments, "control_bss": bss(y, p0),
              "candidate_bss": bss(y, p1), "reliability": [rel0, rel1],
              "resolution": [res0, res1],
              "rms": float(np.sqrt(np.mean((p1-p0)**2))),
              "pearson": float(np.corrcoef(p0, p1)[0,1]),
              "feature_delta": deltas[0], "control_parity": ctl_parity,
              "candidate_parity": cand_parity, "verdict": verdict}
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2)); print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
