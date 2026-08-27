"""LEAFIT seed-3 gate: cell-arm `leaf_estimation_iterations` 1 -> 10.

Contract: `docs/NEXT_GPU_PREREGISTRATION_20260828.md`, including its pre-fit
amendment (judging surface, laptop host). Parity is asserted before any BSS is
printed; a parity failure is an invalidated comparison, not a model verdict.

    python tools/leafit_gate.py [--seeds 3,4,5,6,8,13]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import invalidated  # noqa: E402
import judge  # noqa: E402  -- the adoption rule lives there, unmodified

W_CELL = 0.55
BASE = "LEAFIT_base"
CTRL = "LEAFITCTL_cell"
CAND = "LEAFIT_cell"
RMS_REDUNDANT = 0.002   # fixed in the preregistration, before any fit


def _stem(tag, seed):
    """A single-seed run drops the `_s<seed>` suffix; a multi-seed run keeps it.

    Getting this wrong is silent -- the file simply is not there -- so resolve it
    once and let a genuinely missing artifact raise with both names shown.
    """
    for cand in (f"cat_{tag}_s{seed}", f"cat_{tag}"):
        if os.path.exists(f"model/{cand}.pkl"):
            return cand
    raise FileNotFoundError(f"neither model/cat_{tag}_s{seed}.pkl nor "
                            f"model/cat_{tag}.pkl exists")


def arrays(tag, seed, split):
    z = np.load(f"out/{_stem(tag, seed)}_{split}_preds.npz", allow_pickle=True)
    return z["pred"].astype(float), z["y"].astype(float), z["row_id"]


def murphy(y, p, bins=20):
    order = np.argsort(p)
    base = float(y.mean())
    rel = res = 0.0
    for idx in np.array_split(order, bins):
        w = len(idx) / len(y)
        rel += w * (float(p[idx].mean()) - float(y[idx].mean())) ** 2
        res += w * (float(y[idx].mean()) - base) ** 2
    return rel, res


def effective(tag, seed):
    pk = joblib.load(f"model/{_stem(tag, seed)}.pkl")
    a = pk["model"].get_all_params()
    return {"leaf_estimation_iterations": a.get("leaf_estimation_iterations"),
            "loss_function": a.get("loss_function"), "depth": a.get("depth"),
            "l2_leaf_reg": a.get("l2_leaf_reg"),
            "border_count": a.get("border_count"),
            "max_ctr_complexity": a.get("max_ctr_complexity"),
            "random_seed": a.get("random_seed"), "task_type": a.get("task_type"),
            "bootstrap_type": a.get("bootstrap_type"),
            "leaf_estimation_method": a.get("leaf_estimation_method"),
            "grow_policy": a.get("grow_policy"),
            "n_features": len(pk.get("features") or []),
            "features": tuple(pk.get("features") or []),
            "cat_cols": tuple(pk.get("cat_cols") or []),
            "fm_success": tuple(pk.get("fm_success") or ()),
            "best_iteration": pk.get("best_iteration")}


def parity(seed, split):
    """Everything that must match before a score may be read."""
    problems = []
    ids = {}
    for tag in (BASE, CTRL, CAND):
        _, y, rid = arrays(tag, seed, split)
        ids[tag] = (rid, y)
    r0, y0 = ids[BASE]
    for tag in (CTRL, CAND):
        r, y = ids[tag]
        if not np.array_equal(r, r0):
            problems.append(f"{tag}: row_id differs from {BASE}")
        if not np.array_equal(y, y0):
            problems.append(f"{tag}: target differs from {BASE}")

    ec, ek = effective(CTRL, seed), effective(CAND, seed)
    if ec["features"] != ek["features"]:
        problems.append("cell feature list/order differs")
    if ec["cat_cols"] != ek["cat_cols"]:
        problems.append("cell categorical columns differ")
    if ec["fm_success"] != ek["fm_success"]:
        problems.append(f"fm_success differs {ec['fm_success']} vs {ek['fm_success']}")
    if ec["fm_success"] != (9, 10, 11):
        problems.append(f"fm_success is {ec['fm_success']}, expected (9,10,11)")

    diff = {k: (ec[k], ek[k]) for k in ec
            if k not in ("features", "cat_cols", "best_iteration")
            and ec[k] != ek[k]}
    if set(diff) != {"leaf_estimation_iterations"}:
        problems.append(f"effective-param diff is not exactly "
                        f"leaf_estimation_iterations: {diff}")
    if ec["leaf_estimation_iterations"] != 1:
        problems.append(f"control reports {ec['leaf_estimation_iterations']}, expected 1")
    if ek["leaf_estimation_iterations"] != 10:
        problems.append(f"candidate reports {ek['leaf_estimation_iterations']}, expected 10")
    return problems, ec, ek, len(r0)


def one_seed(seed, split, seg):
    b, y, rid = arrays(BASE, seed, split)
    c, _, _ = arrays(CTRL, seed, split)
    k, _, _ = arrays(CAND, seed, split)
    pc = judge.debias((1 - W_CELL) * b + W_CELL * c)
    pk = judge.debias((1 - W_CELL) * b + W_CELL * k)
    out = {
        "base_bss": judge.bss(judge.debias(b), y),
        "ctrl_cell_bss": judge.bss(judge.debias(c), y),
        "cand_cell_bss": judge.bss(judge.debias(k), y),
        "ctrl_core_bss": judge.bss(pc, y),
        "cand_core_bss": judge.bss(pk, y),
        "cell_rms": float(np.sqrt(np.mean((k - c) ** 2))),
        "core_rms": float(np.sqrt(np.mean((pk - pc) ** 2))),
        "pearson": float(np.corrcoef(pk, pc)[0, 1]),
        "mean_pred_diff": float(pk.mean() - pc.mean()),
    }
    out["core_delta"] = out["cand_core_bss"] - out["ctrl_core_bss"]
    out["cell_delta"] = out["cand_cell_bss"] - out["ctrl_cell_bss"]
    rc, sc = murphy(y, pc)
    rk, sk = murphy(y, pk)
    out.update(ctrl_reliability=rc, ctrl_resolution=sc,
               cand_reliability=rk, cand_resolution=sk,
               d_reliability=rk - rc, d_resolution=sk - sc)
    if seg is not None:
        g = pd.Series(rid).map(seg["league"]).to_numpy()
        m = pd.Series(rid).map(seg["month"]).to_numpy(dtype=float)
        for name, mask in (("R", g == "R"), ("F", g == "F"),
                           ("early", m <= 6), ("late", m > 6)):
            mask = np.asarray(mask, bool)
            if mask.sum() > 100:
                out[f"seg_{name}"] = judge.bss(pk[mask], y[mask]) - judge.bss(pc[mask], y[mask])
                out[f"n_{name}"] = int(mask.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3")
    ap.add_argument("--split", default="test",
                    help="test = the untouched season, the primary evidence")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s]

    tr = pd.read_csv("data/train.csv", usecols=["row_id", "game_type", "game_month"])
    seg = {"league": dict(zip(tr.row_id, tr.game_type)),
           "month": dict(zip(tr.row_id, tr.game_month))}

    # Overnight contract §21: any tool naming historical tags must refuse the
    # invalidated ones. These three are new, so this is a no-op today -- it is
    # here so a later reuse with an older tag cannot silently score it.
    invalidated.guard([BASE, CTRL, CAND])

    print(f"=== LEAFIT gate | split={a.split} | seeds={seeds} ===\n")
    print("--- parity, read before any BSS ---")
    bad = []
    for s in seeds:
        probs, ec, ek, n = parity(s, a.split)
        print(f"  seed {s}: rows {n:,} | control leaf_iters={ec['leaf_estimation_iterations']} "
              f"candidate={ek['leaf_estimation_iterations']} | cell features "
              f"{ec['n_features']} | best_iter ctrl={ec['best_iteration']} "
              f"cand={ek['best_iteration']}")
        for p in probs:
            print(f"    !! {p}")
        bad += probs
    if bad:
        print("\nPARITY FAILED -- the comparison is invalidated, not judged.")
        return 2
    print("  all parity checks pass\n")

    rows = [one_seed(s, a.split, seg) for s in seeds]

    rms = float(np.mean([r["cell_rms"] for r in rows]))
    print(f"--- redundancy kill-check (pre-registered threshold {RMS_REDUNDANT}) ---")
    print(f"  mean rms(candidate cell, control cell) = {rms:.6f}")
    if rms < RMS_REDUNDANT:
        print("  BELOW THRESHOLD -> the two arms are the same model; axis CLOSES "
              "as redundant regardless of sign.")
        return 3
    print("  above threshold: the arms are genuinely distinct\n")

    print(f"{'seed':>5} {'base':>9} {'cellC':>9} {'cellK':>9} {'coreC':>10} "
          f"{'coreK':>10} {'delta':>9}")
    for s, r in zip(seeds, rows):
        print(f"{s:>5} {r['base_bss']:>9.2f} {r['ctrl_cell_bss']:>9.2f} "
              f"{r['cand_cell_bss']:>9.2f} {r['ctrl_core_bss']:>10.3f} "
              f"{r['cand_core_bss']:>10.3f} {r['core_delta']:>+9.3f}")

    d = [r["core_delta"] for r in rows]
    print("\n--- mechanism (pre-registered: the claim is RESOLUTION) ---")
    for s, r in zip(seeds, rows):
        print(f"  seed {s}: reliability {r['ctrl_reliability']:.8f} -> "
              f"{r['cand_reliability']:.8f} ({r['d_reliability']:+.8f}) | "
              f"resolution {r['ctrl_resolution']:.8f} -> "
              f"{r['cand_resolution']:.8f} ({r['d_resolution']:+.8f})")

    print("\n--- diagnostics (never used to choose) ---")
    for s, r in zip(seeds, rows):
        segs = " ".join(f"{k[4:]}={r[k]:+.2f}" for k in r if k.startswith("seg_"))
        print(f"  seed {s}: cell delta {r['cell_delta']:+.3f} | core rms "
              f"{r['core_rms']:.6f} | pearson {r['pearson']:.7f} | mean pred diff "
              f"{r['mean_pred_diff']:+.6f} | {segs}")

    if len(d) == 1:
        v = d[0]
        call = "FAIL" if v <= 0 else ("HOLD" if v < 3 else "PASS_SEED3")
        print(f"\nseed-3 core delta {v:+.3f} -> **{call}**")
    else:
        v = judge.verdict(d)
        print(f"\n  {judge.line(v)}")
        print(f"  positive {v['positive']}/{len(d)}  median {v['median']:+.3f}")
        print(f"\nVERDICT: {v['verdict']}")
    json.dump({"seeds": seeds, "split": a.split, "rows": rows,
               "cell_rms_mean": rms},
              open(f"out/leafit_gate_{a.split}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
