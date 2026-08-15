"""Pre-registered seed-3 gate for P3-C2 structured-success weighting."""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import failmode as fm                                             # noqa: E402
import invalidated                                                # noqa: E402
import train_gbdt2 as trainer                                     # noqa: E402

W_CELL = 0.55


def raw_bss(y, p):
    r = float(np.mean(y))
    return float(1e5 * (1.0 - np.mean((p - y) ** 2) / (r * (1.0 - r))))


def read_npz(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k].copy() for k in z.files}


def scalar(tag, split):
    path = os.path.join(ROOT, "out", f"cat_{tag}_{split}_preds.npz")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    z = read_npz(path)
    return z, path


def align(ids, z, key="pred"):
    s = pd.Series(z[key], index=z["row_id"])
    out = s.reindex(ids).to_numpy()
    if not np.isfinite(out).all():
        raise SystemExit("row_id sets differ")
    return out


def require_same_order(ids, z, label):
    if not np.array_equal(np.asarray(z["row_id"]), np.asarray(ids)):
        raise SystemExit(f"{label}: row_id order differs")


def murphy(y, p, bins=20):
    order = np.argsort(p)
    mean = float(np.mean(y))
    rel = res = 0.0
    for idx in np.array_split(order, bins):
        if not len(idx):
            continue
        weight = len(idx) / len(y)
        rel += weight * (float(np.mean(p[idx])) - float(np.mean(y[idx]))) ** 2
        res += weight * (float(np.mean(y[idx])) - mean) ** 2
    return rel, res


def delta(y, control, candidate, denom):
    return float(1e5 * (np.mean((control-y)**2) -
                        np.mean((candidate-y)**2)) / denom)


def p3meta(tag):
    pack = joblib.load(os.path.join(ROOT, "model", f"cat_{tag}.pkl"))
    required = ("class_weights", "classes_order", "balanced_cells",
                "residual_cells", "analytic_deweight", "selection_counts",
                "selection_weights", "refit_counts", "refit_weights")
    missing = [k for k in required if k not in pack]
    if missing:
        raise SystemExit(f"candidate pack missing P3-C2 keys: {missing}")
    if pack["fm_success"] != [9, 10, 11] or pack["balanced_cells"] != [9, 10]:
        raise SystemExit("candidate changed the frozen success/balancing sets")
    if pack["residual_cells"] != [11] or pack["class_weights"][11] != 1.0:
        raise SystemExit("candidate changed residual cell 11")
    return pack


def actual_test_codes(ids):
    raw, _, = trainer.load(drop_f_pre=2022)[:2]
    fit = (raw["season"] <= 2023).to_numpy()
    code, names, succ = fm.build_cells(
        raw, modes=("middle", "ball", "reverse"), min_share=0.005,
        fit_mask=fit, verbose=False)
    # `code` is a Series with the frame's numeric index.  Constructing another
    # Series from it would align numeric labels against string row_ids and turn
    # every value into NaN; use the positional values explicitly.
    s = pd.Series(np.asarray(code), index=raw["row_id"].to_numpy())
    out = s.reindex(ids).to_numpy()
    expected = ["0000", "0001", "0010", "0011", "0100", "0101",
                "0110", "0111", "0xxx", "1000", "1010", "1xxx"]
    if pd.isna(out).any() or list(names) != expected:
        raise SystemExit(
            "failed to reconstruct the frozen 12-cell taxonomy: "
            f"missing={int(pd.isna(out).sum())} names={list(names)!r} "
            f"npz_dtype={np.asarray(ids).dtype} npz_head={np.asarray(ids)[:3]!r} "
            f"raw_dtype={raw['row_id'].dtype} raw_head={raw['row_id'].head(3).tolist()!r}")
    if sorted(succ) != [9, 10, 11]:
        raise SystemExit("success taxonomy drift")
    return out.astype(np.int16)


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--candidate", required=True)
    a = ap.parse_args()
    invalidated.guard([a.base, a.control, a.candidate])

    cb, _ = scalar(a.base, "test")
    cc, _ = scalar(a.control, "test")
    ca, _ = scalar(a.candidate, "test")
    ids = cb["row_id"]
    require_same_order(ids, cc, a.control)
    require_same_order(ids, ca, a.candidate)
    y = cb["y"].astype(np.float64)
    pb = cb["pred"].astype(np.float64)
    pc = align(ids, cc).astype(np.float64)
    pa = align(ids, ca).astype(np.float64)
    if not np.array_equal(y, align(ids, cc, "y")) or not np.array_equal(
            y, align(ids, ca, "y")):
        raise SystemExit("targets differ elementwise")
    for z, tag in ((cb, a.base), (cc, a.control), (ca, a.candidate)):
        host = str(z["host"].item())
        surface = str(z["surface"].item())
        seed = int(z["seed"])
        print(f"{tag}: host={host} seed={seed} surface={surface}")
        if host != "DESKTOP-053T952" or seed != 3 or surface != "val2023->test2024":
            raise SystemExit("host/seed/surface contract mismatch")

    control_core = (1-W_CELL)*pb + W_CELL*pc
    candidate_core = (1-W_CELL)*pb + W_CELL*pa
    denom = float(np.mean(y) * (1-np.mean(y)))
    full_delta = delta(y, control_core, candidate_core, denom)

    print(f"\nrows {len(y):,} target_mean {y.mean():.9f}")
    print(f"cell BSS control {raw_bss(y,pc):.3f} candidate {raw_bss(y,pa):.3f} "
          f"delta {raw_bss(y,pa)-raw_bss(y,pc):+.3f}")
    print(f"core BSS control {raw_bss(y,control_core):.3f} "
          f"candidate {raw_bss(y,candidate_core):.3f} delta {full_delta:+.3f}")
    print(f"pred mean cell {pc.mean():.7f}->{pa.mean():.7f} "
          f"core {control_core.mean():.7f}->{candidate_core.mean():.7f}")
    print(f"RMS(cell) {np.sqrt(np.mean((pa-pc)**2)):.7f} "
          f"pearson {np.corrcoef(pc,pa)[0,1]:+.7f}")
    for name, pred in (("CONTROL", control_core), ("P3C2", candidate_core)):
        rel, res = murphy(y, pred)
        print(f"{name} reliability {rel:.8f} resolution {res:.8f}")

    segments = {}
    half = len(y)//2
    segments["first"] = np.arange(len(y)) < half
    segments["second"] = ~segments["first"]
    info = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                       encoding="utf-8-sig", usecols=["row_id", "game_type"])
    gt = pd.Series(info["game_type"].to_numpy(), index=info["row_id"]).reindex(ids)
    if gt.isna().any():
        raise SystemExit("game_type row alignment failed")
    segments["R"] = gt.to_numpy() == "R"
    segments["F"] = gt.to_numpy() == "F"
    print("segments (global Brier denominator)")
    seg_delta = {}
    for name, mask in segments.items():
        seg_delta[name] = delta(y[mask], control_core[mask],
                                candidate_core[mask], denom)
        print(f"  {name:<6} n={int(mask.sum()):>7,} delta={seg_delta[name]:+.3f}")

    vb, _ = scalar(a.base, "val")
    vc, _ = scalar(a.control, "val")
    va, _ = scalar(a.candidate, "val")
    vid = vb["row_id"]
    require_same_order(vid, vc, f"{a.control} val")
    require_same_order(vid, va, f"{a.candidate} val")
    vy = vb["y"].astype(np.float64)
    if not np.array_equal(vy, vc["y"]) or not np.array_equal(vy, va["y"]):
        raise SystemExit("validation targets differ elementwise")
    vpc, vpa = align(vid, vc), align(vid, va)
    vpb = vb["pred"].astype(np.float64)
    vctrl = (1-W_CELL)*vpb + W_CELL*vpc
    vcand = (1-W_CELL)*vpb + W_CELL*vpa
    print(f"source val2023 core delta "
          f"{raw_bss(vy,vcand)-raw_bss(vy,vctrl):+.3f}")

    pack = p3meta(a.candidate)
    print("P3-C2 weights")
    for where in ("selection", "refit"):
        counts = np.asarray(pack[f"{where}_counts"])
        weights = np.asarray(pack[f"{where}_weights"])
        err = abs(float(np.dot(counts[[9,10,11]], weights[[9,10,11]]) -
                        counts[[9,10,11]].sum()))
        print(f"  {where}: n9/10/11={counts[9]}/{counts[10]}/{counts[11]} "
              f"w9/10/11={weights[9]:.8f}/{weights[10]:.8f}/{weights[11]:.1f} "
              f"mass_error={err:.3e}")

    cfull = read_npz(os.path.join(ROOT, "out",
                                  f"cat_{a.control}_test_preds.npz"))
    afull = read_npz(os.path.join(ROOT, "out",
                                  f"cat_{a.candidate}_test_preds.npz"))
    require_same_order(ids, cfull, f"{a.control} full proba")
    require_same_order(ids, afull, f"{a.candidate} full proba")
    cp = cfull["cell_proba"].astype(np.float64)
    ap = afull["cell_proba"].astype(np.float64)
    aq = afull["cell_proba_weighted"].astype(np.float64)
    codes = actual_test_codes(ids)
    print("success-cell calibration (actual share / predicted mass; candidate q->p)")
    for c in (9,10,11):
        actual = float(np.mean(codes == c))
        print(f"  cell {c}: actual {actual:.7f} control {cp[:,c].mean():.7f} "
              f"candidate {aq[:,c].mean():.7f}->{ap[:,c].mean():.7f}")

    halves_ok = seg_delta["first"] >= 0 and seg_delta["second"] >= 0
    if full_delta <= 0:
        verdict = "FAIL — untouched fixed-core delta <= 0"
    elif full_delta < 5 or not halves_ok:
        verdict = "HOLD — no n=6 extension"
    else:
        verdict = "PASS — extend to n=6"
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
