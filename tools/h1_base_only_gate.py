"""H1ADD in the base slot only, against a same-session fresh control.

The champion blends `0.45*base + 0.55*cell`. The `H1_ADDITIVE` axis was DROPped
on the **full** core, where both arms carried `h1_hand_delta` and the cell arm
disliked it (base +2.161, cell -1.235, core -0.304). An independent Mac run then
reproduced the base half of that (+4.995 base-only, 8 seeds). Both machines put
the base arm on the same side of zero, so the open question is narrow and fixed
in advance:

    keep the cell EXACTLY as the control's, change the base slot only.

Everything else is frozen: the 0.45/0.55 weight, the taxonomy, the seeds, the
post-processing. Nothing here is searched — no H1 coefficient, no shrink k, no
weight, no choice of which arm to apply it to. Those are all decided before the
first number is read, and the point of this tool is that it cannot quietly
become a search.

**The statistical risk is recorded, not hidden.** This candidate was defined
*after* seeing the family split, so its prior is worse than a pre-registered
one. Organiser rules permit choosing models, features and ensembles, so this is
not a rules question -- it is a selection-bias question, and the honest handling
is to hold it to the same bar and say plainly where it came from.

Scoring is on **raw, deployable predictions**. Centred numbers are printed as a
diagnostic and are never the verdict: the centring constant is the evaluation
season's own mean and does not exist at submission time.

  python tools/h1_base_only_gate.py --control CTRL_base --cand H1ADD_base --cell NULLC_cell
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from invalidated import guard as _guard_invalidated              # noqa: E402
from judge import bss, centred_bss, debias, line, verdict        # noqa: E402

W_CELL = 0.55


def members(tag):
    """{seed: (pred, y, row_id, host, surface)} from the val npz."""
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "out",
                                           f"cat_{tag}_s*_val_preds.npz"))):
        z = np.load(p, allow_pickle=True)
        seed = int(os.path.basename(p).split("_s")[-1].split("_")[0])
        out[seed] = (z["pred"].astype(np.float64), z["y"].astype(np.float64),
                     z["row_id"], str(z["host"]), str(z["surface"]))
    return out


def murphy(y, p, nb=20):
    """(reliability, resolution) on equal-count bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    r = y.mean()
    order = np.argsort(p, kind="mergesort")
    rel = res = 0.0
    for idx in np.array_split(order, nb):
        if not len(idx):
            continue
        w = len(idx) / len(y)
        pb, ob = p[idx].mean(), y[idx].mean()
        rel += w * (pb - ob) ** 2
        res += w * (ob - r) ** 2
    return rel, res


def segments(row_id):
    """Half-season and league masks, from the row's own columns only."""
    hdr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                      usecols=["row_id", "game_type", "game_month"],
                      encoding="utf-8-sig")
    hdr = hdr.set_index("row_id").reindex(row_id)
    gt = hdr["game_type"].to_numpy()
    mo = hdr["game_month"].to_numpy(float)
    med = np.nanmedian(mo)
    return {"early (month <= %g)" % med: mo <= med,
            "late  (month >  %g)" % med: mo > med,
            "R league": gt == "R",
            "F league": gt == "F"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--cell", required=True,
                    help="the FIXED cell family, identical in both arms")
    a = ap.parse_args()
    _guard_invalidated([a.control, a.cand, a.cell])

    C, K, L = members(a.control), members(a.cand), members(a.cell)
    seeds = sorted(set(C) & set(K) & set(L))
    if not seeds:
        raise SystemExit(f"no shared seeds: {sorted(C)} / {sorted(K)} / {sorted(L)}")

    # ---- provenance, before any number is read ---------------------------
    print("provenance")
    hosts = {C[s][3] for s in seeds} | {K[s][3] for s in seeds} | {L[s][3] for s in seeds}
    surfs = {C[s][4] for s in seeds} | {K[s][4] for s in seeds} | {L[s][4] for s in seeds}
    if len(hosts) > 1:
        raise SystemExit(f"hosts differ: {hosts} -- an 11-point machine effect "
                         f"makes this comparison meaningless")
    if len(surfs) > 1:
        raise SystemExit(f"surfaces differ: {surfs}")
    ref_id, y0 = C[seeds[0]][2], C[seeds[0]][1]
    for s in seeds:
        for nm, M in ((a.control, C), (a.cand, K), (a.cell, L)):
            if not np.array_equal(M[s][2], ref_id):
                raise SystemExit(f"{nm} seed {s}: row_id differs from the control")
            if not np.array_equal(M[s][1], y0):
                raise SystemExit(f"{nm} seed {s}: target differs from the control")
    print(f"  host {hosts.pop()} | surface {surfs.pop()} | {len(seeds)} paired "
          f"seeds {seeds}")
    print(f"  row_id and target identical elementwise across all "
          f"{3 * len(seeds)} members, {len(y0):,} rows, target mean "
          f"{y0.mean():.9f}")
    print(f"  cell family {a.cell} is FIXED: the same predictions enter both "
          f"arms, so it cancels in the paired difference")

    # ---- per seed ---------------------------------------------------------
    print(f"\n{'seed':>5} {'ctl_base':>10} {'cand_base':>10} {'d_base':>8}   "
          f"{'ctl_core':>10} {'cand_core':>10} {'d_core':>8}")
    d_base, d_core = [], []
    for s in seeds:
        cb, kb, cl = C[s][0], K[s][0], L[s][0]
        bc, bk = bss(cb, y0), bss(kb, y0)
        cc = bss((1 - W_CELL) * cb + W_CELL * cl, y0)
        ck = bss((1 - W_CELL) * kb + W_CELL * cl, y0)
        d_base.append(bk - bc)
        d_core.append(ck - cc)
        print(f"{s:>5} {bc:>10.2f} {bk:>10.2f} {bk - bc:>+8.2f}   "
              f"{cc:>10.2f} {ck:>10.2f} {ck - cc:>+8.2f}")

    print("\nbase arm alone (raw, deployable)")
    vb = verdict(d_base)
    print("  " + line(vb))
    print(f"\nFIXED CORE  0.45*{a.cand} + 0.55*{a.cell}  vs  "
          f"0.45*{a.control} + 0.55*{a.cell}   <- THE VERDICT")
    vc = verdict(d_core)
    print("  " + line(vc))
    print(f"  median {vc['median']:+.3f}  positive {vc['positive']}/{vc['n']}")

    # ---- ensembles --------------------------------------------------------
    eb_c = np.mean([C[s][0] for s in seeds], 0)
    eb_k = np.mean([K[s][0] for s in seeds], 0)
    el = np.mean([L[s][0] for s in seeds], 0)
    core_c = (1 - W_CELL) * eb_c + W_CELL * el
    core_k = (1 - W_CELL) * eb_k + W_CELL * el
    print(f"\nseed-ensemble (what would actually ship)")
    print(f"  base    {bss(eb_c, y0):>9.2f} -> {bss(eb_k, y0):>9.2f}   "
          f"{bss(eb_k, y0) - bss(eb_c, y0):>+8.2f}")
    print(f"  core    {bss(core_c, y0):>9.2f} -> {bss(core_k, y0):>9.2f}   "
          f"{bss(core_k, y0) - bss(core_c, y0):>+8.2f}")
    print(f"  core debiased {bss(debias(core_c), y0):>9.2f} -> "
          f"{bss(debias(core_k), y0):>9.2f}   "
          f"{bss(debias(core_k), y0) - bss(debias(core_c), y0):>+8.2f}"
          f"   (+139.03 -> LB scale)")
    print(f"  core CENTRED  {centred_bss(core_c, y0):>9.2f} -> "
          f"{centred_bss(core_k, y0):>9.2f}   DIAGNOSTIC ONLY, not the verdict")

    rc, sc = murphy(y0, core_c)
    rk, sk = murphy(y0, core_k)
    print(f"  reliability {rc:.6f} -> {rk:.6f}   "
          f"resolution {sc:.6f} -> {sk:.6f}")
    print(f"  bias  {core_c.mean() - y0.mean():+.6f} -> "
          f"{core_k.mean() - y0.mean():+.6f}")
    rms = float(np.sqrt(np.mean((eb_k - eb_c) ** 2)))
    print(f"  RMS(cand base, control base) {rms:.6f}   "
          f"pearson {np.corrcoef(eb_k, eb_c)[0, 1]:+.6f}")

    # ---- segments ---------------------------------------------------------
    print("\nsegments (fixed core, seed-ensemble, raw)")
    try:
        segs = segments(ref_id)
    except Exception as e:                     # data/train.csv absent on a worker
        print(f"  (skipped: {e})")
        segs = {}
    for name, m in segs.items():
        if m.sum() < 100:
            continue
        print(f"  {name:<22} n={int(m.sum()):>7,}  "
              f"{bss(core_c[m], y0[m]):>9.2f} -> {bss(core_k[m], y0[m]):>9.2f}"
              f"   {bss(core_k[m], y0[m]) - bss(core_c[m], y0[m]):>+8.2f}")

    # ---- gate -------------------------------------------------------------
    print("\ngate, fixed before the run")
    print(f"  core delta <= 0                      -> DROP        "
          f"({vc['mean']:+.3f})")
    print(f"  95% upper < +3                       -> DROP        "
          f"(upper {vc['hi']:+.3f})")
    print(f"  0 < delta < +3, or t < 2.4           -> PARK, no GPU"
          f"  (t {vc['t']:+.2f})")
    print(f"  delta >= +3 AND n >= 6 AND t >= 2.4  -> refit candidate")
    if vc["mean"] <= 0:
        v = "DROP -- the fixed core does not improve"
    elif vc["hi"] < 3:
        v = "DROP -- the interval cannot reach +3"
    elif vc["mean"] >= 3 and vc["t"] >= 2.4 and vc["n"] >= 6:
        v = "SUBMISSION-FORM REFIT CANDIDATE -- report before building anything"
    else:
        v = "PARK -- no additional GPU, no coefficient or weight search"
    print(f"\nVERDICT: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
