"""The pre-registered single-seed gate for the ranking member.

Fixed-weight replacement only. The champion blends 0.45*base + 0.55*cell; this
swaps the **base slot** for the ranker at that same 0.45 and changes nothing
else. No weight is fitted, searched or reported as an alternative -- a ranking
member that only wins at a weight chosen after seeing the answer is the same
self-selection that closed league-conditional blending and two-strike routing.

Gate, fixed before the run:

  PASS to multiseed   untouched-season raw BSS > 0
                      AND |mean(pred) - mean(y)| < 0.03
                      AND fixed .45 replacement core delta >= +5
                      (and preferably RMS(rank, base) >= 0.003)
  FAIL                core delta <= 0, or a broken artifact
  HOLD                0 < delta < +5

+5 is a screening threshold for one seed, not the adoption bar. Adoption stays
delta >= +3 with t >= 2.4 on 6 paired seeds.

  python tools/rank_gate.py --rank RANK16_s3 --base B1SMOKE_base --cell B1SMOKE_cell
"""

from __future__ import annotations

import argparse
import os

import numpy as np

W_CELL = 0.55
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def raw_bss(y, p):
    r = float(np.mean(y))
    return float(1e5 * (1.0 - np.mean((p - y) ** 2) / (r * (1.0 - r))))


def load(tag, model="cat"):
    for m in ([model] if model else []) + ["cat", "rank"]:
        f = os.path.join(ROOT, "out", f"{m}_{tag}_test_preds.npz")
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            return (z["y"].astype(np.float64), z["pred"].astype(np.float64),
                    z["row_id"], f)
    raise SystemExit(f"no test predictions for {tag}")


def align(ref_id, rid, p):
    """Re-attach by row_id. Position agreement is an assumption, not a fact."""
    import pandas as pd
    s = pd.Series(p, index=rid)
    out = s.reindex(ref_id).to_numpy()
    if not np.isfinite(out).all():
        raise SystemExit("row_id sets differ between members")
    return out


def murphy(y, p, nb=20):
    o = np.argsort(p)
    ybar = float(np.mean(y))
    rel = res = 0.0
    for k in np.array_split(o, nb):
        if not len(k):
            continue
        w = len(k) / len(y)
        rel += w * (float(np.mean(p[k])) - float(np.mean(y[k]))) ** 2
        res += w * (float(np.mean(y[k])) - ybar) ** 2
    return rel, res, ybar * (1 - ybar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--cell", required=True)
    args = ap.parse_args()

    yb, pb, ib, fb = load(args.base, "cat")
    yc, pc, ic, fc = load(args.cell, "cat")
    yr, pr, ir, fr = load(args.rank, "rank")
    y = yb
    pc = align(ib, ic, pc)
    pr = align(ib, ir, pr)
    if not np.array_equal(yb, align(ib, ic, yc)):
        raise SystemExit("targets differ between base and cell")
    print(f"base {os.path.basename(fb)}\ncell {os.path.basename(fc)}\n"
          f"rank {os.path.basename(fr)}\nrows {len(y):,}, target mean "
          f"{y.mean():.6f}\n")

    print(f"{'member':<16}{'raw BSS':>12}{'pred mean':>12}{'bias':>10}"
          f"{'reliability':>13}{'resolution':>12}")
    for nm, p in (("base", pb), ("cell", pc), ("rank", pr)):
        rel, res, _ = murphy(y, p)
        print(f"{nm:<16}{raw_bss(y, p):>12.2f}{p.mean():>12.6f}"
              f"{p.mean() - y.mean():>+10.5f}{rel:>13.6f}{res:>12.6f}")

    ctrl = (1 - W_CELL) * pb + W_CELL * pc
    rk = (1 - W_CELL) * pr + W_CELL * pc
    b_ctrl, b_rk = raw_bss(y, ctrl), raw_bss(y, rk)
    delta = b_rk - b_ctrl
    rms = float(np.sqrt(np.mean((pr - pb) ** 2)))
    corr = float(np.corrcoef(pr, pb)[0, 1])

    print(f"\nfixed-weight replacement, base slot only, w_cell = {W_CELL}")
    print(f"  CONTROL_CORE  .45*base + .55*cell   {b_ctrl:>10.2f}")
    print(f"  RANK_CORE     .45*rank + .55*cell   {b_rk:>10.2f}")
    print(f"  delta                               {delta:>+10.2f}")
    print(f"  RMS(rank, base) {rms:.6f}   pearson {corr:+.6f}")
    for nm, p in (("CONTROL_CORE", ctrl), ("RANK_CORE", rk)):
        rel, res, unc = murphy(y, p)
        print(f"  {nm:<13} reliability {rel:.6f}  resolution {res:.6f}  "
              f"({100 * res / unc:.3f}% of uncertainty)")

    bias = abs(float(pr.mean() - y.mean()))
    ok_out = bool(np.isfinite(pr).all() and (pr >= 0).all() and (pr <= 1).all())
    standalone = raw_bss(y, pr)
    print("\ngate")
    print(f"  rank output finite and in [0,1]      {ok_out}")
    print(f"  untouched-season raw BSS > 0         {standalone > 0} "
          f"({standalone:.2f})")
    print(f"  |bias| < 0.03                        {bias < 0.03} ({bias:.5f})")
    print(f"  fixed-core delta >= +5               {delta >= 5} ({delta:+.2f})")
    print(f"  RMS >= 0.003 (preferred)             {rms >= 0.003} ({rms:.6f})")

    if not ok_out or standalone <= 0 or bias >= 0.03:
        v = "FAIL — broken artifact or unstable calibration"
    elif delta <= 0:
        v = "FAIL — fixed-weight replacement does not help"
    elif delta >= 5:
        v = "PASS — proceed to 6 paired seeds, group16 fixed, nothing else changed"
    else:
        v = f"HOLD — {delta:+.2f} is inside (0, +5); no multiseed, no rescue"
    print(f"\nVERDICT: {v}")


if __name__ == "__main__":
    main()
