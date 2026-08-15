"""Unseen-surface arm table: paired per-seed deltas against a baseline, with t.

Reads `test_preds` npz only, so console encoding cannot corrupt it (reading the
Korean logs broke differently on each machine and cost two false starts).

**Two things changed on 2026-08-15.** Both were judgement defects, not display
choices:

1. **The score is no longer oracle-centred by default.** This tool used to
   score `p - (p.mean() - r)`, shifting every candidate onto the mean of the
   season being scored. That constant does not exist at submission time, so a
   centred number hides a candidate's calibration loss and answers "did
   resolution improve?" rather than "would this score better?". The default is
   now the champion's fixed debiasing (constants that do exist at submission
   time). `--centred` restores the old behaviour and labels the table
   DIAGNOSTIC.

2. **The adoption rule now comes from `tools/judge.py`**, the same module
   `arm_compare.py` uses. This tool adopted on `t >= 2.4` alone and used
   `1.96 * se` at n=6, where the correct critical value is
   `t(.975, df=5) = 2.571`; the documented bar is
   `delta >= +3 AND t >= 2.4 AND n >= 6`. The two tools could return different
   verdicts on identical evidence, and did.

**They share the rule, not the surface.** This tool scores `*_test_preds.npz`
— the unseen season. `arm_compare.py` scores `*_val_preds.npz` and pairs the
early-stopping iteration alongside. The same arm will show different numbers in
the two tables and that is correct; do not read one against the other.

  python tools/surf_report.py BASELINE CAND1 CAND2 ...
  python tools/surf_report.py --centred BASELINE CAND1     # resolution only
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from invalidated import guard as _guard_invalidated           # noqa: E402
from judge import bss, centred_bss, debias, verdict           # noqa: E402


def machine_of(tag):
    """Which machine produced a tag, from the ledger.

    2026-08-08: `--std-k 40` measured +15.05 on the A100 and +4.27 on the 4070 —
    same setting, same seeds, 11 points of **machine**. A baseline was nearly
    compared across hosts and shipped. Checked before every comparison now.
    """
    try:
        with open("./LEDGER.tsv", encoding="utf-8") as f:
            hosts = {c[1] for c in (l.split("\t") for l in f)
                     if len(c) > 2 and c[2].startswith(tag + "_s")}
        return "/".join(sorted(hosts)) if hosts else "?"
    except OSError:
        return "?"


def load(tag):
    fs = sorted(glob.glob(f"./out/*_{tag}_s*_test_preds.npz"))
    if not fs:
        return None, None, []
    seeds, preds, y = [], [], None
    for f in fs:
        z = np.load(f, allow_pickle=True)
        seeds.append(f.split("_s")[-1].split("_")[0])
        preds.append(z["pred"].astype(np.float64))
        y = z["y"].astype(np.float64)
    return dict(zip(seeds, preds)), y, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--centred", action="store_true",
                    help="score each arm shifted onto the evaluation season's "
                         "own mean. DIAGNOSTIC ONLY -- that constant does not "
                         "exist at submission time.")
    a = ap.parse_args()
    tags = a.tags
    if len(tags) < 2:
        print(__doc__)
        return 1
    _guard_invalidated(tags)

    B, y, bs = load(tags[0])
    if B is None:
        print(f"baseline not found: {tags[0]}")
        return 1

    sc = (lambda p: centred_bss(p, y)) if a.centred else \
         (lambda p: bss(debias(p), y))
    mode = ("CENTRED -- resolution diagnostic, NOT an adoption number"
            if a.centred else "debiased (submission-time constants)")

    b_ens = sc(np.mean(list(B.values()), 0))
    b_host = machine_of(tags[0])
    print(f"baseline {tags[0]}  {len(bs)} seeds, ensemble {b_ens:.2f}  "
          f"[{b_host}]  (unseen {len(y):,} rows)")
    print(f"score: {mode}\n")
    print(f"{'arm':<14}{'seeds':>6}{'ens':>10}{'d_ens':>9}{'paired':>10}"
          f"{'SE':>7}{'t':>7}{'95% lo':>9}  {'call':<8}machine")
    rc = 0
    for t in tags[1:]:
        A, _, _ = load(t)
        if A is None:
            print(f"{t:<14}  (no predictions)")
            continue
        common = sorted(set(A) & set(B))
        if not common:
            print(f"{t:<14}  (no shared seeds)")
            continue
        d = np.array([sc(A[s]) - sc(B[s]) for s in common])
        v = verdict(d)
        ens = sc(np.mean([A[s] for s in common], 0))
        host = machine_of(t)
        call = v["verdict"]
        # A cross-machine comparison is void however large t is: the measured
        # machine effect (11 points) exceeds most candidate effects.
        if host != "?" and b_host != "?" and host != b_host:
            call = "VOID(host)"
            rc = 2
        elif a.centred:
            call = call + "*"
        print(f"{t:<14}{v['n']:>6}{ens:>10.2f}{ens - b_ens:>+9.2f}"
              f"{v['mean']:>+10.2f}{v['se']:>7.2f}{v['t']:>+7.2f}"
              f"{v['lo']:>+9.2f}  {call:<8}{host}")

    print("\nKEEP: mean>=+3 AND t>=2.4 AND n>=6 | DROP: 95% upper < +3 | "
          "else PARK (more seeds)")
    print("Interval is Student-t (2.571 at n=6), from tools/judge.py -- the "
          "same rule arm_compare.py applies.")
    if a.centred:
        print("* CENTRED: these are resolution diagnostics. No adoption may "
              "rest on them.")
    print("A different machine from the baseline voids the row -- --std-k 40 "
          "measured A100 +15.05 vs 4070 +4.27.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
