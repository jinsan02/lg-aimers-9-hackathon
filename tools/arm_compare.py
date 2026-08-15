"""Paired arm comparison that shows the stopping point next to the score.

Why the iteration column belongs in the table. `NULLC_cell` re-ran `B1S_cell`'s
exact command on 2026-08-14 and reproduced it: per-seed core
[+0.50, 0.00, +0.01, 0.00, -0.00, -0.00]. The variation that did appear was not
diffuse GPU nondeterminism, it was the early-stopping pick moving --
corr(|delta best_iter|, |delta val BSS|) = 0.9906, three seeds at delta_iter 0
landing on 0.000 exactly.

The base arm behaves differently. It stops at 950-1800 where the eval curve is
flat, and between two identical runs its pick moved by up to +459 iterations,
worth +2.62 BSS. That is the number previously recorded as this machine's
"noise floor". It is not a floor -- it is stopping instability, and it is
diagnosable: where best_iter is unchanged the re-run is exact, so the delta is
mechanism; where it jumped, part of the delta is lottery.

So every row here carries both. A candidate that wins while its stopping point
moved several hundred iterations has not been separated from the lottery yet.

The adoption rule comes from `tools/judge.py`, shared with `surf_report.py`.
**They share the rule, not the surface**: this tool scores `*_val_preds.npz`,
`surf_report` scores the unseen `*_test_preds.npz`. The same arm shows
different numbers in the two tables and that is correct; do not read one
against the other.

  python tools/arm_compare.py --control CTRL_base --cand CTR3_base
  python tools/arm_compare.py --control NULLC_cell --cand CTR3_cell --base CTRL_base
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The adoption rule lives in tools/judge.py so this tool and surf_report.py
# cannot drift apart again -- they disagreed on the interval, the bar and the
# centring until 2026-08-15.
from judge import SHIFT, SLOPE, T975, bss, debias, verdict     # noqa: E402,F401

W_CELL = 0.55


def seed_of(p):
    return p.rsplit("_s", 1)[1].split("_", 1)[0]


def preds(tag, split="val"):
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "out",
                                           f"*_{tag}_s*_{split}_preds.npz"))):
        z = np.load(p, allow_pickle=True)
        out[int(seed_of(p))] = (z["pred"].astype(np.float64),
                                z["y"].astype(np.float64))
    return out


def ledger(tag, strict=True):
    """Newest row per seed, and a refusal when a re-run changed the command.

    `drop_duplicates("tag")` kept the **first** match, which is the *oldest*
    row. A tag re-run under different flags -- the ordinary way a control gets
    refreshed -- therefore reported the superseded run's iteration count, host
    and argv while the npz on disk came from the new one. Sorting by timestamp
    and keeping the last fixes the read; `strict` makes a silent overwrite an
    error rather than a preference, because if two runs share a tag and differ
    in argv, "the newest" is a guess about which produced the predictions.
    """
    d = pd.read_csv(os.path.join(ROOT, "LEDGER.tsv"), sep="\t", header=None,
                    names=["ts", "host", "tag", "m", "v", "t", "seed", "it",
                           "val", "test", "argv"])
    d = d[d.tag.str.match(rf"{tag}_s\d+$", na=False)].sort_values("ts")
    if strict:
        for tg, grp in d.groupby("tag"):
            argvs = {a for a in grp.argv.astype(str)}
            if len(argvs) > 1:
                raise SystemExit(
                    f"{tg} appears {len(grp)} times in the ledger with "
                    f"{len(argvs)} different commands:\n  "
                    + "\n  ".join(f"{r.ts}  {r.argv}" for r in grp.itertuples())
                    + f"\nThe npz on disk came from one of them and the tag "
                      f"does not say which. Re-run under a fresh tag.")
    d = d.drop_duplicates("tag", keep="last")
    return {int(r.seed): (int(r.it), float(r.val), r.host, r.argv)
            for r in d.itertuples()}


def report(name, d):
    d = np.asarray(d, float)
    v = verdict(d)
    print(f"\n  {name}")
    print(f"    mean {v['mean']:+.3f}  SE {v['se']:.3f}  t {v['t']:+.2f}  "
          f"95% CI [{v['lo']:+.2f}, {v['hi']:+.2f}]")
    print(f"    median {v['median']:+.3f}  "
          f"trimmed {np.sort(d)[1:-1].mean():+.3f}  "
          f"positive {v['positive']}/{v['n']}")
    print(f"    delta>=+3 {'Y' if v['gate_delta'] else 'N'}  "
          f"t>=2.4 {'Y' if v['gate_t'] else 'N'}  "
          f"n>=6 {'Y' if v['gate_n'] else 'N'}  "
          f"95% upper>=+3 {'Y' if v['hi'] >= 3 else 'N (reject)'}  -> "
          f"{v['verdict']}")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True,
                    help="a same-session re-run of the champion's command")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base", default=None,
                    help="when comparing cell arms, the base family to hold "
                         "fixed so core can be formed")
    a = ap.parse_args()

    LC, LK = ledger(a.control), ledger(a.cand)
    PC, PK = preds(a.control), preds(a.cand)
    seeds = sorted(set(LC) & set(LK) & set(PC) & set(PK))
    if not seeds:
        raise SystemExit(f"no paired seeds: ledger {sorted(LC)}/{sorted(LK)} "
                         f"preds {sorted(PC)}/{sorted(PK)}")

    hosts = {LC[s][2] for s in seeds} | {LK[s][2] for s in seeds}
    if len(hosts) > 1:
        raise SystemExit(f"hosts differ: {hosts} -- an 11-point machine effect "
                         f"makes this comparison meaningless")
    ca, ka = LC[seeds[0]][3].split(), LK[seeds[0]][3].split()
    only_c = [x for x in ca if x not in ka]
    only_k = [x for x in ka if x not in ca]
    print(f"{a.cand} vs {a.control} | host {hosts.pop()} | {len(seeds)} paired seeds")
    print(f"  argv only in control   : {only_c}")
    print(f"  argv only in candidate : {only_k}")

    y0 = PC[seeds[0]][1]
    for s in seeds:
        for p, t in ((PC[s], a.control), (PK[s], a.cand)):
            if not np.array_equal(p[1], y0):
                raise SystemExit(f"{t} seed {s} scores different rows")

    print(f"\n{'seed':>5} {'ctl_iter':>9} {'cand_iter':>10} {'d_iter':>7}   "
          f"{'ctl_BSS':>9} {'cand_BSS':>9} {'d_BSS':>8}   "
          f"{'ctl_deb':>9} {'cand_deb':>9} {'d_deb':>8}")
    d_raw, d_deb, d_iter = [], [], []
    for s in seeds:
        it_c, v_c, _, _ = LC[s]
        it_k, v_k, _, _ = LK[s]
        db_c, db_k = bss(debias(PC[s][0]), y0), bss(debias(PK[s][0]), y0)
        d_iter.append(it_k - it_c)
        d_raw.append(v_k - v_c)
        d_deb.append(db_k - db_c)
        print(f"{s:>5} {it_c:>9} {it_k:>10} {it_k - it_c:>+7}   "
              f"{v_c:>9.2f} {v_k:>9.2f} {v_k - v_c:>+8.2f}   "
              f"{db_c:>9.2f} {db_k:>9.2f} {db_k - db_c:>+8.2f}")

    ai = np.abs(d_iter)
    print(f"\n  |d_iter| mean {ai.mean():.1f}  max {ai.max()}  "
          f"seeds unchanged {int((ai == 0).sum())}/{len(seeds)}")
    if len(seeds) > 2 and ai.std() > 0 and np.std(np.abs(d_deb)) > 0:
        r = np.corrcoef(ai, np.abs(d_deb))[0, 1]
        print(f"  corr(|d_iter|, |d_deb|) = {r:+.4f}"
              + ("   <- the stopping point is driving the delta"
                 if r > 0.8 else ""))
    if (ai == 0).sum() and abs(np.asarray(d_deb)[ai == 0]).max() < 1e-9:
        print("  seeds whose stopping point did not move reproduce exactly, so "
              "their delta is mechanism")

    report("arm alone, debiased", d_deb)

    if a.base:
        B = preds(a.base)
        common = [s for s in seeds if s in B]
        if len(common) < len(seeds):
            print(f"\n  note: base {a.base} covers only {common}")
        core = []
        for s in common:
            cc = bss(debias((1 - W_CELL) * B[s][0] + W_CELL * PC[s][0]), y0)
            ck = bss(debias((1 - W_CELL) * B[s][0] + W_CELL * PK[s][0]), y0)
            core.append(ck - cc)
        report(f"core (0.45 x {a.base} + 0.55 x cell)", core)
    return 0


if __name__ == "__main__":
    sys.exit(main())
