"""Experiments that ran but were never judged.

The ledger is the record of what the GPUs actually did; SETTLED is the record of
what we concluded. Nothing keeps them in step, so an axis can be measured, cost
an hour of GPU, and then never be written down -- and the next agent proposes it
again. This tool lists the gap.

Matching is on flag names, not prose. For every ledger family the distinguishing
tokens are whatever its argv holds beyond the canonical core for its surface;
a family is "covered" when some FLAG line in SETTLED names all of them. That is
deliberately generous in one direction -- a FLAG that merely mentions a flag
counts -- because the failure mode worth catching is a config nobody wrote about
at all, not one described imprecisely.

Surfaces are kept apart and hosts are kept apart. An 11-point machine effect and
a banned surface mix (SETTLED -> drop-f-pre-omitted) both produce fake deltas,
so a family is only ever compared against a baseline that shares both.

  python tools/orphan_audit.py
  python tools/orphan_audit.py --min-delta 3
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Flags that describe bookkeeping rather than a hypothesis.
IGNORE = {"--model", "--tag", "--seeds", "--seed", "--device", "--dump-cell-proba",
          "--no-refit", "--verbose"}
# The judging/submission core. Anything here is the baseline, not a candidate.
CORE = {
    "--feat-v2", "--feat-k", "200", "--te", "p,pc,ph,b,pi", "--te-k", "50",
    "--te-dev", "--feat-std", "--std-k", "80", "--std-to-prior",
    "--std-season-prior", "--feat-domain", "--feat-skill-pc", "--lr", "0.01",
    "--iters", "3000", "--es", "500", "--l2", "10", "--border-count", "254",
    "--loss", "Logloss", "--eval-metric", "--refit-mult", "1.5", "cat",
    "--depth", "8", "5", "--failmode-cells", "--fm-modes",
    "middle,ball,reverse", "--fm-min-share", "0.005",
}


def surface(argv):
    v = re.search(r"--val-season\s+(\d+)", argv)
    t = re.search(r"--test-season\s+(\d+)", argv)
    d = re.search(r"--drop-f-pre\s+(\d+)", argv)
    return (f"val{v.group(1) if v else '?'}",
            f"test{t.group(1) if t else 'none'}",
            f"dropf{d.group(1) if d else 'none'}")


def distinguishing(argv):
    toks = [t for t in str(argv).split() if t not in CORE]
    out, skip = [], False
    for i, t in enumerate(toks):
        if skip:
            skip = False
            continue
        if t in IGNORE:
            skip = True
            continue
        if t.startswith("--"):
            if t in ("--val-season", "--test-season", "--drop-f-pre",
                     "--max-train-season"):
                skip = True
                continue
            out.append(t)
    return tuple(sorted(set(out)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-delta", type=float, default=3.0)
    a = ap.parse_args()

    d = pd.read_csv(os.path.join(ROOT, "LEDGER.tsv"), sep="\t", header=None,
                    names=["ts", "host", "tag", "model", "val_s", "test_s",
                           "seed", "iters", "val", "test", "argv"])
    d = d.dropna(subset=["argv"])
    d["fam"] = d.tag.str.replace(r"_s\d+$", "", regex=True)
    d["surf"] = [surface(x) for x in d.argv]
    d["flags"] = [distinguishing(x) for x in d.argv]

    settled = open(os.path.join(ROOT, "docs", "SETTLED.md"),
                   encoding="utf-8").read()
    named = set(re.findall(r"--[a-z0-9][a-z0-9-]*", settled))

    print(f"ledger {len(d):,} rows | {d.fam.nunique()} families | "
          f"SETTLED names {len(named)} distinct flags\n")

    groups = defaultdict(list)
    for r in d.itertuples():
        groups[(r.host, r.surf, r.flags)].append(r)

    # A family with no distinguishing flags is a baseline for its cell.
    base = {}
    for (host, surf, flags), rows in groups.items():
        if not flags or flags == ("--p1",):
            for r in rows:
                arm = "cell" if "--failmode-cells" in str(r.argv) else "base"
                key = (host, surf, arm, flags)
                base.setdefault(key, {})[int(r.seed)] = r.val

    rows_out = []
    for (host, surf, flags), rows in sorted(groups.items()):
        if not flags or flags == ("--p1",):
            continue
        arm = "cell" if "--failmode-cells" in str(rows[0].argv) else "base"
        cand = {int(r.seed): r.val for r in rows}
        uncovered = [f for f in flags if f not in named and f != "--p1"]
        # Prefer a baseline that shares the --p1 setting.
        want = ("--p1",) if "--p1" in flags else ()
        ref = base.get((host, surf, arm, want)) or base.get((host, surf, arm, ()))
        delta = seeds = np.nan
        if ref:
            common = sorted(set(ref) & set(cand))
            if common:
                dl = np.array([cand[s] - ref[s] for s in common])
                delta, seeds = dl.mean(), len(common)
        rows_out.append({
            "host": host, "surface": "/".join(surf), "arm": arm,
            "flags": " ".join(flags), "fams": ",".join(sorted({r.fam for r in rows})),
            "n": len(rows), "paired": seeds, "delta": delta,
            "covered": not uncovered, "uncovered": " ".join(uncovered),
        })

    t = pd.DataFrame(rows_out)
    orph = t[~t.covered].copy()
    print("=" * 100)
    print(f"ORPHANS -- configs whose distinguishing flags appear nowhere in SETTLED")
    print("=" * 100)
    if orph.empty:
        print("  none: every measured config has at least a flag-level mention")
    else:
        orph = orph.sort_values("delta", ascending=False, na_position="last")
        for r in orph.itertuples():
            dl = f"{r.delta:+8.2f}" if r.delta == r.delta else "       ?"
            pr = f"{int(r.paired)}" if r.paired == r.paired else "-"
            print(f"  {dl} ({pr} paired)  {r.arm:4} {r.surface:28} {r.uncovered}")
            print(f"           tags {r.fams}")

    strong = t[(t.delta >= a.min_delta)].sort_values("delta", ascending=False)
    print("\n" + "=" * 100)
    print(f"ANY config at >= +{a.min_delta:g} against its own baseline "
          f"(covered or not) -- candidates to re-check")
    print("=" * 100)
    if strong.empty:
        print("  none")
    for r in strong.itertuples():
        mark = "COVERED" if r.covered else "ORPHAN "
        print(f"  {r.delta:+8.2f} ({int(r.paired)} paired)  {mark}  {r.arm:4} "
              f"{r.surface:28} {r.flags}")
        print(f"           tags {r.fams}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
