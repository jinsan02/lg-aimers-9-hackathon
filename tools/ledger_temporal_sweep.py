"""Derive the contaminated set from the ledger instead of trusting a hand list.

Before `5b61fbd` (2026-08-13 03:23:39) the trainer removed only rows whose
season equalled `--test-season` and left every *later* season in the fit pool.
So a run labelled "fit ≤2021, unseen 2022" could have 2023 and 2024 in its
training data, and its "next season" number measured nothing.

`docs/INVALIDATED.tsv` is the list that gates judging (`tools/invalidated.py`),
but a hand-maintained list is only as good as the day someone remembered to
edit it — two runs sat off it for two days because they were the runs that
exposed the bug. This script recomputes the set from the ledger under one rule:

    contaminated  <=>  timestamp < the fix
                  AND  no --max-train-season
                  AND  the fit partition holds a season later than --val-season

where the fit partition is every season except the val and test seasons. Run it
after any ledger merge; it should agree with `docs/INVALIDATED.tsv` exactly.

  python tools/ledger_temporal_sweep.py           # summary + the bad tags
  python tools/ledger_temporal_sweep.py --check   # exit 2 if the TSV disagrees
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import invalidated                                               # noqa: E402

# 5b61fbd, "팀원 감사 P0 수정 + 판정 도구 강화"
FIX = "2026-08-13 03:23:39"
SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)   # train.csv, all present


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def sweep(path=None):
    """Return {tag: dict} for every contaminated run in the ledger."""
    path = path or os.path.join(ROOT, "LEDGER.tsv")
    bad, total, pre_clean = {}, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 11:
                continue
            total += 1
            ts, tag, val, test, args = c[0], c[2], _int(c[4]), _int(c[5]), c[10]
            if ts >= FIX or val is None:
                continue
            m = re.search(r"--max-train-season\s+(\d+)", args)
            cap = int(m.group(1)) if m else None
            fit = [s for s in SEASONS
                   if s != val and (test is None or s != test)
                   and (cap is None or s <= cap)]
            future = [s for s in fit if s > val]
            if not future:
                pre_clean += 1
                continue
            r = bad.setdefault(tag, {"ts": ts, "val": val, "test": test,
                                     "leaked": future, "n": 0})
            r["n"] += 1
            r["ts"] = min(r["ts"], ts)
    return bad, total, pre_clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 2 if docs/INVALIDATED.tsv disagrees with the sweep")
    a = ap.parse_args()

    bad, total, pre_clean = sweep()
    print(f"ledger rows {total} | pre-fix clean {pre_clean} | "
          f"contaminated {sum(r['n'] for r in bad.values())} rows "
          f"over {len(bad)} tags\n")
    for tag in sorted(bad, key=lambda k: bad[k]["ts"]):
        r = bad[tag]
        print(f"  {tag:<16} {r['ts']}  val={r['val']} test={r['test']}  "
              f"leaked {','.join(str(s) for s in r['leaked'])}")

    if not a.check:
        return 0
    listed = set(invalidated.load())
    found = set(bad)
    missing, extra = sorted(found - listed), sorted(listed - found)
    if missing:
        print(f"\nNOT LISTED in docs/INVALIDATED.tsv: {missing}")
    if extra:
        print(f"\nlisted but not found by the sweep (fine if hand-added for "
              f"another reason): {extra}")
    if missing:
        print("\nadd them before judging anything.")
        return 2
    print(f"\ndocs/INVALIDATED.tsv covers all {len(found)} contaminated tags.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
