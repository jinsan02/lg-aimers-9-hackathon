"""The contamination list has to be a gate in every tool that reads a tag.

`docs/INVALIDATED.tsv` and `tools/invalidated.py` were written on 2026-08-13,
the day the fit-partition bug was found. Three judging tools called `guard()`.
Then on 2026-08-15 three *new* tools were written -- `season_transfer_map.py`,
`season_transfer_core.py`, `matchup_decomp.py` -- none of which called it, and
they produced a season-transfer map over four listed tags. The list did its job;
nothing consulted it. A list that new code can bypass is a note, not a gate.

So this test does not check that the three original tools still call `guard()`.
It scans **every** file under `tools/` for a reference to a contaminated model
and requires that file to import the guard. A tool added tomorrow that names
`cat_MV21_base.pkl` fails here before it can produce evidence.

Run: python tests/test_invalidated_guard.py
"""

from __future__ import annotations

import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import invalidated                                               # noqa: E402

FAIL = []

# Every tag whose run predates 5b61fbd (2026-08-13 03:23:39) and carried later
# seasons in its fit partition. Ten were listed on 08-13; the two P0SMOKE runs
# were added on 08-15 -- they ran at 03:20 and 03:21, three minutes before the
# fix, and were missed because they were the runs that exposed the bug.
CONTAMINATED = [
    "MV21_base", "MV21_cell", "MVB22_native", "MVCELL22_s42",
    "CTR2_H22", "CTR3_H22", "P0SMOKE_base", "P0SMOKE_cell",
    "PMTC21", "PMTC22", "XCR0", "XCR21",
]

# Tags on the 2023->2024 boundary, plus the champion. No season exists after
# 2024, so the bug had nothing to leak; these must stay usable.
CLEAN = [
    "MVN3_s3", "MVN3_s4", "MVN3_s5", "MVCELL_s42",
    "B1SMOKE_base", "B1SMOKE_cell", "B1S_base", "B1S_cell", "RK16_s3",
]


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def main():
    print("the list itself")
    listed = invalidated.load()
    missing = [t for t in CONTAMINATED if t not in listed]
    check("1 all 12 contaminated tags are listed", not missing,
          f"{len(listed)} listed" + (f", missing {missing}" if missing else ""))
    check("2 every entry carries a reason", all(v[1].strip()
                                                for v in listed.values()))

    print("\nguard() verdicts")
    refused = [t for t in CONTAMINATED if invalidated.guard([t], hard=False)]
    check("3 refuses all 12", len(refused) == 12,
          f"{len(refused)}/12" + (f", let through {sorted(set(CONTAMINATED) - set(refused))}"
                                  if len(refused) != 12 else ""))
    let_through = [t for t in CLEAN if invalidated.guard([t], hard=False)]
    check("4 does NOT refuse the clean 2023->2024 tags or the champion",
          not let_through, f"wrongly refused {let_through}" if let_through else
          f"{len(CLEAN)} tags pass")
    try:
        invalidated.guard(["MV21_base"])
        check("5 hard=True raises", False, "it returned")
    except SystemExit:
        check("5 hard=True raises", True)

    print("\nno tool may name a contaminated model without the gate")
    # Substrings that only appear when a file actually points at one of these
    # runs -- the tag, or the pickle it wrote.
    needles = sorted({t for t in CONTAMINATED} | {f"cat_{t}" for t in CONTAMINATED})
    offenders = []
    scanned = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "tools", "*.py"))):
        base = os.path.basename(path)
        if base == "invalidated.py":
            continue
        src = open(path, encoding="utf-8").read()
        scanned += 1
        hits = sorted({n for n in needles if n in src})
        if hits and "invalidated" not in src:
            offenders.append(f"{base} names {hits[:3]}")
    check("6 every tool naming one of them imports the guard", not offenders,
          f"{scanned} tools scanned" if not offenders else "; ".join(offenders))

    # And the three that were bypassed on 08-15 must call it, not merely import.
    for mod in ("season_transfer_map.py", "season_transfer_core.py",
                "matchup_decomp.py"):
        src = open(os.path.join(ROOT, "tools", mod), encoding="utf-8").read()
        check(f"7 {mod} calls the guard",
              "from invalidated import" in src
              and "_guard_invalidated(" in src, "import + call")

    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
