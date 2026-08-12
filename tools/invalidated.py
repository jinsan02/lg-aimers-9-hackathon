"""Refuse tags whose runs are known to be contaminated.

`docs/INVALIDATED.tsv` is the single list. Judging tools call `guard()` so a
poisoned tag cannot silently become evidence again -- writing the fact in a
document was not enough the last three times.

Currently listed: ten rolling runs from before 2026-08-13, when `--test-season S`
flagged only rows equal to S and left every later season in the training pool
(253,507 rows of 2024 in the val2022 runs, 499,032 rows in the val2021 runs).
The 2023->2024 surface is unaffected -- the data ends at 2024.
"""

from __future__ import annotations

import os

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "docs", "INVALIDATED.tsv")


def load() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    if not os.path.exists(_PATH):
        return out
    with open(_PATH, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 4 and c[0]:
                out[c[0]] = (c[1], c[3])
    return out


def guard(tags, hard: bool = True) -> list[str]:
    """Print a refusal for every listed tag. Raises when `hard` (the default)."""
    bad = load()
    hits = []
    for t in tags:
        for listed, (status, why) in bad.items():
            base = listed[:-4] if listed.endswith("_s42") else listed
            if t == listed or t == base:
                hits.append(t)
                print(f"!! {t}: {status}\n   {why}")
    if hits and hard:
        raise SystemExit(
            f"refusing to judge with {len(hits)} invalidated tag(s). "
            f"Re-run them on the fixed trainer first, or drop them from the comparison.")
    return hits
