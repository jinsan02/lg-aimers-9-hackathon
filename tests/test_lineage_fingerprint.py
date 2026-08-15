"""The training fingerprint has to work on the ids this competition actually has.

`row_id` is `TRAIN_0000001`, a string. The first version of `rowid_sha` cast to
int64, so every real run printed

    pack lineage 생성 실패: invalid literal for int() with base 10: 'TRAIN_0000001'

and wrote no fingerprint at all. The failure was caught -- provenance must never
kill a fit -- which is exactly why it could have gone unnoticed: the run
succeeds, the pack looks normal, and the `lineage` key is quietly `None`. It was
found in the first live run after the change, because the unit test had used
integer ids.

So this test uses the real column. It reads a few thousand ids straight out of
`data/train.csv` rather than inventing any.

Run: python tests/test_lineage_fingerprint.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import lineage                                                   # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def main():
    # Every 300th row, not the first N: train.csv is in season order, so
    # `nrows=5000` is entirely 2019 and a "fit vs all rows" comparison then
    # compares a set with itself.
    df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                     usecols=["row_id", "season"],
                     encoding="utf-8-sig").iloc[::300].reset_index(drop=True)
    ids = df["row_id"].tolist()
    print(f"  ({len(df):,} sampled rows, seasons "
          f"{df.season.min()}-{df.season.max()})")

    print("real row ids")
    check("1 they are strings, not integers",
          df["row_id"].dtype == object and isinstance(ids[0], str),
          f"e.g. {ids[0]!r}")
    h = lineage.rowid_sha(ids)
    check("2 rowid_sha handles them", isinstance(h, str) and len(h) == 16, h)
    check("3 order does not matter", lineage.rowid_sha(ids[::-1]) == h,
          "reversed -> same hash")
    check("4 dropping one row does", lineage.rowid_sha(ids[:-1]) != h)
    check("5 a different row set of the same size does",
          lineage.rowid_sha(ids[:-1] + ["TRAIN_9999999"]) != h)
    check("6 integer ids still work (other tables use them)",
          isinstance(lineage.rowid_sha([3, 1, 2]), str)
          and lineage.rowid_sha([3, 1, 2]) == lineage.rowid_sha([1, 2, 3]))

    print("\npack_record on a real frame")
    args = SimpleNamespace(val_season=2023, test_season=2024, feat_k=200,
                           lr=0.01, depth=8, tag="T", out="o")
    is_fit = df.season <= 2022
    is_val = df.season == 2023
    rec = lineage.pack_record(args, ["a", "b"], df, is_fit, is_val)
    check("7 the fit row-set hash is populated, not None",
          isinstance(rec.get("fit_rowid_sha"), str), str(rec["fit_rowid_sha"]))
    check("8 it differs from the hash of every row",
          rec["fit_rowid_sha"] != lineage.rowid_sha(ids),
          "the fit partition is a subset")
    check("9 both ends of the fit era are recorded",
          rec["rows"]["fit_min_season"] is not None
          and rec["rows"]["fit_max_season"] is not None,
          f"{rec['rows']['fit_min_season']}-{rec['rows']['fit_max_season']}")
    check("10 training-relevant flags are captured, display flags are not",
          "lr" in rec["flags"] and "depth" in rec["flags"]
          and "tag" not in rec["flags"] and "out" not in rec["flags"],
          sorted(rec["flags"]))

    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
