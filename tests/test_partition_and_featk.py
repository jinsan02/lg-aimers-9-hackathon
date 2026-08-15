"""Two contracts that were stated in prose and not in code.

**The fit era must end before the validation season.** `_assert_partitions`
checked that fit held neither the val nor the test season, and that nothing
exceeded the test season — but never that the *latest* fit season is earlier
than val. So `--val-season 2022` with no `--test-season` left 2023 and 2024 in
fit, and `--val-season 2022 --test-season 2024` left 2023, and both printed
"partitions ok". An audit of all 692 ledger rows found no run that took either
path, so nothing was invalidated; the hole is closed here so it cannot open.

**Shrinkage strength must travel with the artifact.** `fpipe.fit` passed
`k=args.feat_k` while `fpipe.transform` called `add_features` with no `k` and
`feat_k` was never stored, so any model trained at `--feat-k != 200` would have
scored on features it never learned. Every ledger run used 200 (explicitly or by
default), so again nothing was invalidated — and again the contract is now
enforced rather than assumed.

Run: python tests/test_partition_and_featk.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import features                                                  # noqa: E402
import train_gbdt2 as T                                          # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def partitions(val, test, seasons=(2019, 2020, 2021, 2022, 2023, 2024)):
    df = pd.DataFrame({"season": list(seasons)})
    args = SimpleNamespace(val_season=val, test_season=test)
    is_val = df.season == val
    if test:
        df["_is_test"] = df.season == test
        is_fit = ~is_val & ~df["_is_test"]
    else:
        is_fit = ~is_val
    return df, args, is_val, is_fit


def rejects(val, test, seasons=(2019, 2020, 2021, 2022, 2023, 2024)):
    try:
        T._assert_partitions(*partitions(val, test, seasons))
        return False
    except AssertionError:
        return True


def main():
    print("partition guard")
    check("1 val=2022, no test season, 2023/2024 still in fit -> REJECTED",
          rejects(2022, None), "the exact case that used to print 'ok'")
    check("2 val=2022, test=2024, 2023 still in fit -> REJECTED",
          rejects(2022, 2024))
    check("3 the submission surface (val=2024, fit <= 2023) -> ACCEPTED",
          not rejects(2024, None))
    check("4 the judging surface (val=2023, test=2024, fit <= 2022) -> ACCEPTED",
          not rejects(2023, 2024, (2019, 2020, 2021, 2022, 2023, 2024)))
    check("5 the stress surface (val=2022, test=2023, fit <= 2021) -> ACCEPTED",
          not rejects(2022, 2023, (2019, 2020, 2021, 2022, 2023)))

    print("\nfeat_k travels with the artifact")
    n = 4000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "asof_pitcher_n": rng.integers(0, 900, n).astype(float),
        "asof_pitcher_success_rate": rng.normal(0.52, 0.05, n),
        "asof_batter_n": rng.integers(0, 900, n).astype(float),
        "asof_batter_success_rate": rng.normal(0.52, 0.05, n),
    })
    priors = {c: float(df[c].mean()) for c in df.columns if c.endswith("rate")}
    a40, _ = features.add_features(df.copy(), priors, k=40)
    a200, _ = features.add_features(df.copy(), priors, k=200)
    shr = [c for c in a40.columns if c.endswith("_shr")]
    check("6 k really changes the shrunk columns", bool(shr) and
          not np.allclose(a40[shr[0]], a200[shr[0]]),
          f"{len(shr)} shrunk column(s), e.g. {shr[0] if shr else '-'}")
    check("7 features.K is the documented default", features.K == 200,
          f"K = {features.K}")

    import inspect

    import fpipe
    src = inspect.getsource(fpipe)
    check("8 fpipe.fit stores feat_k in the artifact",
          'art["feat_k"]' in src)
    check("9 fpipe.transform reads it back",
          'art.get("feat_k"' in src)
    check("10 transform no longer calls add_features without k",
          'add_features(df, art["priors"])' not in src,
          "the un-keyed call is the defect")

    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
