"""P1 validation-contract fixes -- do the flags actually change what they claim?

Two confirmed leaks into the selection view, both fixed behind flags so the
legacy-like B0-JL surface stays reproducible:

1. `target_enc.build_te` took its global shrink prior from the whole frame, so
   the validation season's target mean entered every shrunk rate. The expanding
   count/sum body was already safe (season shift(1)); only this scalar leaked.
2. `skill.build` gave the first season coefficients fit on the **whole** frame,
   so 2019 training rows received a regression built from future targets.

Run: python tests/test_p1_contract.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import skill as sk                                             # noqa: E402
import target_enc as te                                        # noqa: E402

VAL_SEASON = 2023
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def main() -> int:
    cols = ["row_id", "season", "pitcher_id", "batter_id", "balls_before",
            "strikes_before", "control_success"]
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    fit = (d.season < VAL_SEASON).to_numpy()

    # --- 1. TE prior -------------------------------------------------------
    keys = ["pitcher_id"]
    legacy = te.build_te(d, keys, k=50)
    fixed = te.build_te(d, keys, k=50, fit_mask=fit)
    whole, part = float(d.control_success.mean()), float(d.control_success[fit].mean())
    check("fit partition mean differs from whole-frame mean",
          abs(whole - part) > 1e-6, f"{whole:.6f} vs {part:.6f}")
    rate_col = [c for c in legacy.columns if c.endswith("_rate")][0]
    n_col = [c for c in legacy.columns if c.endswith("_n")][0]
    moved = float(np.nanmax(np.abs(legacy[rate_col].to_numpy()
                                   - fixed[rate_col].to_numpy())))
    check("restricting the prior actually moves the TE rate", moved > 1e-9,
          f"max |delta| {moved:.3e}")
    # The expanding body must be untouched -- counts do not depend on the prior.
    check("expanding count body unchanged",
          np.allclose(legacy[n_col].to_numpy(), fixed[n_col].to_numpy(),
                      equal_nan=True))
    # And the shift must follow the sign of (fit prior - whole prior). Success
    # rates fall season over season, so dropping 2023+ raises the fit prior.
    delta = float(np.nanmean(fixed[rate_col].to_numpy()
                             - legacy[rate_col].to_numpy()))
    check("rate moves toward the fit-partition prior",
          np.sign(delta) == np.sign(part - whole),
          f"prior {whole:.6f} -> {part:.6f}, rate delta {delta:+.3e}")

    # --- 2. skill first season --------------------------------------------
    sub = d[d.season <= 2020].copy()
    for c in sk.FEAT:
        if c not in sub.columns:
            sub[c] = 0.5
    legacy_pack = sk.build(sub)
    fixed_pack = sk.build(sub, neutral_first=True)
    first = min(sub.season.unique())
    check("legacy fits the first season on the whole frame",
          not isinstance(legacy_pack["coef"][first], str))
    check("neutral_first marks the first season as having no past",
          isinstance(fixed_pack["coef"][first], str))
    check("later seasons are identical",
          all(np.allclose(legacy_pack["coef"][s], fixed_pack["coef"][s])
              for s in sub.season.unique() if s != first))

    a, _ = sk.add(sub.copy(), legacy_pack)
    b, _ = sk.add(sub.copy(), fixed_pack)
    col = "skill_hat"
    m = (sub.season == first).to_numpy()
    check("first-season estimate becomes missing", bool(b[col][m].isna().all()),
          f"{int(m.sum()):,} rows")
    check("first season was NOT missing before", bool(a[col][m].notna().any()))
    check("other seasons unchanged",
          bool(a[col][~m].equals(b[col][~m])))

    print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
