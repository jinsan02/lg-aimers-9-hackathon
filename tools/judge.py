"""One adoption rule, in one place, imported by every judging tool.

`surf_report.py` and `arm_compare.py` were written months apart and judged the
same evidence differently. Measured 2026-08-15, all three disagreements were
live:

  interval   surf_report used `1.96 * se` at n=6; arm_compare used
             `t(.975, df=5) = 2.571`. At n=6 the normal quantile is 24% too
             narrow, so surf_report rejected candidates arm_compare parked.
  bar        surf_report adopted on `t >= 2.4` alone. The documented bar is
             `delta >= +3 AND t >= 2.4 AND n >= 6`; a candidate with mean +0.4
             and tiny variance passed surf_report and failed arm_compare.
  centring   surf_report scored `p - (p.mean() - r)`, subtracting the mean of
             the season being scored. That constant does not exist at
             submission time. It is a resolution diagnostic and must never be
             the default for an adoption call.

The rule below is the documented one. `n >= 6` is part of it: a t-statistic
from four seeds has not earned an adoption.

**What the t-statistic does and does not cover.** It is paired over seeds, so
it measures training RNG and early-stopping variation on one surface. It does
not measure season-transfer uncertainty, and it does not account for the
selection pressure of 692 recorded runs. A KEEP here is a necessary condition,
not a sufficient one.
"""

from __future__ import annotations

import numpy as np

# Keyed by n, not by degrees of freedom: T975[6] is t(.975, df=5) = 2.571.
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}

MIN_DELTA = 3.0
MIN_T = 2.4
MIN_N = 6

# The champion's fixed post-hoc calibration. Applied to both arms of a
# comparison, so it cancels in the paired difference -- unlike oracle centring,
# it uses only constants that exist at submission time.
SHIFT, SLOPE = 0.0052, 1.0416


def crit(n: int) -> float:
    """Two-sided 95% critical value. Falls back to the normal only past the table."""
    return T975.get(n, 1.96)


def bss(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    r = y.mean()
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2) / (r * (1 - r)))


def debias(p):
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.clip(1 / (1 + np.exp(-SLOPE * np.log(q / (1 - q)))) - SHIFT, 0, 1)


def centred_bss(p, y) -> float:
    """Diagnostic only -- uses the evaluation season's own mean.

    Never call this on an adoption path. It removes the candidate's calibration
    error using a constant that is unknown at submission time, so it answers
    "did resolution improve?" and not "would this score better?".
    """
    p, y = np.asarray(p, float), np.asarray(y, float)
    r = y.mean()
    return bss(p - (p.mean() - r), y)


def verdict(d) -> dict:
    """Paired per-seed deltas -> the adoption call.

    KEEP  mean >= +3 AND t >= 2.4 AND n >= 6
    DROP  the 95% upper bound is below +3 (the effect cannot reach the bar)
    PARK  everything else -- more seeds needed, or the effect is real but small
    """
    d = np.asarray(d, float)
    n = len(d)
    mean = float(d.mean()) if n else float("nan")
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t = mean / se if se else float("nan")
    c = crit(n)
    lo = mean - c * se if se == se else float("nan")
    hi = mean + c * se if se == se else float("nan")
    keep = bool(mean >= MIN_DELTA and t >= MIN_T and n >= MIN_N)
    call = "KEEP" if keep else ("DROP" if hi == hi and hi < MIN_DELTA else "PARK")
    return {"n": n, "mean": mean, "se": se, "t": t, "crit": c,
            "lo": lo, "hi": hi, "verdict": call,
            "median": float(np.median(d)) if n else float("nan"),
            "positive": int((d > 0).sum()),
            "gate_delta": bool(mean >= MIN_DELTA),
            "gate_t": bool(t >= MIN_T),
            "gate_n": bool(n >= MIN_N)}


def line(v: dict) -> str:
    return (f"mean {v['mean']:+.3f}  SE {v['se']:.3f}  t {v['t']:+.2f}  "
            f"95% CI [{v['lo']:+.2f}, {v['hi']:+.2f}]  n {v['n']}  "
            f"-> {v['verdict']}")
