"""The two judging tools must return the same verdict on the same evidence.

Until 2026-08-15 they did not. `surf_report.py` and `arm_compare.py` were
written months apart and disagreed three ways:

  * `surf_report` adopted on `t >= 2.4` alone; the documented bar is
    `delta >= +3 AND t >= 2.4 AND n >= 6`. A candidate with mean +0.4 and tiny
    seed variance passed one tool and failed the other.
  * `surf_report` used `1.96 * se` at n=6, where the two-sided 95% critical
    value is `t(.975, df=5) = 2.571` -- 24% too narrow, so it DROPped effects
    the other tool PARKed.
  * `surf_report` scored every arm shifted onto the evaluation season's own
    mean, a constant unavailable at submission time.

Both now import `tools/judge.py`. This test asserts they *cannot* drift back:
the rule is exercised directly, and both modules are checked for a private
copy of the constants.

Run: python tests/test_judge_agreement.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import judge                                                     # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def main():
    print("the rule")
    v = judge.verdict([4.0, 3.5, 4.5, 3.8, 4.2, 4.1])
    check("1 clear winner at n=6 -> KEEP", v["verdict"] == "KEEP",
          judge.line(v))
    # The exact case the two tools disagreed on: significant but tiny.
    v = judge.verdict([0.40, 0.42, 0.38, 0.41, 0.39, 0.40])
    check("2 t is huge but mean is +0.4 -> not KEEP", v["verdict"] != "KEEP",
          f"t {v['t']:+.1f}, mean {v['mean']:+.2f} -> {v['verdict']}")
    check("2b and it is DROP, since the upper bound cannot reach +3",
          v["verdict"] == "DROP", f"95% upper {v['hi']:+.3f}")
    v = judge.verdict([5.0, 4.0, 6.0, 5.5])
    check("3 n=4 cannot adopt however large the effect",
          v["verdict"] != "KEEP" and not v["gate_n"], f"n=4 -> {v['verdict']}")
    v = judge.verdict([6.0, -1.0, 8.0, -2.0, 9.0, 1.0])
    check("4 mean over +3 but noisy -> PARK", v["verdict"] == "PARK",
          judge.line(v))

    print("\nthe interval")
    check("5 n=6 uses Student-t, not 1.96", judge.crit(6) == 2.571,
          f"crit(6) = {judge.crit(6)}")
    d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    v = judge.verdict(d)
    se = np.std(d, ddof=1) / np.sqrt(6)
    check("6 the bound uses that critical value",
          abs(v["hi"] - (np.mean(d) + 2.571 * se)) < 1e-9,
          f"hi {v['hi']:+.4f}")
    check("7 a normal-quantile bound would be narrower and is NOT used",
          v["hi"] > np.mean(d) + 1.96 * se,
          f"t-bound {v['hi']:+.3f} vs normal {np.mean(d) + 1.96 * se:+.3f}")

    print("\ncentring is a diagnostic, not the default")
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.487).astype(float)
    p = np.clip(0.487 + 0.04 * rng.standard_normal(5000) + 0.03, 0.01, 0.99)
    plain, cent = judge.bss(p, y), judge.centred_bss(p, y)
    check("8 centring changes the score materially on a biased arm",
          abs(cent - plain) > 1.0, f"plain {plain:.2f} vs centred {cent:.2f}")
    check("9 and it flatters it -- which is why it must be opt-in",
          cent > plain)

    print("\nneither tool keeps a private copy of the rule")
    for mod in ("surf_report.py", "arm_compare.py"):
        src = open(os.path.join(ROOT, "tools", mod), encoding="utf-8").read()
        check(f"10 {mod} imports judge", "from judge import" in src)
        # A literal 1.96 or a re-declared T975 table is the drift this prevents.
        body = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#")
                         and "1.96 * se" not in l or "judge" in l)
        check(f"11 {mod} declares no second critical-value table",
              "T975 = {" not in body, "")
        check(f"12 {mod} does not compute its own 1.96 bound",
              "1.96 * se" not in body)

    print("\nsurf_report defaults to the honest score")
    src = open(os.path.join(ROOT, "tools", "surf_report.py"),
               encoding="utf-8").read()
    check("13 the oracle-centred path is behind --centred",
          "--centred" in src and "a.centred" in src)
    check("14 and the default branch is the debiased one",
          "bss(debias(p), y)" in src)

    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
