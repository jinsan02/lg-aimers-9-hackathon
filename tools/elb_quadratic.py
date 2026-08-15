"""Compute the pre-registered E-LB1 parabola after the user supplies scores."""

from __future__ import annotations

import argparse
import json


def solve(s0, sp, sm, h=0.01):
    curvature = sp + sm - 2*s0
    if curvature >= 0:
        raise ValueError(f"non-concave score probe: second difference {curvature}")
    delta = -h * (sp-sm) / (2*curvature)
    gain = (sp-sm)**2 / (-8*curvature)
    return {"baseline": s0, "plus": sp, "minus": sm, "h": h,
            "second_difference": curvature, "delta_star": delta,
            "expected_gain": gain, "expected_score": s0+gain,
            "within_preregistered_bound": abs(delta) <= .02}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=float, default=1108.4333490288)
    ap.add_argument("--plus", type=float, required=True)
    ap.add_argument("--minus", type=float, required=True)
    ap.add_argument("--h", type=float, default=.01)
    a = ap.parse_args()
    ans = solve(a.base, a.plus, a.minus, a.h)
    print(json.dumps(ans, indent=2))
    return 0 if ans["within_preregistered_bound"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
