"""E-LB1 parabola algebra and package constants."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from elb_quadratic import solve


def main():
    # S(theta)=1000-400000*(theta-.004)^2 has optimum .004 and gain 6.4.
    score = lambda x: 1000 - 400000*(x-.004)**2
    got = solve(score(0), score(.01), score(-.01))
    assert abs(got["delta_star"]-.004) < 1e-12
    assert abs(got["expected_gain"]-6.4) < 1e-10
    assert abs(.55 + got["delta_star"] - .554) < 1e-12
    assert got["within_preregistered_bound"]
    try:
        solve(1, 2, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("convex probe was accepted")


if __name__ == "__main__":
    main()
