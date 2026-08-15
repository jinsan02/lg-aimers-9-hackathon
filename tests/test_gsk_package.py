"""GSK package renderer changes only the frozen deployment hooks."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from prepare_gsk_package import NEW_RETURN, OLD_RETURN, render


def main():
    src = open(os.path.join(ROOT, "src", "script_blend_b1s.py"),
               encoding="utf-8").read()
    got = render(src)
    assert 'f"cat_GSKDEP_cell_s{s}.pkl"' in got
    assert 'f"cat_B1S_cell_s{s}.pkl"' not in got
    assert OLD_RETURN not in got and got.count(NEW_RETURN) == 1
    # Base family, blend and all other post-processing remain frozen.
    assert got.count('f"cat_B1S_base_s{s}.pkl"') == 1
    assert "_W_CELL = 0.55" in got
    assert len(got) - len(src) == len(NEW_RETURN) - len(OLD_RETURN) + 3


if __name__ == "__main__":
    main()
