"""Contracts for the GSK2 feature delta and pre-registered verdict."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gsk2_gate as gate


def main():
    got = gate.check_feature_delta(
        ["a", "b"], ["a", "skill_hat", "b", "skill_hat_vs_std"])
    assert got["added"] == ["skill_hat", "skill_hat_vs_std"]
    try:
        gate.check_feature_delta(
            ["a", "b"], ["b", "skill_hat", "a", "skill_hat_vs_std"])
    except ValueError:
        pass
    else:
        raise AssertionError("existing feature reorder was accepted")
    seg = {"first": 1, "second": 1, "R": 1, "F": 1}
    assert gate.gate_verdict(3.1, 2.5, 1, seg, 5) == "KEEP"
    assert gate.gate_verdict(0, 0, -1, seg, 2.9) == "DROP"
    assert gate.gate_verdict(1, 1, 1, seg, 4) == "HOLD"


if __name__ == "__main__":
    main()
