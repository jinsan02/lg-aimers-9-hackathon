"""Exact duplicate audit for skill_pc_hat - skill_hat."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = joblib.load(ROOT / "model/cat_B1J6_base_s3.pkl")
    cell = joblib.load(ROOT / "model/cat_GSKDEP_cell_s3.pkl")
    bf, cf = set(base["features"]), set(cell["features"])
    both_cell = {"skill_pc_hat", "skill_hat"} <= cf
    base_pc_only = "skill_pc_hat" in bf and "skill_hat" not in bf
    # d = [-1, +1] dot [skill_pc_hat, skill_hat] exactly.  This is algebraic,
    # so the arm-level reconstructibility is one without fitting or target use.
    result = {
        "formula": "skill_pc_hat - skill_hat",
        "base": {"n_features": len(base["features"]),
                 "skill_pc_hat": "skill_pc_hat" in bf,
                 "skill_hat": "skill_hat" in bf},
        "champion_cell": {"n_features": len(cell["features"]),
                          "skill_pc_hat": "skill_pc_hat" in cf,
                          "skill_hat": "skill_hat" in cf,
                          "exact_linear_reconstructibility_r2": 1.0
                          if both_cell else None},
        "base_only_equivalence": (
            "Because base already contains skill_pc_hat, adding d is an invertible "
            "linear reparameterisation of adding skill_hat. The general-skill "
            "additive base arm was already measured and did not promote."),
        "verdict": "DUPLICATE" if both_cell and base_pc_only else "INCONCLUSIVE",
        "reason": (
            "Both components coexist in the shipped 123-feature cell; in the "
            "121-feature base, d adds exactly the same information as the already-"
            "measured skill_hat additive arm."),
    }
    out = ROOT / "out/skill_disagreement_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
