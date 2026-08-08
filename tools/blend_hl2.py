"""Compare the current submission blend with a halflife=2 member.

Seeds are averaged inside each configuration before any weight is selected.
Scores are bias-centered because the submission applies one global SHIFT after
blending.  Besides the optimistic full-season grid, weights selected on one
half of 2024 are evaluated on the other half.

Run on the 4070 after TH2_hl2 finishes:
  python tools/blend_hl2.py
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd


SHIP_CELL_SEEDS = {"42", "7", "13", "3", "4", "5"}


def load(tag: str, seeds: set[str] | None = None):
    files = sorted(glob.glob(f"./out/cat_{tag}_s*_val_preds.npz"))
    if seeds is not None:
        files = [f for f in files if f.rsplit("_s", 1)[1].split("_", 1)[0] in seeds]
    if not files:
        raise FileNotFoundError(tag)
    zs = [np.load(f, allow_pickle=True) for f in files]
    y = zs[0]["y"].astype(np.float64)
    if any(not np.array_equal(y, z["y"]) for z in zs[1:]):
        raise ValueError(f"target mismatch inside {tag}")
    return np.mean([z["pred"] for z in zs], axis=0).astype(np.float64), y, len(zs)


def centered_bss(y: np.ndarray, pred: np.ndarray) -> float:
    rate = float(y.mean())
    q = pred - (float(pred.mean()) - rate)
    return 1e5 * (1.0 - np.mean((np.clip(q, 0.0, 1.0) - y) ** 2)
                  / (rate * (1.0 - rate)))


def candidates(base, cell, hl2):
    # (base, cell, hl2). Include conservative fixed choices independently of
    # the validation-optimized grid.
    return {
        "current": (0.45, 0.55, 0.00),
        "replace_base": (0.00, 0.55, 0.45),
        "split_base_50": (0.225, 0.55, 0.225),
        "split_base_75hl": (0.1125, 0.55, 0.3375),
        "hl2_only": (0.00, 0.00, 1.00),
    }


def mix(weights, preds):
    return sum(w * p for w, p in zip(weights, preds))


def best_grid(y, preds, mask):
    best = (-np.inf, None)
    # 0.025 resolution: enough to choose a stable shipping region without
    # pretending that a percentage point is measurable here.
    vals = np.arange(0.0, 1.0001, 0.025)
    for wb in vals:
        for wc in vals:
            wh = 1.0 - wb - wc
            if wh < -1e-12:
                continue
            weights = (float(wb), float(wc), float(max(0.0, wh)))
            score = centered_bss(y[mask], mix(weights, preds)[mask])
            if score > best[0]:
                best = (score, weights)
    return best


def main() -> int:
    base, y, nb = load("VB2_base")
    cell, yc, nc = load("ZD5", SHIP_CELL_SEEDS)
    hl2, yh, nh = load("TH2_hl2")
    if not np.array_equal(y, yc) or not np.array_equal(y, yh):
        raise ValueError("target mismatch across configurations")
    preds = (base, cell, hl2)
    print(f"2024 n={len(y):,} | seeds base={nb} cell={nc} hl2={nh}")
    for name, pred in zip(("base", "cell", "hl2"), preds):
        print(f"{name:<12} centered={centered_bss(y, pred):9.3f} "
              f"mean={pred.mean():.6f} sd={pred.std():.6f}")

    current = centered_bss(y, mix((.45, .55, 0), preds))
    print("\nfixed candidates (full 2024; diagnostic)")
    for name, weights in candidates(*preds).items():
        score = centered_bss(y, mix(weights, preds))
        print(f"{name:<18} w={weights} score={score:9.3f} delta={score-current:+8.3f}")

    months = pd.read_csv("./data/train.csv", usecols=["season", "game_month"])
    months = months.loc[months.season == 2024, "game_month"].to_numpy()
    if len(months) != len(y):
        raise ValueError(f"2024 metadata mismatch {len(months)} != {len(y)}")
    first = months <= 6
    second = ~first
    full_score, full_w = best_grid(y, preds, np.ones(len(y), dtype=bool))
    a_score, a_w = best_grid(y, preds, first)
    b_score, b_w = best_grid(y, preds, second)
    print("\ngrid weights: base, cell, hl2")
    print(f"full fit       w={full_w} score={full_score:.3f} "
          f"delta={full_score-current:+.3f}")
    print(f"Mar-Jun fit    w={a_w} train={a_score:.3f} "
          f"Jul-Oct delta vs current={centered_bss(y[second], mix(a_w, preds)[second]) - centered_bss(y[second], mix((.45,.55,0), preds)[second]):+.3f}")
    print(f"Jul-Oct fit    w={b_w} train={b_score:.3f} "
          f"Mar-Jun delta vs current={centered_bss(y[first], mix(b_w, preds)[first]) - centered_bss(y[first], mix((.45,.55,0), preds)[first]):+.3f}")
    print("\nShipping weights should lie in the stable overlap of the two half-season "
          "solutions; do not ship the full-fit optimum blindly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
