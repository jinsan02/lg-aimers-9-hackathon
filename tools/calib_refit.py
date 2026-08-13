"""EXP-A -- re-derive the legacy calibration constants from OOF.

The shipped champion applies, in this order:

    p -> sigmoid(SLOPE * logit(clip(p)))     SLOPE = 1.0416
      -> p - SHIFT                           SHIFT = 0.0052

`CHAMPION_v11_RECIPE.md` part 6 lists both as weakly evidenced: the reasoning
that produced them is not usable as independent grounds today.

The improvement plan proposes fitting `p_cal = sigmoid(a*logit(p) + b)` by
minimising OOF Brier and comparing OOF Brier before and after. That fits and
scores on the same rows and cannot lose. Measured on B1-J OOF, the gap between
in-sample and transferred calibration is not a rounding detail:

    core, unseen 2024                        894.93
    + isotonic fitted on 2024 itself         929.48   (+34.56)
    + isotonic fitted on val2023, frozen     795.78   (-99.14)
        after removing the mean bias         814.89   (-81.93)

So this tool always reports both, and the transfer column is the one that
decides. A two-parameter map is far more constrained than isotonic and may
survive where it does not -- that is the question, not an assumption.

  python tools/calib_refit.py B1J6            # fit val2023, apply to 2024
  python tools/calib_refit.py B1S --no-target # submission shape, no future

Rule 9 / `measure-what-you-ship`: the constants that ship must come from the
OOF of the model that ships (B1S). B1J6 only establishes whether the *form*
transfers across a season boundary.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.optimize import minimize

SLOPE_LEGACY, SHIFT_LEGACY = 1.0416, 0.0052
EPS = 1e-6
SEEDS = (3, 4, 5, 6, 8, 13)
W_CELL = 0.55


def bss(y, p):
    r = y.mean()
    return float(1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / (r * (1 - r))))


def brier(y, p):
    return float(((np.clip(p, 0, 1) - y) ** 2).mean())


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_core(tag, split, seeds=SEEDS):
    out = {}
    for arm in ("base", "cell"):
        zs = [np.load(f"./out/cat_{tag}_{arm}_s{s}_{split}_preds.npz",
                      allow_pickle=True) for s in seeds]
        rid = zs[0]["row_id"]
        for q in zs[1:]:
            if not np.array_equal(q["row_id"], rid):
                raise SystemExit(f"{tag} {arm} {split}: seeds disagree on row_id")
        out[arm] = (np.mean([q["pred"].astype(float) for q in zs], 0),
                    zs[0]["y"].astype(float), rid)
    (b, y, rb), (c, yc, rc) = out["base"], out["cell"]
    if not np.array_equal(rb, rc):
        raise SystemExit(f"{tag} {split}: base and cell are not the same rows")
    if not np.array_equal(y, yc):
        raise SystemExit(f"{tag} {split}: base and cell disagree on y")
    return (1 - W_CELL) * b + W_CELL * c, y


def legacy(p):
    return np.clip(sig(SLOPE_LEGACY * logit(p)) - SHIFT_LEGACY, 0, 1)


def fit_ab(y, p):
    """Minimise Brier over sigmoid(a*logit(p) + b). Convex enough in practice;
    started from the identity so a failed solve is visible as a=1, b=0."""
    z = logit(p)

    def obj(t):
        return brier(y, sig(t[0] * z + t[1]))

    r = minimize(obj, x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                 options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 2000})
    return float(r.x[0]), float(r.x[1])


def fit_slope_shift(y, p):
    """The legacy functional form, refitted: sigmoid(a*logit(p)) - s."""
    z = logit(p)

    def obj(t):
        return brier(y, np.clip(sig(t[0] * z) - t[1], 0, 1))

    r = minimize(obj, x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                 options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 2000})
    return float(r.x[0]), float(r.x[1])


def row(name, y, p):
    return f"  {name:34s} brier {brier(y, p):.7f}   BSS {bss(y, p):8.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--source", default="val")
    ap.add_argument("--target", default="test")
    ap.add_argument("--no-target", action="store_true",
                    help="submission shape: there is no future season to transfer to")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a = ap.parse_args()
    seeds = tuple(int(x) for x in a.seeds.split(",") if x.strip())

    ps, ys = load_core(a.tag, a.source, seeds)
    print(f"=== {a.tag} | source '{a.source}' {len(ys):,} rows, target mean {ys.mean():.6f} ===")
    print(row("raw", ys, ps))
    print(row(f"legacy slope {SLOPE_LEGACY} shift {SHIFT_LEGACY}", ys, legacy(ps)))

    a1, b1 = fit_ab(ys, ps)
    a2, s2 = fit_slope_shift(ys, ps)
    print(row(f"refit sigmoid(a*logit+b)  a={a1:.4f} b={b1:+.4f}", ys, sig(a1 * logit(ps) + b1)))
    print(row(f"refit legacy form  a={a2:.4f} shift={s2:+.5f}", ys,
              np.clip(sig(a2 * logit(ps)) - s2, 0, 1)))
    print("  ^ these two lines are IN-SAMPLE. They cannot lose. Not evidence.")

    if a.no_target:
        print("\n  no target split: transfer is not testable here. The constants above "
              "are candidates only, and only if the form transferred on a surface "
              "that has a future season.")
        return 0

    pt, yt = load_core(a.tag, a.target, seeds)
    print(f"\n=== transferred to '{a.target}' {len(yt):,} rows, target mean {yt.mean():.6f} ===")
    base = bss(yt, pt)
    print(row("raw", yt, pt))
    print(row(f"legacy slope {SLOPE_LEGACY} shift {SHIFT_LEGACY}", yt, legacy(pt))
          + f"   delta {bss(yt, legacy(pt)) - base:+8.2f}")
    q1 = sig(a1 * logit(pt) + b1)
    q2 = np.clip(sig(a2 * logit(pt)) - s2, 0, 1)
    print(row(f"frozen sigmoid(a*logit+b)  a={a1:.4f} b={b1:+.4f}", yt, q1)
          + f"   delta {bss(yt, q1) - base:+8.2f}")
    print(row(f"frozen legacy form  a={a2:.4f} shift={s2:+.5f}", yt, q2)
          + f"   delta {bss(yt, q2) - base:+8.2f}")

    ao, bo = fit_ab(yt, pt)
    print(row(f"oracle on target itself  a={ao:.4f} b={bo:+.4f}", yt,
              sig(ao * logit(pt) + bo))
          + f"   delta {bss(yt, sig(ao * logit(pt) + bo)) - base:+8.2f}")
    print("  ^ oracle fits the answer key. It is the ceiling, never a candidate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
