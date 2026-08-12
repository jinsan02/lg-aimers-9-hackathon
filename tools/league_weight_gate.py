"""B1 -- does a per-league base/cell weight survive a season boundary?

LEVERS_NEXT O2 left the league row blank with "no same-regime F source season".
That is not true. The judging surface drops only *old-regime* F (<=2022), so
**2023 keeps 25,686 F rows** and 2024 has 30,010. Both seasons already have
base and cell predictions from the same machine, so this needs no GPU.

Fit the per-league weight on the source season, freeze it, apply to the target.
Never fit and score on the same season -- that is the selection illusion that
made cell-posterior-stack look like +101 before it turned into -52.

Reuses centered_bss / fit_weight / transfer_segment_pred from orthogonal_gate.

  python tools/league_weight_gate.py                       # MVA_native + MVCELL
  python tools/league_weight_gate.py VB2_base ZD5
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "./tools")
from orthogonal_gate import CURRENT_W, centered_bss, fit_weight, transfer_segment_pred

SEASON_OF = {"val": 2023, "test": 2024}


def load(tag: str, split: str, seeds: set[int] | None = None):
    # Anchor both ends. A loose "*MVCELL*" also matches MVCELL22 -- which is the
    # 2022 season (247,472 rows) -- and MVCELL5K. Silent cross-season mixing is
    # exactly the failure mode this project keeps paying for.
    paths = sorted(glob.glob(f"./out/*_{tag}_s*_{split}_preds.npz")
                   or glob.glob(f"./out/*_{tag}_{split}_preds.npz"))
    if seeds is not None:
        paths = [p for p in paths
                 if int(p.rsplit("_s", 1)[1].split("_", 1)[0]) in seeds]
    if not paths:
        raise FileNotFoundError(f"no predictions: {tag} {split}")
    z = [np.load(p, allow_pickle=True) for p in paths]
    y = z[0]["y"].astype(np.float64)
    if any(not np.array_equal(y, q["y"]) for q in z[1:]):
        raise ValueError(f"seed y mismatch: {tag} {split}")
    return np.stack([q["pred"].astype(np.float64) for q in z]).mean(0), y, len(paths)


def league_of(season: int, n: int) -> np.ndarray:
    raw = pd.read_csv("./data/train.csv", encoding="utf-8-sig",
                      usecols=["season", "game_type", "control_success"])
    raw = raw[raw.season == season].reset_index(drop=True)
    if len(raw) != n:
        raise ValueError(f"{season}: raw {len(raw)} vs preds {n}")
    return raw.game_type.to_numpy(), raw.control_success.to_numpy(np.float64)


def report(name, bs, cs, ys, ls, bt, ct, yt, lt):
    """Fit on source (s), freeze, apply to target (t). Print the target verdict."""
    fixed = (1 - CURRENT_W) * bt + CURRENT_W * ct
    routed, dmin, dmax = transfer_segment_pred(bs, cs, ys, bt, ct, ls, lt)
    base_score = centered_bss(yt, fixed)
    new_score = centered_bss(yt, routed)

    print(f"\n=== {name} ===")
    print(f"  source per-league w:", end="")
    wg = fit_weight(bs, cs, ys)
    for lg in np.unique(ls):
        m = ls == lg
        print(f"   {lg} {fit_weight(bs[m], cs[m], ys[m]):.3f} (n={m.sum():,})", end="")
    print(f"   | global {wg:.3f}")
    print(f"  frozen w delta range on target: [{dmin:+.4f}, {dmax:+.4f}]")
    print(f"  target centered BSS   fixed w=.55 {base_score:9.3f}"
          f"   league-routed {new_score:9.3f}   delta {new_score - base_score:+8.3f}")

    for lg in np.unique(lt):
        m = lt == lg
        d = centered_bss(yt[m], routed[m]) - centered_bss(yt[m], fixed[m])
        print(f"    {lg}  n={m.sum():>7,}  delta {d:+8.3f}")
    return new_score - base_score


def main() -> int:
    base_tag = sys.argv[1] if len(sys.argv) > 1 else "MVA_native"
    cell_tag = sys.argv[2] if len(sys.argv) > 2 else "MVCELL"
    arms = {}
    for split, season in SEASON_OF.items():
        b, yb, nb = load(base_tag, split)
        c, yc, nc = load(cell_tag, split)
        if not np.array_equal(yb, yc):
            raise ValueError(f"{split}: base/cell y mismatch")
        lg, yraw = league_of(season, len(yb))
        if not np.array_equal(yraw, yb):
            raise ValueError(f"{season}: raw/pred alignment mismatch")
        arms[season] = (b, c, yb, lg)
        print(f"{season}: base {nb} seed(s), cell {nc} seed(s), {len(yb):,} rows, "
              f"F {np.mean(lg == 'F'):.2%}")

    s, t = arms[2023], arms[2024]
    fwd = report("2023 -> 2024  (the deployment direction)", *s, *t)
    rev = report("2024 -> 2023  (sign-stability check only)", *t, *s)

    print("\n--- gate ---")
    print(f"  forward delta {fwd:+.3f}  (adopt >= +3, close < +1)")
    print(f"  reverse delta {rev:+.3f}  (must share the forward sign)")
    if fwd >= 3 and rev > 0:
        print("  PASS -- measure the shipping constant on the submission surface,")
        print("          NOT here (this surface uses --drop-f-pre; the submission does not)")
    elif fwd < 1:
        print("  CLOSE -- record O2 as CLOSED in docs/SETTLED.md")
    else:
        print("  HOLD -- keep as a candidate, do not submit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
