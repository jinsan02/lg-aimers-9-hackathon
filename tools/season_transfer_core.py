"""Family transfer on the blended surface — the one the submission is scored on.

`season_transfer_map.py` measures each arm separately. That is the right way to
see the base/cell split, but it is not what the champion pays for: the shipped
prediction is `0.45*base + 0.55*cell`, and a family the two arms use for
*different* things can matter more, or less, in the blend than in either arm.

So this permutes a family in **both arms at once, with the same permutation
index**, and scores the blended prediction. Anything else would be measuring a
model that is not deployed.

Pairing: each transition's base and cell model come from the same host and the
same surface, but not the same seed (MVN3 is seeded 3/4/5, MVCELL 42). The blend
the champion ships is a seed ensemble anyway, so a cross-seed pair is the honest
shape here; it is recorded in the output.

Run (CPU, ~30 min):  python tools/season_transfer_core.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import feature_families as ff                                    # noqa: E402
from season_transfer_map import TARGET, bss, frames, predict     # noqa: E402

W_CELL = 0.55                      # the champion's fixed blend weight

# (label, base pkl, cell pkl, fit era, unseen season)
PAIRS = [
    ("T21", "cat_MV21_base.pkl", "cat_MV21_cell.pkl", 2021, 2022),
    ("T22", "cat_MVB22_native.pkl", "cat_MVCELL22_s42.pkl", 2022, 2023),
    ("T23", "cat_MVN3_s3.pkl", "cat_MVCELL_s42.pkl", 2023, 2024),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/season_transfer_core.csv")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    header = list(pd.read_csv(os.path.join(ROOT, "data", "test.csv"), nrows=0,
                              encoding="utf-8-sig").columns)
    full = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                       encoding="utf-8-sig", usecols=header + [TARGET])

    rows = []
    t_start = time.time()
    for label, bp, cp, fit_era, unseen in PAIRS:
        pb = joblib.load(os.path.join(ROOT, "model", bp))
        pc = joblib.load(os.path.join(ROOT, "model", cp))
        ff.validate(pb["features"])
        if list(pb["features"]) != list(pc["features"]):
            raise SystemExit(f"{label}: base and cell feature lists differ")
        fams = {g: [c for c in pb["features"] if ff.FAMILY_OF[c] == g]
                for g in ff.ORDER}
        print(f"\n=== {label}  {bp} + {cp}  fit<={fit_era}  unseen={unseen}")

        for season in (unseen - 2, unseen - 1, unseen):
            if season not in set(full["season"]):
                continue
            horizon = ("next_season" if season == unseen
                       else "in_sample_val" if season == fit_era
                       else "in_sample_old")
            # Each arm transforms with its own artifact -- that is what the
            # deployed blend does; the two arms do not share feature tables.
            Xb, y, league = frames(full, pb, season)
            Xc, y2, _ = frames(full, pc, season)
            if not np.array_equal(y, y2):
                raise SystemExit(f"{label} @{season}: arm targets differ")

            for lg in ("R", "F"):
                m = league == lg
                if m.sum() < 5000:
                    continue
                Bl = Xb[m].reset_index(drop=True)
                Cl = Xc[m].reset_index(drop=True)
                yl = y[m]

                def blended(B, C):
                    return ((1 - W_CELL) * predict(pb, B)
                            + W_CELL * predict(pc, C))

                intact = bss(yl, blended(Bl, Cl))
                t0 = time.time()
                per_fam = {}
                for g, cols in fams.items():
                    ds = []
                    for r in range(args.repeats):
                        rng = np.random.default_rng(1000 * r + 7)
                        idx = rng.permutation(len(Bl))
                        Bp, Cp = Bl.copy(), Cl.copy()
                        for c in cols:
                            Bp[c] = Bp[c].to_numpy()[idx]
                            Cp[c] = Cp[c].to_numpy()[idx]
                        ds.append(intact - bss(yl, blended(Bp, Cp)))
                    per_fam[g] = (float(np.mean(ds)),
                                  float(np.std(ds, ddof=1)))
                tot = sum(max(v[0], 0.0) for v in per_fam.values()) or 1.0
                for g, (mu, sd) in per_fam.items():
                    rows.append(dict(
                        pair=label, arm="core", base_model=bp, cell_model=cp,
                        fit_era=fit_era, season=season, horizon=horizon,
                        league=lg, n_rows=int(m.sum()),
                        base_bss=round(intact, 3), family=g,
                        n_cols=len(fams[g]), dbss=round(mu, 4),
                        dbss_sd=round(sd, 4),
                        share=round(100.0 * max(mu, 0.0) / tot, 3)))
                print(f"  {season} [{horizon}] {lg} n={int(m.sum()):>7,} "
                      f"BSS {intact:9.2f} ({time.time() - t0:.0f}s)  " +
                      "  ".join(f"{g[:7]} {per_fam[g][0]:+.0f}"
                                for g in sorted(per_fam,
                                                key=lambda k: -per_fam[k][0])[:4]))
            del Xb, Xc

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} rows, "
          f"{time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
