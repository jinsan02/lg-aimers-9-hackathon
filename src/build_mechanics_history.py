"""Build leakage-safe pitcher mechanics history from Trackman.

Release side and horizontal break are mirrored into a common handedness frame.
For prediction season S, every feature uses only Trackman seasons < S.
"""

import os
import sys

import numpy as np
import pandas as pd

DATA = "./data"
OUT = f"{DATA}/processed/mechanics_history.csv"
RAW = ["arm_angle", "rel_height", "rel_side_abs", "release_radius",
       "extension", "rel_scatter", "rel_speed", "spin_rate", "ivb",
       "hb_arm", "fb_speed", "fb_spin", "fb_ivb", "fb_hb_arm"]


def main():
    pmap = pd.read_csv(f"{DATA}/processed/pitcher_map2.csv")
    tm_to_pitcher = pmap.set_index("tm_id")["pitcher_id"]
    cols = ["season", "pitcher_trackman_id", "pitcher_hand", "pitch_type_group",
            "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
            "extension", "rel_height", "rel_side"]
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", usecols=cols)
    tm["pitcher_id"] = tm.pitcher_trackman_id.map(tm_to_pitcher)
    tm = tm.dropna(subset=["pitcher_id", "rel_height", "rel_side"]).copy()
    tm["pitcher_id"] = tm.pitcher_id.astype(np.int64)
    tm["rel_side_abs"] = tm.rel_side.abs()
    tm["arm_angle"] = np.degrees(np.arctan2(tm.rel_height, tm.rel_side_abs + 1e-6))
    tm["release_radius"] = np.hypot(tm.rel_height, tm.rel_side_abs)
    # Positive means break toward the pitcher's arm side in a mirrored frame.
    side_sign = np.sign(tm.rel_side.replace(0, np.nan)).fillna(1.0)
    tm["hb_arm"] = tm.horz_break * side_sign
    tm["ivb"] = tm.induced_vert_break
    fb = tm.pitch_type_group.eq("fastball")
    for src, dst in (("rel_speed", "fb_speed"), ("spin_rate", "fb_spin"),
                     ("ivb", "fb_ivb"), ("hb_arm", "fb_hb_arm")):
        tm[dst] = tm[src].where(fb)

    g = tm.groupby(["pitcher_id", "season"])
    agg = g.agg(
        mech_n=("arm_angle", "size"),
        arm_angle=("arm_angle", "mean"),
        rel_height=("rel_height", "mean"),
        rel_side_abs=("rel_side_abs", "mean"),
        release_radius=("release_radius", "mean"),
        extension=("extension", "mean"),
        rel_scatter=("arm_angle", "std"),
        rel_speed=("rel_speed", "mean"),
        spin_rate=("spin_rate", "mean"),
        ivb=("ivb", "mean"),
        hb_arm=("hb_arm", "mean"),
        fb_speed=("fb_speed", "mean"),
        fb_spin=("fb_spin", "mean"),
        fb_ivb=("fb_ivb", "mean"),
        fb_hb_arm=("fb_hb_arm", "mean"),
    ).reset_index()
    agg = agg[agg.mech_n >= 30].copy()

    rows = []
    max_season = int(tm.season.max()) + 1
    for pid, sub in agg.groupby("pitcher_id"):
        sub = sub.sort_values("season")
        for pred_season in range(int(sub.season.min()) + 1, max_season + 1):
            hist = sub[sub.season < pred_season]
            if hist.empty:
                continue
            last = hist.iloc[-1]
            prior = hist.iloc[:-1]
            rec = {"pitcher_id": pid, "season": pred_season,
                   "mech_last_season": int(last.season),
                   "mech_gap": pred_season - int(last.season),
                   "mech_last_n": float(last.mech_n)}
            for c in RAW:
                rec[f"mech_last_{c}"] = float(last[c]) if pd.notna(last[c]) else np.nan
                if len(prior):
                    ok = prior[c].notna()
                    if ok.any():
                        w = prior.loc[ok, "mech_n"].to_numpy(np.float64)
                        old = np.average(prior.loc[ok, c].to_numpy(np.float64), weights=w)
                        rec[f"mech_delta_{c}"] = rec[f"mech_last_{c}"] - old
                    else:
                        rec[f"mech_delta_{c}"] = np.nan
                else:
                    rec[f"mech_delta_{c}"] = np.nan
            rows.append(rec)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"linked pitches={len(tm):,} pitcher-seasons={len(agg):,}")
    print(f"saved {OUT}: rows={len(out):,}, pitchers={out.pitcher_id.nunique():,}, "
          f"season={out.season.min()}..{out.season.max()}")
    y25 = out[out.season == 2025]
    print(f"2025 lookup pitchers={len(y25):,} | median last_n={y25.mech_last_n.median():.0f} "
          f"gap>1={(y25.mech_gap > 1).mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
