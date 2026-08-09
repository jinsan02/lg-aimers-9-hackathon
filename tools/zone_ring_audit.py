"""Audit the 5x5-ring interpretation using recoverable outcome labels.

The released rows do not contain the current pitch location, so these are
outcome-mode proxies rather than literal geometric cells:
  heart      = middle
  inner_band = strike-like, but not middle or ball-like
  outer_band = ball-like, but not strike-like or middle
The audit keeps an overlap bucket because the published rates are not a
partition (strike-like and ball-like can overlap).
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")


def main():
    import failmode as fm

    cols = [
        "row_id", "season", "game_type", "pitcher_id", "control_success",
        "asof_pitcher_n", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    ]
    df = pd.read_csv("data/train.csv", usecols=cols)
    lab = fm._pitch_labels(df, modes=("middle", "ball", "strike"))
    d = pd.concat([df, lab], axis=1)
    d = d[(d.game_type == "R") & d.season.isin([2022, 2023, 2024])].copy()
    d = d.dropna(subset=["middle", "ball", "strike"])

    mid = d.middle.eq(1)
    ball = d.ball.eq(1)
    strike = d.strike.eq(1)
    groups = {
        "heart_middle": mid,
        "inner_strike_band": strike & ~mid & ~ball,
        "outer_ball_band": ball & ~strike & ~mid,
        "strike_ball_overlap": strike & ball & ~mid,
        "other_nonmiddle": ~mid & ~(strike ^ ball),
    }

    rows = []
    for season in (2022, 2023, 2024):
        sm = d.season.eq(season)
        for name, mask in groups.items():
            take = sm & mask
            n = int(take.sum())
            rows.append({
                "season": season,
                "group": name,
                "n": n,
                "share": n / int(sm.sum()),
                "success": d.loc[take, "control_success"].mean(),
            })
    out = pd.DataFrame(rows)
    print("=== zone-ring proxies (R league) ===")
    print(out.to_string(index=False, formatters={
        "share": lambda x: f"{x:.4f}",
        "success": lambda x: f"{x:.4f}",
    }))

    wide = out.pivot(index="group", columns="season", values="share")
    print("\n=== share changes ===")
    wide["d22_23"] = wide[2023] - wide[2022]
    wide["d23_24"] = wide[2024] - wide[2023]
    print(wide[["d22_23", "d23_24"]].to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\nCaveat: these are overlapping semantic labels, not literal x/y cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
