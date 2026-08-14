"""Information blocks — a partition that is closed under the frame's own algebra.

`feature_families.py` splits the 121 by representation, which is what you want
for reading how a fitted model routes. It is *not* what you want for asking
whether information transfers, because the numeric block of the frame has rank
90 of 112: 22 columns are exact linear functions of the others. The clearest
case is the triple

    std_asof_pitcher_success_rate  =  asof_pitcher_success_rate
                                    + std_asof_pitcher_success_rate_delta

Permuting STD_DELTA on its own leaves both parents in place, so the difference
is still there for the asking and the measured drop reports only how much the
trees happened to route through the precomputed column. The same holds for
SEASON_STD and RAW_ASOF -- any one of the three is recoverable from the other
two, so *none* of those three families can be read as information on its own.

So this file groups columns by **where the information comes from** rather than
by how it is shaped, and `closure_report` checks the property that makes the
grouping usable: no column inside a block may be reconstructible from the
columns outside it. If that holds, permuting a block really does remove
something.

Six blocks, exhaustive and disjoint over the champion's 121.
"""

from __future__ import annotations

BLOCKS: dict[str, list[str]] = {
    # everything computed from this pitcher's own past outcomes, at any
    # shrinkage, in any window, plus the regression that summarises them
    "PITCHER_HISTORY": [
        "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
        "asof_pitcher_success_rate_shr", "asof_pitcher_reverse_rate_shr",
        "asof_pitcher_middle_rate_shr", "asof_pitcher_ball_rate_shr",
        "asof_pitcher_strike_rate_shr", "asof_pitcher_fastball_rate_shr",
        "asof_pitcher_breaking_rate_shr", "asof_pitcher_offspeed_rate_shr",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "form_delta1", "form_delta5",
        "std_pitcher_n", "std_asof_pitcher_success_rate",
        "std_asof_pitcher_reverse_rate", "std_asof_pitcher_middle_rate",
        "std_asof_pitcher_ball_rate", "std_asof_pitcher_strike_rate",
        "std_pitchmix_n", "std_asof_pitcher_fastball_rate",
        "std_asof_pitcher_breaking_rate", "std_asof_pitcher_offspeed_rate",
        "std_asof_pitcher_success_rate_delta",
        "std_asof_pitcher_reverse_rate_delta",
        "std_asof_pitcher_middle_rate_delta",
        "std_asof_pitcher_ball_rate_delta",
        "std_asof_pitcher_strike_rate_delta",
        "std_asof_pitcher_fastball_rate_delta",
        "std_asof_pitcher_breaking_rate_delta",
        "std_asof_pitcher_offspeed_rate_delta",
        "te_pitcher_ratio", "te_pitcher_rate", "te_pitcher_n",
        "skill_pc_hat", "skill_pc_hat_vs_std",
        "style_aggr", "zone_minus_cmd", "psr_low",
        "fail_rev_share", "fail_mid_share",
        "is_new_pitcher", "prev_missing",
    ],
    "BATTER_HISTORY": [
        "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
        "asof_batter_success_rate_shr", "asof_batter_middle_rate_shr",
        "std_batter_n", "std_asof_batter_success_rate",
        "std_asof_batter_middle_rate",
        "std_asof_batter_success_rate_delta",
        "std_asof_batter_middle_rate_delta",
        "te_batter_ratio", "te_batter_rate", "te_batter_n",
        "is_new_batter", "dom_bat_empty", "dom_bat_runner",
    ],
    # who is facing whom, and the pitcher's record against that handedness
    "MATCHUP": [
        "pitcher_hand", "batter_hand", "matchup_same_hand", "dom_same_hand",
        "dom_same_ball", "dom_same_rev",
        "te_pitcher_batter_hand_ratio", "te_pitcher_batter_hand_rate",
        "te_pitcher_batter_hand_n", "te_pitcher_batter_hand_ratio_dev",
    ],
    "COUNT": [
        "balls_before", "strikes_before", "is_3ball", "is_2strike",
        "count_adv",
        "te_pitcher_balls_before_strikes_before_ratio",
        "te_pitcher_balls_before_strikes_before_rate",
        "te_pitcher_balls_before_strikes_before_n",
        "te_pitcher_balls_before_strikes_before_ratio_dev",
    ],
    "GAME_STATE": [
        "inning", "top_bottom", "outs_before", "run_top_before",
        "run_bot_before", "run_total_before", "score_diff_home",
        "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b",
        "runner_on_3b", "num_runners_on", "base_state",
        "home_win_expectancy", "away_win_expectancy", "li",
        "te_pitcher_inning_bucket_ratio", "te_pitcher_inning_bucket_rate",
        "te_pitcher_inning_bucket_n", "te_pitcher_inning_bucket_ratio_dev",
    ],
    "CALENDAR_ID": [
        "season", "game_month", "game_dayofweek", "month_cat", "dow_cat",
        "is_monday", "season_progress", "game_type",
        "pitcher_team_id", "batter_team_id",
    ],
}

BLOCK_OF: dict[str, str] = {f: b for b, fs in BLOCKS.items() for f in fs}
ORDER = list(BLOCKS)


def validate(features):
    features = list(features)
    missing = [f for f in features if f not in BLOCK_OF]
    extra = sorted(set(BLOCK_OF) - set(features))
    if missing or extra:
        raise SystemExit(f"block map does not match the model\n"
                         f"  unmapped : {missing}\n  absent   : {extra}")
    return True


def closure_report(frame, features, cat_cols, sample=60_000, seed=0,
                   tol=0.9999):
    """Is every block closed? Regress each member on all *non*-members.

    A member with R^2 at the tolerance is reconstructible from columns that a
    block permutation leaves untouched -- so permuting that block would not
    remove it, and the block is not a unit of information.
    """
    import numpy as np
    import pandas as pd

    num = [c for c in features if c not in cat_cols
           and pd.api.types.is_numeric_dtype(frame[c])]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frame), size=min(sample, len(frame)), replace=False)
    M = frame.loc[idx, num].astype(np.float64).to_numpy()
    M = np.where(np.isfinite(M), M, np.nanmedian(M, axis=0))
    mu, sd = M.mean(0), M.std(0)
    sd[sd == 0] = 1.0
    Z = (M - mu) / sd
    n = len(Z)
    leaks = []
    for b in ORDER:
        inside = [i for i, c in enumerate(num) if BLOCK_OF[c] == b]
        outside = [i for i, c in enumerate(num) if BLOCK_OF[c] != b]
        if not inside or not outside:
            continue
        A = Z[:, outside]
        G = A.T @ A + 1e-8 * n * np.eye(len(outside))
        B = np.linalg.solve(G, A.T @ Z[:, inside])
        res = Z[:, inside] - A @ B
        ss = (Z[:, inside] ** 2).sum(0)
        r2 = np.where(ss > 0, 1.0 - (res ** 2).sum(0) / np.maximum(ss, 1e-30),
                      np.nan)
        for j, i in enumerate(inside):
            if np.isfinite(r2[j]) and r2[j] >= tol:
                leaks.append((b, num[i], float(r2[j])))
    return leaks


if __name__ == "__main__":
    import sys
    import joblib
    import pandas as pd

    sys.path.insert(0, "src")
    import fpipe

    pkl = sys.argv[1] if len(sys.argv) > 1 else "model/cat_B1S_base_s3.pkl"
    season = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    pack = joblib.load(pkl)
    validate(pack["features"])
    header = list(pd.read_csv("data/test.csv", nrows=0,
                              encoding="utf-8-sig").columns)
    full = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                       usecols=header + ["control_success"])
    fr = fpipe.transform(full[full["season"] == season].reset_index(drop=True),
                         pack["fpipe"])
    print(f"{len(pack['features'])} features -> {len(BLOCKS)} blocks")
    for b in ORDER:
        print(f"  {b:16} {len(BLOCKS[b]):3d}")
    leaks = closure_report(fr, pack["features"], pack["cat_cols"])
    if leaks:
        print(f"\nNOT CLOSED -- {len(leaks)} column(s) survive their own "
              f"block's permutation:")
        for b, c, r2 in leaks:
            print(f"  {b:16} {c:44} R^2 from outside = {r2:.6f}")
    else:
        print("\nCLOSED -- no column is reconstructible from outside its block")
