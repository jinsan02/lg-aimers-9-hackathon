"""Family map for the 121 champion features.

Every feature gets exactly one `family`, plus two secondary labels that let the
same table be rolled up a different way without re-deciding the primary split:

  axis   what entity the value is about  (pitcher / batter / count / hand /
         inning / pitchmix / situation / calendar / global)
  kind   what shape the value has        (raw / level / delta / n / flag /
         derived / estimate)

Two decisions worth stating, because they are choices and not facts:

**Representation wins over content.** `std_asof_pitcher_fastball_rate` is both a
season-standardised level and a pitchmix column. It is filed under SEASON_STD,
not PITCHMIX, because the sharpest open question -- which of TE level / TE dev /
season_std / skill survives a season boundary -- needs those blocks intact.
The pitchmix roll-up is still one `groupby("axis")` away.

**The listed families are split where the split is the point.** `_shr` columns
are career rates under hand-tuned shrinkage; `std_*` are season rates under the
same idea. Keeping ASOF_SHR separate from RAW_ASOF and STD_DELTA separate from
SEASON_STD turns "level vs deviation" and "career vs season" into contrasts the
table can answer, rather than sums it has already taken.

`season` is a member of RAW_CONTEXT and is constant inside any single-season
slice, so it contributes nothing to a within-season permutation. That is correct
here -- every surface in the transfer map is one season -- but it means
RAW_CONTEXT's number is about the other 23 columns.
"""

from __future__ import annotations

# family -> ordered feature names, exactly as they appear in the champion pack.
FAMILIES: dict[str, list[str]] = {
    "RAW_CONTEXT": [
        "season", "inning", "top_bottom", "game_type", "balls_before",
        "strikes_before", "outs_before", "run_top_before", "run_bot_before",
        "run_total_before", "score_diff_home", "score_diff_pitcher_team",
        "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
        "base_state", "home_win_expectancy", "away_win_expectancy", "li",
        "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    ],
    "CALENDAR": [
        "game_month", "game_dayofweek", "month_cat", "season_progress",
        "dow_cat", "is_monday",
    ],
    "RAW_ASOF": [
        "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_batter_n", "asof_batter_success_rate",
        "asof_batter_middle_rate", "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ],
    "ASOF_SHR": [
        "asof_pitcher_success_rate_shr", "asof_pitcher_reverse_rate_shr",
        "asof_pitcher_middle_rate_shr", "asof_pitcher_ball_rate_shr",
        "asof_pitcher_strike_rate_shr", "asof_batter_success_rate_shr",
        "asof_batter_middle_rate_shr", "asof_pitcher_fastball_rate_shr",
        "asof_pitcher_breaking_rate_shr", "asof_pitcher_offspeed_rate_shr",
    ],
    "RECENT": [
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "form_delta5", "form_delta1",
    ],
    "DOMAIN": [
        "is_new_pitcher", "is_new_batter", "prev_missing", "style_aggr",
        "fail_rev_share", "fail_mid_share", "zone_minus_cmd",
        "matchup_same_hand", "psr_low", "dom_same_hand", "dom_same_ball",
        "dom_same_rev", "dom_bat_empty", "dom_bat_runner",
    ],
    "COUNT_DERIVED": ["is_3ball", "is_2strike", "count_adv"],
    "SEASON_STD": [
        "std_pitcher_n", "std_asof_pitcher_success_rate",
        "std_asof_pitcher_reverse_rate", "std_asof_pitcher_middle_rate",
        "std_asof_pitcher_ball_rate", "std_asof_pitcher_strike_rate",
        "std_batter_n", "std_asof_batter_success_rate",
        "std_asof_batter_middle_rate", "std_pitchmix_n",
        "std_asof_pitcher_fastball_rate", "std_asof_pitcher_breaking_rate",
        "std_asof_pitcher_offspeed_rate",
    ],
    "STD_DELTA": [
        "std_asof_pitcher_success_rate_delta",
        "std_asof_pitcher_reverse_rate_delta",
        "std_asof_pitcher_middle_rate_delta",
        "std_asof_pitcher_ball_rate_delta",
        "std_asof_pitcher_strike_rate_delta",
        "std_asof_batter_success_rate_delta",
        "std_asof_batter_middle_rate_delta",
        "std_asof_pitcher_fastball_rate_delta",
        "std_asof_pitcher_breaking_rate_delta",
        "std_asof_pitcher_offspeed_rate_delta",
    ],
    "TE_LEVEL": [
        "te_pitcher_ratio", "te_pitcher_rate", "te_pitcher_n",
        "te_pitcher_balls_before_strikes_before_ratio",
        "te_pitcher_balls_before_strikes_before_rate",
        "te_pitcher_balls_before_strikes_before_n",
        "te_pitcher_batter_hand_ratio", "te_pitcher_batter_hand_rate",
        "te_pitcher_batter_hand_n", "te_batter_ratio", "te_batter_rate",
        "te_batter_n", "te_pitcher_inning_bucket_ratio",
        "te_pitcher_inning_bucket_rate", "te_pitcher_inning_bucket_n",
    ],
    "TE_DEV": [
        "te_pitcher_balls_before_strikes_before_ratio_dev",
        "te_pitcher_batter_hand_ratio_dev",
        "te_pitcher_inning_bucket_ratio_dev",
    ],
    "SKILL": ["skill_pc_hat", "skill_pc_hat_vs_std"],
}

# Secondary labels. Anything unlisted falls back to the rules in `axis_of`.
_AXIS_OVERRIDE = {
    "season": "calendar", "game_month": "calendar", "game_dayofweek": "calendar",
    "month_cat": "calendar", "dow_cat": "calendar", "is_monday": "calendar",
    "season_progress": "calendar",
    "inning": "situation", "top_bottom": "situation", "game_type": "global",
    "balls_before": "count", "strikes_before": "count", "is_3ball": "count",
    "is_2strike": "count", "count_adv": "count",
    "pitcher_hand": "hand", "batter_hand": "hand", "matchup_same_hand": "hand",
    "dom_same_hand": "hand",
    "pitcher_team_id": "team", "batter_team_id": "team",
    "style_aggr": "pitcher", "zone_minus_cmd": "pitcher", "psr_low": "pitcher",
    "fail_rev_share": "pitcher", "fail_mid_share": "pitcher",
    "form_delta1": "pitcher", "form_delta5": "pitcher",
    "prev_missing": "pitcher", "is_new_pitcher": "pitcher",
    "is_new_batter": "batter", "dom_bat_empty": "batter",
    "dom_bat_runner": "batter", "dom_same_ball": "pitcher",
    "dom_same_rev": "pitcher",
    "skill_pc_hat": "count", "skill_pc_hat_vs_std": "count",
}

_SITUATION = {
    "outs_before", "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy", "li",
}


def axis_of(name: str) -> str:
    if name in _AXIS_OVERRIDE:
        return _AXIS_OVERRIDE[name]
    if name in _SITUATION:
        return "situation"
    if "fastball" in name or "breaking" in name or "offspeed" in name \
            or "pitchmix" in name:
        return "pitchmix"
    if "balls_before_strikes_before" in name:
        return "count"
    if "batter_hand" in name:
        return "hand"
    if "inning_bucket" in name:
        return "inning"
    if "batter" in name:
        return "batter"
    if "pitcher" in name:
        return "pitcher"
    return "global"


def kind_of(name: str) -> str:
    if name.endswith("_delta") or name.endswith("_dev"):
        return "delta"
    if name.endswith("_n") or name in ("std_pitcher_n", "std_batter_n",
                                       "std_pitchmix_n"):
        return "n"
    if name.startswith("skill_"):
        return "estimate"
    if name.startswith(("is_", "runner_on", "matchup_same", "dom_", "prev_")):
        return "flag"
    if name.endswith(("_shr", "_ratio", "_rate")) or name.startswith("std_"):
        return "level"
    if name.startswith(("form_", "count_adv", "style_", "zone_", "fail_",
                        "psr_", "season_progress")):
        return "derived"
    return "raw"


#: feature name -> family, built once so lookups cannot disagree with FAMILIES.
FAMILY_OF: dict[str, str] = {
    f: fam for fam, fs in FAMILIES.items() for f in fs
}

ORDER = list(FAMILIES)


def validate(features):
    """Refuse to be used against a feature list this map does not exactly cover.

    A silently-unmapped column would be excluded from every family and would
    make the shares add to less than the whole while still looking complete.
    """
    features = list(features)
    mapped = set(FAMILY_OF)
    missing = [f for f in features if f not in mapped]
    extra = sorted(mapped - set(features))
    dupes = sorted({f for f in FAMILY_OF
                    if sum(f in fs for fs in FAMILIES.values()) > 1})
    if missing or extra or dupes:
        raise SystemExit(
            f"family map does not match the model\n"
            f"  unmapped features : {missing}\n"
            f"  mapped but absent : {extra}\n"
            f"  in two families   : {dupes}")
    return True


def table(features):
    import pandas as pd
    return pd.DataFrame({
        "feature": list(features),
        "family": [FAMILY_OF[f] for f in features],
        "axis": [axis_of(f) for f in features],
        "kind": [kind_of(f) for f in features],
    })


if __name__ == "__main__":
    import sys
    import joblib
    import pandas as pd

    pkl = sys.argv[1] if len(sys.argv) > 1 else "model/cat_B1S_base_s3.pkl"
    feats = joblib.load(pkl)["features"]
    validate(feats)
    t = table(feats)
    print(f"{len(feats)} features, {t.family.nunique()} families "
          f"(map validated against {pkl})\n")
    print(t.groupby("family").size().reindex(ORDER).to_frame("n")
          .assign(share=lambda d: (100 * d.n / len(feats)).round(1))
          .to_string())
    print("\nby axis:")
    print(t.groupby("axis").size().sort_values(ascending=False).to_string())
    print("\nby kind:")
    print(t.groupby("kind").size().sort_values(ascending=False).to_string())
    out = "out/feature_families.csv"
    t.to_csv(out, index=False)
    print(f"\nwrote {out}")
