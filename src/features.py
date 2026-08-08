"""피처 v2 — cold-start shrinkage + 파생 피처.

학습과 추론(script.py)에서 동일하게 사용해야 함 (제출 zip에 이 파일 포함).
priors는 train에서 계산해 모델 pkl에 저장하고 추론 시 재사용.
"""

import numpy as np
import pandas as pd

# (rate 컬럼, 대응 표본수 컬럼)
SHRINK = [
    ("asof_pitcher_success_rate", "asof_pitcher_n"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("asof_pitcher_ball_rate", "asof_pitcher_n"),
    ("asof_pitcher_strike_rate", "asof_pitcher_n"),
    ("asof_batter_success_rate", "asof_batter_n"),
    ("asof_batter_middle_rate", "asof_batter_n"),
    ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"),
]
K = 200  # shrinkage 강도 (n=200에서 절반 가중)


def compute_priors(df):
    """rate 피처들의 리그 평균 — 반드시 학습 데이터로만 계산.

    피처 선택으로 일부 컬럼이 빠질 수 있으므로 존재하는 것만 계산한다.
    """
    return {c: float(df[c].mean()) for c, n in SHRINK
            if c in df.columns and n in df.columns}


def add_features(df, priors, k=K, v6=False):
    """파생 피처 추가. df를 복사해 반환 (원본 유지). 추가된 컬럼명 리스트도 반환.

    제거된 컬럼에 의존하는 파생은 건너뛴다 (--drop-unstable 등과 호환).
    v6=True는 실패구성비·pa_depth 추가 — **실측 -26이라 기본 비활성**(E51).
    """
    out = df.copy()
    new = []
    has = out.columns.__contains__

    # 1) cold-start shrinkage: (n·rate + k·prior) / (n + k)
    for c, ncol in SHRINK:
        if c not in priors or not has(c) or not has(ncol):
            continue
        n = out[ncol].fillna(0)
        r = out[c].fillna(priors[c])
        out[f"{c}_shr"] = (n * r + k * priors[c]) / (n + k)
        new.append(f"{c}_shr")

    # 2) 이력 부족/결측 지시자
    if has("asof_pitcher_n"):
        out["is_new_pitcher"] = (out["asof_pitcher_n"].fillna(0) < 50).astype(np.int8)
        new.append("is_new_pitcher")
    if has("asof_batter_n"):
        out["is_new_batter"] = (out["asof_batter_n"].fillna(0) < 50).astype(np.int8)
        new.append("is_new_batter")
    if has("asof_pitcher_prev1_game_success_rate"):
        out["prev_missing"] = out["asof_pitcher_prev1_game_success_rate"].isna().astype(np.int8)
        new.append("prev_missing")

    # 3) 폼 변화: 최근 성적 − 누적 성적
    if has("asof_pitcher_success_rate"):
        for src, name in [("asof_pitcher_prev5_game_success_rate", "form_delta5"),
                          ("asof_pitcher_prev1_game_success_rate", "form_delta1")]:
            if has(src):
                out[name] = out[src] - out["asof_pitcher_success_rate"]
                new.append(name)

    # 3-b) 실패 모드 구성비 — 대회의 제구 실패 정의 3가지를 직접 반영
    #   ① 존 한가운데 ② 존에서 크게 벗어남 ③ 포수 요구 반대 방향
    #   실측(2024, 표본 200+ 투수): reverse가 success와 상관 -0.848로 **지배적 실패 모드**,
    #   단독 신호도 212.2(success_rate 240.8에 근접). middle과 ball은 상관 -0.205로
    #   서로 반대 축 = '승부형 vs 유인형' 스타일 축이다.
    #   차이/비율은 축 정렬 분할만 하는 트리가 만들기 어려우므로 명시적으로 준다.
    if has("asof_pitcher_middle_rate") and has("asof_pitcher_ball_rate"):
        mid = out["asof_pitcher_middle_rate"]
        ball = out["asof_pitcher_ball_rate"]
        out["style_aggr"] = mid - ball          # +면 승부형(한가운데), −면 유인형(빠지는 공)
        new.append("style_aggr")
        if has("asof_pitcher_reverse_rate"):
            rev = out["asof_pitcher_reverse_rate"]
            fail = mid + ball + rev
            out["fail_rev_share"] = rev / fail.replace(0, np.nan)   # 방향 실수 비중
            out["fail_mid_share"] = mid / fail.replace(0, np.nan)
            new += ["fail_rev_share", "fail_mid_share"]
    if has("asof_pitcher_strike_rate") and has("asof_pitcher_success_rate"):
        # 존 안에 넣긴 하는데 원하는 곳엔 못 넣는 정도 = 순수 커맨드 결핍
        out["zone_minus_cmd"] = (out["asof_pitcher_strike_rate"]
                                 - out["asof_pitcher_success_rate"])
        new.append("zone_minus_cmd")

    # 4) 카운트/상황 파생
    if has("balls_before"):
        out["is_3ball"] = (out["balls_before"] == 3).astype(np.int8)
        new.append("is_3ball")
    if has("strikes_before"):
        out["is_2strike"] = (out["strikes_before"] == 2).astype(np.int8)
        new.append("is_2strike")
    if has("strikes_before") and has("balls_before"):
        out["count_adv"] = out["strikes_before"] - out["balls_before"]
        new.append("count_adv")
    if has("pitcher_hand") and has("batter_hand"):
        out["matchup_same_hand"] = (out["pitcher_hand"] == out["batter_hand"]).astype(np.int8)
        new.append("matchup_same_hand")

    # 5) v3: 오차 분석(eda2)에서 발견된 미보정 세그먼트 직접 노출
    #    - 월요일 경기(휴식일 특수 경기) 성공률 급락 미반영 → dow 범주화
    #    - 시즌 내 후반부 drift(7~9월 bias) → month 범주화 + 진행도
    #    - 제구 최하위 투수 구간 과대예측 → 저성공률 플래그
    if has("game_month"):
        out["month_cat"] = out["game_month"].astype(str)
        out["season_progress"] = (out["game_month"] - 3).clip(lower=0)
        new += ["month_cat", "season_progress"]
    if has("game_dayofweek"):
        out["dow_cat"] = out["game_dayofweek"].astype(str)
        out["is_monday"] = (out["game_dayofweek"] == 0).astype(np.int8)
        new += ["dow_cat", "is_monday"]
    if has("asof_pitcher_success_rate"):
        out["psr_low"] = (out["asof_pitcher_success_rate"].fillna(0.5) < 0.45).astype(np.int8)
        new.append("psr_low")

    # 6) v6: 실패 '방식' 구성비 (tools/brainstorm.py A 근거)
    #    같은 성공률이어도 볼로 실패하는 투수 vs 반대방향으로 실패하는 투수는 다르다.
    #    mix_ball +0.066 / mix_reverse -0.070 — 최상위 단일 피처(0.084)에 필적.
    #    비율은 트리가 분할로 근사하기 어려워 명시 가치가 있다.
    fm = ["asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
          "asof_pitcher_reverse_rate"]
    if v6 and all(has(c) for c in fm):
        tot = out[fm].sum(axis=1).replace(0, np.nan)
        for c in fm:
            name = f"mix_{c.split('_')[-2]}"
            out[name] = out[c] / tot
            new.append(name)
        out["fail_total"] = tot
        new.append("fail_total")

    # 7) v6: 타석 진행 깊이 (balls+strikes = 이 타석 최소 투구수)
    if v6 and has("balls_before") and has("strikes_before"):
        out["pa_depth"] = out["balls_before"] + out["strikes_before"]
        new.append("pa_depth")

    return out, new


# add_features가 만드는 컬럼 중 범주형으로 취급할 것들
NEW_CAT = ["month_cat", "dow_cat"]


def add_abs_regime(df):
    """ABS(자동 볼판정) 측정 체제 플래그.

    근거(tools/kbo_calendar.py 실측):
      F리그 2022 0.7087 → 2023 0.4729 (-0.236)   ← 퓨처스 ABS 시범 도입
      R리그 2023 0.5031 → 2024 0.4897 (-0.013)   ← 1군 ABS 도입
    타깃이 존 위치 기반이므로 이건 선수 실력이 아니라 **측정 체계의 변화**다.

    핵심: 이 체제 구분은 (game_type=='F' AND season>=2023) OR (season>=2024) 라는
    **이접(disjunction)** 이라, 트리가 축 정렬 분할로 표현하기 어렵다. 명시 가치가 있다.
    평가(2025)는 두 리그 모두 ABS 체제 → 1.
    """
    out = df.copy()
    is_f = out["game_type"].astype(str) == "F"
    out["is_abs"] = (((is_f) & (out["season"] >= 2023))
                     | (out["season"] >= 2024)).astype(np.int8)
    # 체제 내 경과 연차 (0=이전, 1=1년차, 2=2년차…)
    yrs = np.where(is_f, out["season"] - 2023 + 1, out["season"] - 2024 + 1)
    out["abs_year"] = np.clip(yrs, 0, None).astype(np.int8)
    return out, ["is_abs", "abs_year"]


# ---------- v5: 시즌 상대화 (drift 흡수) ----------
# 근거: CatBoost 상호작용 중요도 상위 25쌍 중 12쌍이 season×asof_* (tools/interaction_scan.py).
# 모델이 매 시즌 지표를 재보정하는 데 용량을 쓰고 있음 → 상대값을 직접 주면 그 부담이 사라지고,
# 2025(미학습 시즌)에도 "리그 대비 상대 위치"는 그대로 이전됨.
REL_COLS = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
            "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
            "asof_batter_success_rate", "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate"]


def compute_season_means(df, cols=REL_COLS):
    """시즌별 리그 평균 — 학습 데이터로만 계산. {col: {season: mean}}"""
    return {c: df.groupby("season")[c].mean().to_dict() for c in cols}


def add_season_relative(df, season_means, fallback_trend=True):
    """각 asof 지표를 그 시즌 리그 평균 대비 상대값으로. 미학습 시즌(2025)은 추세 외삽.

    ⚠️ 평가 데이터의 다른 행을 쓰지 않는다 — 시즌 평균은 전부 train에서 사전 계산된 상수.
    """
    out = df.copy()
    new = []
    for c, per_season in season_means.items():
        seasons = sorted(per_season)
        if fallback_trend and len(seasons) >= 2:
            coef = np.polyfit(seasons, [per_season[s] for s in seasons], 1)
        else:
            coef = None
        def lookup(s):
            if s in per_season:
                return per_season[s]
            return float(np.polyval(coef, s)) if coef is not None else np.nan
        base = out["season"].map(lookup)
        out[f"{c}_rel"] = out[c] - base
        new.append(f"{c}_rel")
    # 투수 대 타자 상대 실력 (상호작용 0.306)
    if {"asof_pitcher_success_rate_rel", "asof_batter_success_rate_rel"} <= set(out.columns):
        out["pb_quality_gap"] = (out["asof_pitcher_success_rate_rel"]
                                 - out["asof_batter_success_rate_rel"])
        new.append("pb_quality_gap")
    return out, new


# ---------- v4: 엔티티×상황 타깃 통계 (train 이력 lookup) ----------
# 검증 시엔 2019~2023로만 계산(2024 누수 방지), 제출 refit 시엔 전체(2019~2024)로 재계산.
CTX_KEYS = [("pitcher_id", "count_adv"),
            ("pitcher_id", "batter_hand"),
            ("batter_id", "pitcher_hand"),
            ("pitcher_id", "base_empty")]
K_CELL = 100   # 셀 → 엔티티 평균으로 shrink
K_ENT = 300    # 엔티티 → 리그 평균으로 shrink


def build_ctx_stats(hist, target="control_success"):
    """hist: count_adv/base_empty가 추가된 학습 이력 (타깃 포함).

    ⚠️ 이 함수만 단독으로 쓰면 in-sample 누수(E31 실패, BSS 191). 반드시
    build_ctx_stats_expanding()으로 시즌별 과거 전용 테이블을 만들어 쓸 것.
    """
    league = float(hist[target].mean())
    tables = {"__league__": league}
    for a, b in CTX_KEYS:
        ent = hist.groupby(a)[target].agg(["mean", "size"])
        ent_shr = (ent["size"] * ent["mean"] + K_ENT * league) / (ent["size"] + K_ENT)
        cell = hist.groupby([a, b])[target].agg(["mean", "size"])
        prior = ent_shr.reindex(cell.index.get_level_values(0)).to_numpy()
        cell_shr = (cell["size"].to_numpy() * cell["mean"].to_numpy()
                    + K_CELL * prior) / (cell["size"].to_numpy() + K_CELL)
        tables[(a, b)] = pd.Series(cell_shr, index=cell.index)
        tables[("__ent__", a)] = ent_shr
    return tables


def build_ctx_stats_expanding(df, target="control_success"):
    """시즌 확장(expanding) 방식 — 시즌 S의 통계는 시즌 < S 데이터로만 계산.

    누수 없음 + 평가(2025)가 2019~2024 전체로 계산되는 것과 동일한 구조.
    반환: {season: tables} — 최소 시즌은 테이블 없음(리그 평균 fallback).
    """
    per_season = {}
    seasons = sorted(df["season"].unique())
    for s in seasons[1:]:
        prior = df[df["season"] < s]
        per_season[s] = build_ctx_stats(prior, target)
    return per_season


def add_ctx_features_expanding(df, per_season, global_tables=None):
    """행의 season에 맞는 테이블로 lookup. 테이블 없는 시즌은 리그 평균.

    global_tables: 평가(2025)용 — 전체 시즌으로 만든 단일 테이블.
    """
    out = df.copy()
    if "base_empty" not in out.columns:
        out["base_empty"] = (out["base_state"] == "___").astype(np.int8)
    new = [f"ctx_{a}_{b}" for a, b in CTX_KEYS]
    for c in new:
        out[c] = np.nan

    for s, grp in out.groupby("season"):
        tables = per_season.get(s, global_tables)
        if tables is None:  # 최초 시즌 — 과거가 없음
            continue
        sub, _ = add_ctx_features(grp, tables)
        out.loc[grp.index, new] = sub[new].to_numpy()

    # 과거가 없던 행은 전역 리그 평균으로 채움
    any_tables = global_tables or next(iter(per_season.values()), None)
    if any_tables is not None:
        out[new] = out[new].fillna(any_tables["__league__"])
    return out, new


def add_ctx_features(df, tables):
    """lookup 병합. 미지 엔티티/셀은 상위 prior로 fallback. 반환: (df, 새 컬럼)"""
    out = df.copy()
    if "base_empty" not in out.columns:
        out["base_empty"] = (out["base_state"] == "___").astype(np.int8)
    league = tables["__league__"]
    new = []
    for a, b in CTX_KEYS:
        col = f"ctx_{a}_{b}"
        idx = pd.MultiIndex.from_arrays([out[a], out[b]])
        v = tables[(a, b)].reindex(idx).to_numpy()
        ent = tables[("__ent__", a)].reindex(out[a]).to_numpy()
        v = np.where(np.isnan(v), ent, v)
        out[col] = np.where(np.isnan(v), league, v)
        new.append(col)
    return out, new
