"""학습된 투수 실력 추정기 (E116) — 손으로 정한 수축을 회귀로 대체.

동기 (tools/skill_estimator.py 실측, 2019~23 학습 -> 2024 평가 184,639행):
  '그 투수의 남은 시즌 성공률'을 얼마나 맞히는가

    상수                        MSE 0.00340   설명력  0.0%
    통산 rate                       0.00277        18.6%
    우리 std (k=80 손으로 정함)        0.00213        37.3%
    **학습된 선형결합**               0.00140        59.0%
    학습된 GBDT                     0.00182        46.5%

두 가지가 드러났다.
  ① 학습된 결합이 우리 수축보다 **21.7%p 더 설명한다**. k=80 은 최적이 아니었다.
  ② **선형이 GBDT 를 크게 이긴다**(59.0 vs 46.5). 최적 수축은 본질적으로
     가중평균 = 매끄러운 선형 결합인데 트리는 계단으로만 근사한다.
     즉 본 모델(CatBoost)도 같은 입력을 다 갖고 있으면서 최적 결합을 못 만든다.

그래서 실력 추정만 **별도 선형 회귀**로 풀어 그 예측을 피처로 넣는다.
본 모델은 남은 어려운 부분(상황·매치업)에 집중하면 된다.

누수 차단: TE 와 같은 규율. **시즌 S 행의 추정기는 시즌 < S 자료로만 적합**한다.
  2025 행에는 2019~2024 전체로 적합한 계수를 쓴다.
  타깃(남은 시즌 성공률)은 학습 시즌 안에서만 만들어지고, 각 행에 붙는 것은
  그 행 자신의 asof/std 컬럼으로 계산한 예측값뿐이므로 행 독립 원칙을 지킨다.
"""

import numpy as np
import pandas as pd

TARGET = "control_success"

# 추정기 입력 - 전부 '그 투구 이전'에 알 수 있는 값이다.
FEAT = ["asof_pitcher_success_rate", "std_asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_batter_success_rate"]
# 표본수는 로그로 (수축 강도가 표본수에 비선형으로 걸리므로)
NCOL = ["asof_pitcher_n", "std_pitcher_n"]
MINF = 150          # 타깃이 안정적이려면 남은 투구가 이만큼은 있어야 한다


# 축별 설정. TE 가 그 축을 '과거 빈도'로 잡고 있으므로 회귀에 물려주면
# 매끄러운 타깃의 이점을 얹을 수 있다.
#   feat  회귀에 추가로 넣을 컬럼
#   keys  '남은 성적' 타깃을 집계할 그룹 키 (투수·시즌에 덧붙는다)
#   cells 원핫으로 펼칠 (컬럼, 값) 조합 — 셀마다 기울기가 다를 수 있다
AXES = {
    # E117: 투수x볼카운트
    "count": {
        "feat": ["te_pitcher_balls_before_strikes_before_ratio",
                 "te_pitcher_balls_before_strikes_before_ratio_dev",
                 "te_pitcher_balls_before_strikes_before_n",
                 "balls_before", "strikes_before"],
        "keys": ["balls_before", "strikes_before"],
        "cells": [("balls_before", (0, 1, 2, 3)),
                  ("strikes_before", (0, 1, 2))],
    },
    # E123: 투수x타자손. tools/honest_ceiling.py 교차적합 잔차에서 이 축이
    # k=100 수축 기준 +31 로 **남은 구조가 가장 큰 축**이었다.
    "hand": {
        "feat": ["te_pitcher_batter_hand_ratio",
                 "te_pitcher_batter_hand_ratio_dev",
                 "te_pitcher_batter_hand_n"],
        "keys": ["batter_hand"],
        "cells": [("batter_hand", ("L", "R"))],
    },
}
PC_FEAT = AXES["count"]["feat"]        # 옛 이름 (호환)


def _design(df, med, axis=""):
    """설계행렬. 표본수는 log1p, 결측은 학습 중앙값, 절편 포함.

    **폭은 axis 로만 정해진다.** 컬럼이 없어도 자리를 비우지 않고 중앙값으로
    채운다 — 상황에 따라 폭이 변하면 학습/추론 설계행렬이 어긋난다 (실제로
    41 vs 44 로 터진 적이 있다).
    """
    cols = []
    for c in FEAT:
        v = df[c].to_numpy(np.float64) if c in df.columns             else np.full(len(df), np.nan)
        cols.append(np.where(np.isfinite(v), v, med.get(c, 0.5)))
    for c in NCOL:
        v = df[c].to_numpy(np.float64) if c in df.columns             else np.full(len(df), 0.0)
        v = np.where(np.isfinite(v), v, 0.0)
        cols.append(np.log1p(np.maximum(v, 0.0)))
        # 수축은 n/(n+k) 형태로 들어가므로 그 모양도 직접 준다
        cols.append(v / (v + 80.0))
    spec = AXES.get(axis)
    if spec:
        for c in spec["feat"]:
            v = (df[c].to_numpy(np.float64) if c in df.columns
                 else np.full(len(df), np.nan))
            cols.append(np.where(np.isfinite(v), v, med.get(c, 0.0)))
        # 투수의 기본 수준 — 셀마다 기울기가 다를 수 있어 교차항으로 넣는다
        base = df.get("std_asof_pitcher_success_rate")
        base = (np.nan_to_num(base.to_numpy(np.float64), nan=0.5)
                if base is not None else np.full(len(df), 0.5))
        for m in _cells(df, spec["cells"]):
            cols.append(m)
            cols.append(m * base)
    X = np.column_stack(cols)
    return np.column_stack([np.ones(len(X)), X])


def _cells(df, cells):
    """(컬럼, 값들) 목록의 데카르트곱을 원핫 마스크로 펼친다."""
    masks = [np.ones(len(df), np.float64)]
    for col, vals in cells:
        v = df[col].to_numpy() if col in df.columns else np.full(len(df), None)
        masks = [m * (v == x).astype(np.float64) for m in masks for x in vals]
    return masks


def _fit(df, med, ridge=1.0, axis=""):
    """남은 성적을 타깃으로 능형회귀."""
    d = df[df["_futn"] >= MINF]
    if len(d) < 5000:
        return None
    X = _design(d, med, axis)
    y = d["_futr"].to_numpy(np.float64)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    A[0, 0] -= ridge                      # 절편은 규제하지 않는다
    return np.linalg.solve(A, X.T @ y)


def _future(df, axis=""):
    """그룹 안에서 그 행 **이후** 남은 성공률과 투구수.

    axis 가 비면 (투수, 시즌). 축이 있으면 그 축 컬럼이 키에 더해져
    '그 투수의 **이 상황에서의** 남은 성적'이 타깃이 된다.
    """
    # Order inside a group is time order, and "the rest of the season" is a
    # cumulative sum, so a reordered frame silently changes every target. A
    # stable sort only preserves whatever order the caller happened to pass;
    # sorting on row_id makes the pitch order explicit and independent of it
    # (audit 4.2: "명시적 row_id 정렬"). row_id is not a feature -- load() drops
    # it from the model input -- so this costs nothing but determinism.
    keys = ["pitcher_id", "season"] + AXES.get(axis, {}).get("keys", [])
    if "row_id" in df.columns:
        d = df.sort_values(keys + ["row_id"], kind="stable")
    else:
        d = df.sort_values(keys, kind="stable")
    g = d.groupby(keys, sort=False)
    tot = g[TARGET].transform("sum").to_numpy(np.float64)
    cnt = g[TARGET].transform("size").to_numpy(np.float64)
    cs = g[TARGET].cumsum().to_numpy(np.float64)
    idx = g.cumcount().to_numpy(np.float64)
    fn = cnt - idx - 1
    fr = np.where(fn > 0, (tot - cs) / np.maximum(fn, 1.0), np.nan)
    return pd.Series(fn, index=d.index), pd.Series(fr, index=d.index)


def build(train, axis="", neutral_first=False):
    """시즌별 계수 묶음을 만든다. 시즌 S 계수는 **S 미만 시즌**으로만 적합.

    axis="count" (E117) 타깃 = 그 투수의 이 볼카운트에서의 남은 성공률.
    axis="hand"  (E123) 타깃 = 그 투수의 이 타자손 상대 남은 성공률.
                 교차적합 잔차에서 남은 구조가 가장 컸던 축이다(+31, k=100).
    """
    df = train.copy()
    fn, fr = _future(df, axis)
    df["_futn"] = fn.reindex(df.index)
    df["_futr"] = fr.reindex(df.index)
    extra = AXES.get(axis, {}).get("feat", [])
    med = {c: float(df[c].median()) for c in FEAT + extra if c in df.columns}
    seasons = sorted(df["season"].unique())
    coef = {}
    for i, s in enumerate(seasons):
        b = _fit(df[df["season"] < s], med, axis=axis) if i > 0 else None
        if b is None:                      # 첫 시즌은 과거가 없다
            # Legacy: fall back to coefficients fit on the **whole** frame, so
            # 2019 training rows receive a regression built from future-season
            # targets. `neutral_first` says the honest thing instead -- there is
            # no past, so the estimate is missing and CatBoost treats it as such.
            b = "neutral" if neutral_first else _fit(df, med, axis=axis)
        coef[s] = b
    coef[max(seasons) + 1] = _fit(df, med, axis=axis)   # 2025 행용
    return {"coef": coef, "med": med, "last": max(seasons) + 1, "axis": axis}


def add(df, pack):
    """실력 추정치를 피처로 붙인다."""
    if pack is None:
        return df, []
    # 옛 pkl 은 per_count 불리언을 들고 있다
    axis = pack.get("axis", "count" if pack.get("per_count") else "")
    med, coef = pack["med"], pack["coef"]
    X = _design(df, med, axis)
    se = df["season"].to_numpy()
    out = np.full(len(df), np.nan)
    for s in np.unique(se):
        b = coef.get(int(s))
        if isinstance(b, str):             # "neutral" -- no past season exists
            continue                       # leave NaN; CatBoost reads it natively
        if b is None:                      # 학습에 없던 시즌 = 마지막 계수
            b = coef[pack["last"]]
        m = se == s
        out[m] = X[m] @ b
    nm = f"skill_{axis}_hat" if axis else "skill_hat"
    if axis == "count":
        nm = "skill_pc_hat"                # 기존 모델과 컬럼명 호환
    df[nm] = np.clip(out, 0.0, 1.0)
    # 추정치와 원시 관측의 차이 = 수축이 얼마나 끌어당겼는가
    if "std_asof_pitcher_success_rate" in df.columns:
        df[nm + "_vs_std"] = df[nm] - df["std_asof_pitcher_success_rate"]
        return df, [nm, nm + "_vs_std"]
    return df, [nm]
