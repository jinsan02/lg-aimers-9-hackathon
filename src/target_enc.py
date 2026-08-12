"""시즌 단위 expanding 타깃 인코딩 - 누수 없이 '투수의 현재 수준'을 추정한다.

동기 (tools/signal_audit.py 오라클 측정):
  2024 실제 그룹 평균을 알 때의 천장
    투수         990.8
    투수x타자손  1425.5
    투수x카운트  2740.9
  반면 공식 asof_pitcher_success_rate 단독은 240.8, 우리 모델 전체가 784.5.
  => 병목은 모델이 아니라 **투수 수준 추정의 질**이다.

핵심 설계 - '상대 실력비(ratio)':
  asof_*는 여러 시즌(ABS 전/후)이 섞인 raw 비율이라 레짐 드리프트에 오염된다.
  대신 그 투수가 던진 각 투구를 **그 시즌 리그평균**으로 기대값 처리해서
      ratio = (성공수 + k) / (기대성공수 + k)
  를 쓰면 리그 수준이 변해도 '리그 대비 몇 배'는 이전된다.
  현재 시즌의 리그 수준은 모델이 season 피처로 따로 학습한다.

누수 차단 - **시즌 S 행의 피처는 시즌 < S 자료만 사용**한다.
  학습 2024행 -> 2019~2023 / 평가 2025행 -> 2019~2024.
  둘 다 "직전까지의 전 시즌"이라 학습·추론의 정의가 일치한다.
  (투구 단위 expanding을 쓰면 2025행만 시즌 내 이력이 없어 정의가 어긋난다.)
"""

import numpy as np
import pandas as pd

TARGET = "control_success"
K_DEFAULT = 50


def _grid(g, keys, seasons):
    """(키 x 시즌) 전 격자로 채워 누적합이 시즌 경계를 건너뛰지 않게 한다."""
    idx = pd.MultiIndex.from_product(
        [g.index.get_level_values(k).unique() for k in keys] + [seasons],
        names=keys + ["season"])
    return g.reindex(idx, fill_value=0.0).sort_index()


def build_te(df, keys, k=K_DEFAULT, half_life=0.0, prefix=None, strat=True,
             fit_mask=None):
    """keys 조합에 대한 시즌 expanding 타깃 인코딩 표를 만든다.

    반환: (키..., season) -> [ratio, rate, n] 을 담은 DataFrame.
      season=S 행은 **S 미만 시즌만** 집계한 값이다. max(season)+1 행도 만들어
      제출 추론(2025)에서 바로 쓸 수 있게 한다.

    half_life > 0 이면 시즌 반감기 가중(최근 시즌을 더 크게)을 적용한다.

    strat=True 면 기대값을 **투수를 뺀 나머지 키 × 시즌** 평균으로 잡는다.
      예: keys=[pitcher_id, balls, strikes] 이면 기대값은 (시즌, 볼, 스트라이크)
      평균이다. 이렇게 해야 ratio에서 '이 카운트가 원래 쉽다' 성분이 빠지고
      **'이 투수가 이 카운트에서 남들보다 얼마나 나은가'** 라는 순수 상호작용만 남는다.
      strat=False 면 시즌 전체 평균을 쓴다(구버전 동작).
    """
    seasons = sorted(df["season"].unique())
    horizon = seasons + [max(seasons) + 1]
    # The cumulative count/sum body is season-expanding with shift(1), so no
    # row sees its own season. This scalar was the exception: computed over the
    # whole frame it carries the validation season's target mean into every
    # shrunk rate. `fit_mask` restricts it to the stage's fit partition.
    prior = float(df.loc[fit_mask, TARGET].mean() if fit_mask is not None
                  else df[TARGET].mean())

    rest = [c for c in keys if c not in ("pitcher_id", "batter_id")]
    if strat and rest:
        base_keys = ["season"] + rest
        w = df.groupby(base_keys)[TARGET].transform("mean").to_numpy()
    else:
        lm = df.groupby("season")[TARGET].mean()      # 시즌 리그평균
        w = df["season"].map(lm).to_numpy()           # 각 투구의 기대 성공확률
    tmp = pd.DataFrame({c: df[c].to_numpy() for c in keys})
    tmp["season"] = df["season"].to_numpy()
    tmp["_n"] = 1.0
    tmp["_s"] = df[TARGET].to_numpy().astype(float)
    tmp["_e"] = w

    g = tmp.groupby(keys + ["season"])[["_n", "_s", "_e"]].sum()
    g = _grid(g, keys, horizon)

    if half_life > 0:                                 # 최근 시즌 가중
        decay = 0.5 ** (1.0 / half_life)
        out = []
        for _, sub in g.groupby(level=keys, sort=False):
            v = sub.to_numpy()
            acc = np.zeros(3)
            rows = np.empty_like(v)
            for i in range(len(v)):
                rows[i] = acc                         # **직전 시즌까지**
                acc = acc * decay + v[i]
            out.append(rows)
        cum = np.concatenate(out)
    else:
        cum = (g.groupby(level=keys, sort=False).cumsum()
               .groupby(level=keys, sort=False).shift(1).fillna(0.0).to_numpy())

    n, s, e = cum[:, 0], cum[:, 1], cum[:, 2]
    res = pd.DataFrame(index=g.index)
    # ratio: 리그 대비 배수. 사전분포 1.0 으로 수축.
    res["ratio"] = (s + k) / (e + k)
    # rate: 생 비율(대조군). 사전분포는 전체 평균.
    res["rate"] = (s + k * prior) / (n + k)
    res["n"] = n
    res.loc[n == 0, ["ratio", "rate"]] = np.nan       # 표본 없음 = 결측(cold-start)
    p = prefix or "te_" + "_".join(c.replace("_id", "") for c in keys)
    res.columns = [f"{p}_{c}" for c in res.columns]
    return res.reset_index()


# 사용할 키 조합 (신호감사 오라클 천장 순)
# 야구 의미: 카운트에 따라 승부/유인 전략이 바뀌고, 주자가 있으면 슬라이드스텝·
# 한가운데 회피로 제구 양상이 달라진다. 좌우 스플릿, 이닝(피로/역할)도 마찬가지.
SPECS = {
    "p":    ["pitcher_id"],
    "pc":   ["pitcher_id", "balls_before", "strikes_before"],
    "ph":   ["pitcher_id", "batter_hand"],
    "b":    ["batter_id"],
    "pi":   ["pitcher_id", "inning_bucket"],
    # Batter performance against the opposing pitching team.  The season
    # expanding table only uses seasons < S, so this is available before the
    # pitch and remains row independent at inference time.
    "bo":   ["batter_id", "pitcher_team_id"],
    "pb":   ["pitcher_id", "batter_id"],
    "pbs":  ["pitcher_id", "base_state"],          # 주자 상황별 투구 변화
    "ps":   ["pitcher_id", "strikes_before"],      # 2스트라이크 승부 성향 (pc보다 조밀)
    "pg":   ["pitcher_id", "game_type"],           # 1군/퓨처스 레벨차
    "bc":   ["batter_id", "balls_before", "strikes_before"],
    "bh":   ["batter_id", "pitcher_hand"],         # 타자 좌우 스플릿
    # 3중 상호작용 - 신호감사 오라클이 투수x카운트 2740.9 로 가장 높았고,
    # 좌우까지 쪼개면 이론 천장은 더 올라간다. 그룹 391x12x2 = 약 9.4천개로
    # 시즌 누적이면 그룹당 표본이 충분하다 (k=50 수축이 꼬리를 눌러준다).
    "pchh": ["pitcher_id", "balls_before", "strikes_before", "batter_hand"],
    "pcb":  ["pitcher_id", "balls_before", "strikes_before", "base_state"],
    # Label-free q-bin keys are created by fpipe before this stage.  Their TE
    # uses only seasons < S and is normally hidden from CatBoost itself.
    "rp":   ["pairbin_reverse_prev3"],
    "cp":   ["pairbin_career_prev3"],
}


def add_te(df, tables, dev=False):
    """build_te 결과들을 행에 붙인다. tables: [(keys, table), ...]

    dev=True 면 상호작용 키의 ratio를 **같은 주체의 단독 기준선**으로 나눈 편차를
    추가한다 (투수 키는 te_pitcher_ratio로, 타자 키는 te_batter_ratio로).
    이게 '이 투수가 이 카운트에서 자기 평소보다 얼마나 나은가' 라는 순수 상호작용이다.

    ⚠️ 학습(train_gbdt2)과 추론(script_blend_v5)이 **반드시 이 함수 하나만** 쓴다.
       예전엔 양쪽에 복붙돼 있었고 컬럼명이 어긋나 dev가 통째로 무효였다(08-06 사고).
    """
    new = []
    for keys, tab in tables:
        need = [c for c in keys if c not in df.columns]
        if need:                                       # inning_bucket 등 파생 키
            raise KeyError(f"키 컬럼 없음: {need}")
        df = df.merge(tab, on=keys + ["season"], how="left")
        new += [c for c in tab.columns if c not in keys + ["season"]]
    if dev:
        df, dcols = apply_dev(df, new)
        new = new + dcols
    return df, new


def apply_dev(df, te_cols):
    """상호작용 ratio ÷ 단독 기준선 ratio → 순수 상호작용 성분."""
    base = {"te_pitcher_ratio", "te_batter_ratio"}
    made = []
    for c in [c for c in te_cols if c.endswith("_ratio") and c not in base]:
        if c.startswith("te_pairbin_"):
            continue
        # 그 키가 투수 기준인지 타자 기준인지로 분모를 고른다
        b = "te_batter_ratio" if c.startswith("te_batter") else "te_pitcher_ratio"
        if b not in df.columns:
            continue
        df[c + "_dev"] = df[c] / df[b]
        made.append(c + "_dev")
    return df, made


def add_inning_bucket(df):
    if "inning_bucket" not in df.columns:
        df = df.copy()
        df["inning_bucket"] = np.clip(df["inning"].to_numpy(), 1, 9)
    return df
