"""당해 시즌 성적 복원 - `asof_*` 통산 누적을 차분해 '이번 시즌' 성적을 만든다.

발견(E99): `asof_pitcher_n`은 **시즌마다 리셋되지 않고 통산 누적**이다.
  (투수-시즌 1,468건 전수 확인: 리셋 0.000 / 전 시즌 말과 연결 1.000)

따라서 누적 성공 횟수는
    S = asof_pitcher_n x asof_pitcher_success_rate
이고, 그 투수의 **직전 시즌 말 상태 (n0, S0)** 를 알면
    이번 시즌 성적 = (S - S0) / (n - n0)
로 복원된다.

왜 중요한가: 통산 성공률은 ABS 이전 구체제 데이터에 오염돼 있다. 신호감사에서
'투수의 당해 시즌 실제 성공률'을 알 때의 천장이 990.8이었는데, 공식 통산 피처는
그 근처에 못 간다. 차분하면 그 천장에 접근한다.
  실측(2024 검증, 구간매핑 상한): 통산 374.5 -> **시즌내 534.5 (k=30)**

규칙 적합성: 앵커 (n0, S0)는 **train 데이터만으로** 만들고, 나머지는 그 행 자신의
asof 컬럼이다. 평가 데이터의 다른 행을 전혀 참조하지 않으므로 행 독립 원칙을 지킨다.
"""

import numpy as np
import pandas as pd

TARGET = "control_success"

# (엔티티 접두어, 개수 컬럼, 비율 컬럼들)
SPECS = [
    ("pitcher", "asof_pitcher_n",
     ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
      "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
      "asof_pitcher_strike_rate"]),
    ("batter", "asof_batter_n",
     ["asof_batter_success_rate", "asof_batter_middle_rate"]),
    # 구종 배합도 누적이다(별도 표본수 컬럼 사용) - 당해 시즌 배합 변화 (레버 B)
    ("pitchmix", "asof_pitcher_pitchmix_n",
     ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
      "asof_pitcher_offspeed_rate"]),
]

# 엔티티 접두어 -> 그 엔티티의 id 컬럼 (pitchmix는 투수 단위)
ID_OF = {"pitcher": "pitcher_id", "batter": "batter_id",
         "pitchmix": "pitcher_id"}


def build_anchors(df, last_pitch=False, fit_mask=None):
    """각 (엔티티, 시즌)에 대해 **그 시즌 시작 시점**의 통산 상태를 만든다.

    반환: {접두어: DataFrame(id, season, n0, <비율별 S0>)}
      season 은 '이 앵커를 적용할 시즌'이다. 시즌을 건너뛴 선수도 직전 활동
      시즌의 상태가 이어지도록 forward fill 하고, max(season)+1 행도 만들어
      제출 추론(2025)에서 바로 쓰게 한다.
    """
    out = {}
    seasons = sorted(df["season"].unique())
    horizon = seasons + [max(seasons) + 1]
    for pre, ncol, rcols in SPECS:
        idc = ID_OF[pre]
        if ncol not in df.columns:
            continue
        cols = [idc, "season", ncol] + [c for c in rcols if c in df.columns]
        if last_pitch and TARGET in df.columns:
            cols = cols + [TARGET]
        if last_pitch and fit_mask is not None:
            df = df.assign(_anchor_ok=np.asarray(fit_mask, bool))
            cols = cols + ["_anchor_ok"]
        # 각 (선수,시즌)의 **누적 투구수가 최대인 행** = 그 시즌 종료 시점 상태.
        # asof_n 자체가 단조 증가하므로 row_id 없이도 정렬 기준이 된다.
        last = df[cols].sort_values(ncol).groupby([idc, "season"],
                                                  sort=False).tail(1)
        n_last = last[ncol].to_numpy(np.float64)
        rec = {idc: last[idc].to_numpy(), "season": last["season"].to_numpy(),
               "n0": n_last}
        for c in rcols:
            if c in last.columns:
                rec[f"S0_{c}"] = n_last * last[c].fillna(0).to_numpy(np.float64)
        # The stored asof_* on a row are **pre-pitch**, so the season's last row
        # describes the state before its own final pitch: the anchor is one pitch
        # short. Verified on the official train (2026-08-13): next_first_n ==
        # previous_last_asof_n + 1 for 1,468/1,468 pitcher and 1,563/1,563 batter
        # transitions, no exceptions.
        #
        # Only the success rate can be closed exactly -- the last pitch's outcome
        # is the target itself. middle/ball/reverse/strike/pitchmix would need
        # failmode differencing, which recovers 99.85% and is train-only, so they
        # keep the short anchor and get their **own** denominator. Sharing one n0
        # would put a corrected count under an uncorrected numerator, which is a
        # worse error than the off-by-one it fixes.
        if last_pitch and TARGET in last.columns:
            ok = (last["_anchor_ok"].to_numpy(bool)
                  if "_anchor_ok" in last.columns
                  else np.ones(len(last), bool))
            y = last[TARGET].fillna(0).to_numpy(np.float64)
            for c in rcols:
                key = f"S0_{c}"
                if key not in rec:
                    continue
                if c.endswith("_success_rate"):
                    rec[f"n0_{c}"] = np.where(ok, n_last + 1.0, n_last)
                    rec[key] = np.where(ok, rec[key] + y, rec[key])
                else:
                    rec[f"n0_{c}"] = n_last
        tab = pd.DataFrame(rec)
        # (선수 x 전체 시즌) 격자로 펼쳐 앞으로 채운다 -> 시즌 건너뛴 경우 대응
        ids = tab[idc].unique()
        grid = pd.MultiIndex.from_product([ids, horizon], names=[idc, "season"])
        tab = (tab.set_index([idc, "season"]).reindex(grid)
               .groupby(level=0).ffill())
        # season=S 행은 'S 시작 시점' 이어야 하므로 한 시즌 밀어준다
        tab = tab.groupby(level=0).shift(1).reset_index()
        out[pre] = tab
    return out


def build_profile(df):
    """앵커를 연속 차분해 **선수별·시즌별 성적 궤적**을 만든다 (E100).

    앵커는 각 시즌 시작 시점의 통산 상태이므로, 인접 두 앵커를 빼면
    그 사이 한 시즌의 성적이 그대로 나온다.

    왜 필요한가: E99의 '시즌내 성적'은 정시즌이지만 시즌 초에는 표본이 없다.
    반대로 '직전 완료 시즌 성적'은 표본이 충분하고 **ABS 체제에 한정**돼
    통산 성적처럼 구체제에 오염되지 않는다. 둘은 상보적이다.
      2025행 기준: prev1 = 2024 전체(ABS 1년차), prev2 = 2023 전체.
    """
    out = {}
    for pre, ncol, rcols in SPECS:
        idc = ID_OF[pre]
        if pre not in ("pitcher", "batter") or ncol not in df.columns:
            continue
        rc = [c for c in rcols if c in df.columns]
        last = df[[idc, "season", ncol] + rc].sort_values(ncol) \
            .groupby([idc, "season"], sort=False).tail(1).copy()
        for c in rc:
            last[f"S_{c}"] = last[ncol].to_numpy(np.float64) * \
                last[c].fillna(0).to_numpy(np.float64)
        last = last.sort_values([idc, "season"])
        g = last.groupby(idc)
        rec = {idc: last[idc].to_numpy(), "season": last["season"].to_numpy()}
        dn = last[ncol].to_numpy(np.float64) - g[ncol].shift(1).fillna(0).to_numpy()
        rec["yr_n"] = dn
        for c in rc:
            ds = last[f"S_{c}"].to_numpy() - g[f"S_{c}"].shift(1).fillna(0).to_numpy()
            rec[f"yr_{c}"] = ds
        out[pre] = pd.DataFrame(rec)
    return out


def add_profile(df, profiles, k=60.0, priors=None, lags=(1, 2)):
    """직전 N개 시즌의 성적을 lag 피처로 붙인다."""
    new = []
    df = df.copy()
    for pre, ncol, rcols in SPECS:
        idc = ID_OF[pre]
        if pre not in profiles:
            continue
        prof = profiles[pre]
        rc = [c for c in rcols if f"yr_{c}" in prof.columns]
        for lag in lags:
            p = prof.copy()
            p["season"] = p["season"] + lag        # lag 시즌 전 성적을 현재 행에 붙임
            ren = {"yr_n": f"p{lag}_{pre}_n"}
            for c in rc:
                ren[f"yr_{c}"] = f"p{lag}_{c}_S"
            p = p.rename(columns=ren)
            df = df.merge(p[[idc, "season"] + list(ren.values())],
                          on=[idc, "season"], how="left")
            n = df[f"p{lag}_{pre}_n"].to_numpy(np.float64)
            df[f"p{lag}_{pre}_n"] = np.nan_to_num(n)
            new.append(f"p{lag}_{pre}_n")
            for c in rc:
                pr = 0.5 if priors is None else float(priors.get(c, 0.5))
                s = np.nan_to_num(df[f"p{lag}_{c}_S"].to_numpy(np.float64))
                df[f"p{lag}_{c}"] = (s + k * pr) / (np.nan_to_num(n) + k)
                new.append(f"p{lag}_{c}")
                df = df.drop(columns=[f"p{lag}_{c}_S"])
        # 추세와 시즌내-직전시즌 격차
        for c in rc:
            if f"p1_{c}" in df.columns and f"p2_{c}" in df.columns:
                df[f"trend_{c}"] = df[f"p1_{c}"] - df[f"p2_{c}"]
                new.append(f"trend_{c}")
            if f"p1_{c}" in df.columns and f"std_{c}" in df.columns:
                df[f"gap_{c}"] = df[f"std_{c}"] - df[f"p1_{c}"]
                new.append(f"gap_{c}")
    return df, new


def season_priors(df, rcols=None):
    """시즌별 리그 평균 rate. 수축 목표를 시즌 수준에 맞추기 위해 쓴다 (E102).

    왜 필요한가: 학습 전체 평균(2019~2024 = 0.532)으로 수축하면, 표본이 적은 행을
    **예측 대상 시즌 수준(0.486)보다 훨씬 위로** 끌어올린다. 이 상향 편향이
    전 구간에 깔려 BSS를 40점 가까이 깎는다.
    직전 완료 시즌의 리그 평균으로 수축하면 그 편향이 대부분 사라진다.
    (train만으로 만드는 상수이고, 시즌 S 행에는 S-1 값을 쓰므로 누수도 없다.)
    """
    if rcols is None:
        rcols = [c for spec in SPECS for c in spec[2]]
    m = df.groupby("season")[[c for c in rcols if c in df.columns]].mean()
    m.index = m.index + 1          # 시즌 S 행에는 S-1 평균을 붙인다
    last = m.iloc[[-1]].copy()
    last.index = [m.index.max() + 1]
    return pd.concat([m, last])    # 마지막 시즌+1(=2025) 행도 채움


def add_std(df, anchors, k=30.0, priors=None, to_career=True, multi_k=(),
            season_prior=None, excess=False, ratio=False, k_by=None,
            expose_anchor=False):
    """차분으로 '당해 시즌' 지표를 만들어 붙인다.

    to_career=True (레버 E) - **그 선수의 통산 rate로 수축**한다.
      기존엔 리그 사전확률(0.5)로 수축했는데, 시즌 초 표본이 적은 투수를 0.5로
      끌어당기는 건 틀렸다. "정보가 없으면 그 선수 평소대로"가 올바른 기본값이다.
    multi_k (레버 F) - 수축 강도를 하나로 고정하지 않고 여러 개를 병렬로 준다.
      표본이 적을 때와 많을 때 최적 수축이 다르므로 모델이 맥락에 맞게 고른다.
    """
    new = []
    df = df.copy()
    for pre, ncol, rcols in SPECS:
        idc = ID_OF[pre]
        if pre not in anchors or ncol not in df.columns:
            continue
        df = df.merge(anchors[pre], on=[idc, "season"], how="left")
        n = df[ncol].to_numpy(np.float64)
        n0 = df["n0"].fillna(0).to_numpy(np.float64)
        sn = np.maximum(n - n0, 0.0)
        # E112: 그룹마다 최적 수축 강도가 다르다. 성공률은 이항(분산 0.25)이라
        # 표본 잡음이 크고, 구종 배합비는 투수마다 안정적이라 덜 수축해야 한다.
        kg = float(k) if not k_by else float(k_by.get(pre, k))
        if expose_anchor:
            # 시즌 시작 전에 확정된 순수 과거 표본수. 결측 여부는 별도 실험으로
            # 남겨 두고, 여기서는 기존 add_std와 똑같이 0으로 처리한다.
            df[f"anchor_{pre}_n"] = n0
            new.append(f"anchor_{pre}_n")
        df[f"std_{pre}_n"] = sn
        new.append(f"std_{pre}_n")
        for c in rcols:
            s0col = f"S0_{c}"
            if c not in df.columns or s0col not in df.columns:
                continue
            # Per-rate denominator. Only rates whose last-pitch outcome can be
            # recovered exactly carry a corrected n0; the rest keep the shared
            # one, so a corrected count never lands under an uncorrected sum.
            n0c = (df[f"n0_{c}"].fillna(0).to_numpy(np.float64)
                   if f"n0_{c}" in df.columns else n0)
            snc = np.maximum(n - n0c, 0.0)
            S = n * df[c].fillna(0).to_numpy(np.float64)
            ss = np.maximum(S - df[s0col].fillna(0).to_numpy(np.float64), 0.0)
            pr = 0.5 if priors is None else float(priors.get(c, 0.5))
            if season_prior is not None and c in season_prior.columns:
                # 시즌 수준에 맞춘 기본값 (E102) - 그 행 시즌의 직전 시즌 리그평균
                sp = df["season"].map(season_prior[c]).to_numpy(np.float64)
                sp = np.where(np.isnan(sp), pr, sp)
            else:
                sp = np.full(len(df), pr)
            if expose_anchor:
                # S0/n0 자체는 저표본에서 매우 시끄럽다. std와 같은 k 및 시즌
                # 사전확률로 수축하되, 당해 시즌 행은 한 건도 섞지 않는다.
                s0 = df[s0col].fillna(0).to_numpy(np.float64)
                df[f"anchor_{c}"] = (s0 + kg * sp) / (n0c + kg)
                new.append(f"anchor_{c}")
            if to_career:
                car = np.nan_to_num(df[c].to_numpy(np.float64), nan=pr)
                # 통산 rate는 '그 선수 실력'은 맞지만 **구체제 수준**에 있다.
                # 시즌 수준 비로 스케일하면 선수 개성과 시즌 수준을 둘 다 맞춘다.
                tgt = car * (sp / pr) if season_prior is not None else car
            else:
                tgt = sp
            for kk in (kg,) + tuple(multi_k):
                nm = f"std_{c}" if kk == kg else f"std_{c}_k{int(kk)}"
                df[nm] = (ss + kk * tgt) / (snc + kk)
                new.append(nm)
            # 통산 대비 이번 시즌 편차 = 체제 변화/폼 변화 성분
            df[f"std_{c}_delta"] = df[f"std_{c}"] - df[c]
            new.append(f"std_{c}_delta")
            if ratio:
                # E110: std 는 아직 **절대 성공률**이다. 0.52 는 2022년엔 평균 이하,
                # 2025년엔 평균 한참 위인데 같은 숫자로 들어간다. TE 는 이 문제를
                # (성공수+k)/(기대성공수+k) 배수로 이미 풀었고 std 만 안 풀렸다.
                # 리그 수준으로 나누면 1.0 = 그 시즌 평균이 되어 시즌 간 이전된다.
                df[f"std_{c}_ratio"] = df[f"std_{c}"] / np.maximum(sp, 1e-6)
                new.append(f"std_{c}_ratio")
            if excess:
                # k -> 무한 극한. 수축된 rate는 (표본수, 초과성공수)를 한 숫자로
                # 뭉개버린다. 둘을 따로 주면 트리가 스스로 수축 강도를 고른다.
                #   ex = 이번 시즌 기대 대비 초과 성공 수 (실력 x 출전량)
                #   z  = 그것을 이항 표준편차로 나눈 값 (표본수와 무관한 확신도)
                ex = ss - sn * tgt
                df[f"std_{c}_ex"] = ex
                df[f"std_{c}_z"] = ex / np.sqrt(
                    np.maximum(sn * tgt * (1.0 - tgt), 0.0) + 1.0)
                new += [f"std_{c}_ex", f"std_{c}_z"]
        df = df.drop(columns=[c for c in df.columns
                              if c == "n0" or c.startswith("S0_")])
    return df, new


def add_window(df):
    """중첩된 최근경기 창을 **분리된 창**으로 분해한다 (E118).

    tools/asof_identity.py 실측: prev3 는 최근 3경기 평균, prev5 는 5경기 평균이라
        경기1   = p1
        경기2~3 = (3*p3 - p1)/2
        경기4~5 = (5*p5 - 3*p3)/2
    로 역산되고, **역산값이 [0,1] 범위 밖인 비율이 0.0%** 다 (우연이면 불가능).

    지금 모델은 서로 겹치는 세 값을 받는다. p3 로 분할하면 경기1 이 섞여 들어가
    최근성 가중이 뭉개진다. 3*p3 - p1 같은 **선형 결합은 트리가 못 만든다**
    (오늘 실력추정에서 선형 59.0% vs GBDT 46.5% 였던 것과 같은 이유).

    E108(폼 층위)과 다르다. 그건 std 를 뺀 것이고 이건 중첩 제거다.
    """
    made = []
    for kind in ("success", "middle"):
        c1 = f"asof_pitcher_prev1_game_{kind}_rate"
        c3 = f"asof_pitcher_prev3_game_{kind}_rate"
        c5 = f"asof_pitcher_prev5_game_{kind}_rate"
        if not all(c in df.columns for c in (c1, c3, c5)):
            continue
        p1 = df[c1].to_numpy(np.float64)
        p3 = df[c3].to_numpy(np.float64)
        p5 = df[c5].to_numpy(np.float64)
        g23 = (3.0 * p3 - p1) / 2.0
        g45 = (5.0 * p5 - 3.0 * p3) / 2.0
        df[f"win_{kind}_g23"] = g23
        df[f"win_{kind}_g45"] = g45
        # 분리된 창끼리의 기울기 = 순수 최근성 (중첩이 없어 해석이 깨끗하다)
        df[f"win_{kind}_slope"] = p1 - g45
        df[f"win_{kind}_slope2"] = g23 - g45
        # **변동성** - 주최측이 명시한 신호("제구 안정성 · 변동성", 발표자료 p13/p16)
        # 인데 우리는 평균만 써 왔다. 중첩된 prev1/3/5 로는 산포를 계산할 수 없고
        # (겹쳐 있어 분산이 왜곡된다) 분해했기 때문에 비로소 가능해졌다.
        # 평균이 같아도 기복이 큰 투수는 다음 투구의 예측 가능성이 다르다.
        W = np.vstack([p1, g23, g45])
        df[f"win_{kind}_sd"] = np.nanstd(W, axis=0)
        df[f"win_{kind}_range"] = np.nanmax(W, axis=0) - np.nanmin(W, axis=0)
        made += [f"win_{kind}_g23", f"win_{kind}_g45",
                 f"win_{kind}_slope", f"win_{kind}_slope2",
                 f"win_{kind}_sd", f"win_{kind}_range"]
    return df, made


def add_domain(df):
    """야구 기전으로 지정한 '연속 스타일 x 이산 상황' 교차항 (E111).

    tools/domain_probe.py 실측으로 고른 것만 넣는다 (가설 단계에서 걸러진 것들:
    거르기/고의4구 - KBO 자동고의4구라 투구 자체가 없음, 부호도 반대였다.
    가비지타임 adj -0.0010, 선발 3순회 -0.0009 - 크기가 무의미).

    남은 둘은 실측 근거가 확실하다.
      ① 동일손 매치업: 투수 고정효과 제거 후 adj -0.0104 (단일 상황 중 1위, 상한 40.3).
         같은 손이면 바깥쪽·백도어로 더 공격적으로 들어가 존을 벗어난다.
      ② 타자 위협도 x 주자 유무: 타자 위협도의 기울기가 주자 없을 때 0.220,
         주자 있으면 0.097~0.113. 주자가 있으면 투수 관심이 주자로 가서
         **타자가 누구인지가 덜 중요해진다**.

    형태는 (0/1 상황) x (연속 스타일). 트리는 이런 곱을 여러 분할로만 근사하므로
    명시적으로 주면 이득이 난다 (E95 _dev 가 같은 모양으로 +27.2).
    """
    made = []
    need = ("pitcher_hand", "batter_hand", "num_runners_on")
    if not all(c in df.columns for c in need):
        return df, made
    same = (df["pitcher_hand"].to_numpy() ==
            df["batter_hand"].to_numpy()).astype(np.float64)
    empty = (df["num_runners_on"].fillna(0).to_numpy() == 0).astype(np.float64)
    df["dom_same_hand"] = same
    made.append("dom_same_hand")

    def pick(*names):
        for n in names:
            if n in df.columns:
                return df[n].to_numpy(np.float64)
        return None

    # ① 동일손 x 투수 스타일 - 존을 벗어나는 성향/반대방향 성향이 증폭되는가
    for tag, cols in [("ball", ("std_asof_pitcher_ball_rate",
                                "asof_pitcher_ball_rate")),
                      ("rev", ("std_asof_pitcher_reverse_rate",
                               "asof_pitcher_reverse_rate"))]:
        v = pick(*cols)
        if v is None:
            continue
        df[f"dom_same_{tag}"] = same * np.nan_to_num(v)
        made.append(f"dom_same_{tag}")

    # ② 타자 위협도 x 주자 유무 - 주자가 없을 때만 타자가 크게 중요하다
    bd = pick("std_asof_batter_success_rate", "asof_batter_success_rate")
    if bd is not None:
        bd = np.nan_to_num(bd, nan=0.5)
        df["dom_bat_empty"] = empty * bd
        df["dom_bat_runner"] = (1.0 - empty) * bd
        made += ["dom_bat_empty", "dom_bat_runner"]
    return df, made


def add_count_style(df):
    """카운트가 여는 **실패 모드** x 그 투수의 해당 모드 취약도 (E113).

    tools/count_intent.py 실측 (R리그 2022~24, 성공 ~ 표준화 모드비율):

      구간                     middle     ball   reverse
      3볼 (무조건 스트라이크)   -0.0108  -0.0109  -0.0225
      2스트라이크 (유인 가능)   -0.0059  -0.0120  -0.0221
      짝수 (정상 승부)          -0.0072  -0.0180  -0.0258

    카운트가 바뀌면 **그 투수의 어떤 약점이 발현되는지가 달라진다**.
    3볼이 되면 볼이 많은 투수의 단점이 사라지고(-0.0180 -> -0.0109),
    2스트라이크에선 한가운데 실투가 많은 투수의 단점이 사라진다(-0.0091 -> -0.0059).
    던지는 의도가 바뀌니 그 약점이 나올 상황 자체가 안 오는 것이다.

    처음엔 TE `pc`(투수x카운트)와 중복이라 판단해 안 만들었는데, 같은 판단으로
    미뤘던 E111 이 lr 0.01 에서 +8.3(t=3.4)로 살아났으므로 이쪽도 시험한다.
    std_ 가 있으면 당해 시즌 값을, 없으면 통산 asof 를 쓴다.
    """
    made = []
    need = ("balls_before", "strikes_before")
    if not all(c in df.columns for c in need):
        return df, made
    b = df["balls_before"].to_numpy(np.float64)
    s = df["strikes_before"].to_numpy(np.float64)
    ctx = {
        "must": (b >= 3).astype(np.float64),          # 볼넷 위기 - 무조건 넣어야
        "waste": ((s >= 2) & (b < 3)).astype(np.float64),   # 여유 - 유인구 가능
        "even": (b == s).astype(np.float64),          # 정상 승부
    }

    def pick(tag):
        for n in (f"std_asof_pitcher_{tag}_rate", f"asof_pitcher_{tag}_rate"):
            if n in df.columns:
                return np.nan_to_num(df[n].to_numpy(np.float64))
        return None

    for tag in ("ball", "middle", "reverse"):
        v = pick(tag)
        if v is None:
            continue
        for cn, cv in ctx.items():
            nm = f"cnt_{cn}_{tag}"
            df[nm] = cv * v
            made.append(nm)
    return df, made


def add_form(df):
    """경기 수준 최근 폼을 **당해 시즌 기준선** 대비로 잰다 (E108).

    기존 form_delta 는 (최근 N경기) − (통산)이었다. 그런데 통산은 구체제 수준에
    묶여 있어서(E102 에서 확인) 그 차이는 '폼'이 아니라 '체제 차이'를 크게 섞는다.
    당해 시즌 기준선(std)을 빼면 순수한 hot/cold 성분만 남는다.

    층위가 셋이므로 (통산 / 시즌 / 최근경기) 그 사이 차이를 전부 준다:
      form_*_s1,s3,s5  = 최근 1/3/5경기 − 당해 시즌   (얼마나 벗어나 있나)
      form_*_trend     = 최근1경기 − 최근5경기        (단기 기울기)
    """
    made = []
    for kind in ("success", "middle"):
        base = f"std_asof_pitcher_{kind}_rate"
        cols = {g: f"asof_pitcher_prev{g}_game_{kind}_rate" for g in (1, 3, 5)}
        if base not in df.columns or not all(c in df.columns
                                             for c in cols.values()):
            continue
        for g, c in cols.items():
            nm = f"form_{kind}_s{g}"
            df[nm] = df[c] - df[base]
            made.append(nm)
        df[f"form_{kind}_trend"] = df[cols[1]] - df[cols[5]]
        df[f"form_{kind}_trend3"] = df[cols[1]] - df[cols[3]]
        made += [f"form_{kind}_trend", f"form_{kind}_trend3"]
    return df, made


def add_recent_relation(df, mode="reverse"):
    """Explicit career-style x recent-control relations from the full audit.

    A time-honest 47-column pair screen found that career reverse/success rate
    combined with prev1/3/5 success was one of the few pairs whose raw BSS and
    incremental lookup gain were positive in 2022, 2023 and 2024.  A depth-8
    tree can approximate products but needs several splits, so expose only the
    audited low-rank relation rather than a broad polynomial expansion.
    """
    made = []
    recent = [f"asof_pitcher_prev{g}_game_success_rate" for g in (1, 3, 5)]
    if not all(c in df.columns for c in recent):
        return df, made
    if mode in ("reverse", "both") and "asof_pitcher_reverse_rate" in df:
        z = df["asof_pitcher_reverse_rate"].to_numpy(np.float64) - 0.5
        for g, c in zip((1, 3, 5), recent):
            nm = f"rel_prev{g}_success_x_reverse"
            df[nm] = (df[c].to_numpy(np.float64) - 0.5) * z
            made.append(nm)
    if mode in ("career", "both") and "asof_pitcher_success_rate" in df:
        z = df["asof_pitcher_success_rate"].to_numpy(np.float64)
        for g, c in zip((1, 3, 5), recent):
            v = df[c].to_numpy(np.float64)
            nm = f"rel_prev{g}_success_minus_career"
            df[nm] = v - z
            made.append(nm)
            nm = f"rel_prev{g}_success_x_career"
            df[nm] = (v - 0.5) * (z - 0.5)
            made.append(nm)
    return df, made


def add_cross(df):
    """레버 H - 당해 시즌 실력 x 그 상황에서의 상대 배수.

    신호감사 오라클에서 투수x카운트가 2740.9로 가장 높았다. `std_*`는 그 투수의
    당해 시즌 실력(수준)을, TE `_dev`는 특정 상황에서 자기 평소보다 얼마나
    나은지(배수)를 담는다. 둘을 곱하면 **그 투수의 그 상황 당해 시즌 기대
    성공률**이 되어 타깃을 직접 추정한다. 트리는 곱을 만들기 어려우니 명시적으로 준다.
    """
    made = []
    base = "std_asof_pitcher_success_rate"
    if base not in df.columns:
        return df, made
    for c in [c for c in df.columns if c.endswith("_ratio_dev")]:
        nm = "x_" + c.replace("te_pitcher_", "").replace("_ratio_dev", "")
        df[nm] = df[base] * df[c]
        made.append(nm)
    return df, made
