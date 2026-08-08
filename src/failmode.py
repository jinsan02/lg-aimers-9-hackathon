"""실패모드 라벨 복원 (E124) — train 한 행에 이진 타깃 1개가 아니라 라벨 5개가 있다.

E99 는 asof 가 통산 누적임을 알고 **시즌 차분**으로 성적을 복원했다.
그런데 갱신 단위는 투구마다 +1 이다(감사 100.0%). 그렇다면 **한 투구 차분**은
그 투구 하나의 결과다.

    S_x(t) = asof_pitcher_n(t) x asof_pitcher_x_rate(t)
    S_x(t+1) - S_x(t) = 1{그 투구 t 가 x 였는가}

실측(tools/failmode_labels.py): 99.95% 행에서 복원 가능, 0/1 판정 99.89%,
복원한 success 가 control_success 와 96.79% 일치(나머지는 rate 저장 반올림 오차).

⚠️ 합법성 — 반드시 지킬 것
  이 함수는 **train DataFrame 에만** 부른다. 같은 대수를 평가 데이터에 쓰면
  2025 타깃이 복원되는데, 그것은 "평가 데이터 내부 다른 행을 이용한 피처" 로
  명시적으로 금지돼 있다. 그래서 이 모듈은 **감독 신호(타깃)** 만 만들고
  피처는 하나도 만들지 않는다. 추론 경로(fpipe.transform)는 이 파일을 import
  하지 않는다.

왜 쓰는가:
  실패 정의는 ① 존 중앙(실투) ② 존 밖(커맨드) ③ 포수 요구 반대(배터리 소통) 로
  **서로 다른 능력**이다. 게다가 셋은 분할이 아니다 — 실측 기저율이
  success .5237 / strike .4433 / ball .3695 / middle .1496 / reverse .2290 이고
  strike+ball = .8129 로 1 이 아니다. success > strike 이므로 **제구 성공은
  스트라이크의 부분집합도 아니다** (포수가 존 밖을 요구하면 거기 꽂는 게 성공).

  타깃과 함께 셀로 묶어 MultiClass 로 풀면 행당 그래디언트가 스칼라에서
  벡터가 된다. 출력 기하가 심플렉스로 바뀌므로 형제 CatBoost 와 불일치(rms)가
  커지는데 같은 피처·같은 알고리즘이라 성능 격차(D)는 작다 — 블렌드 기여
  조건(rms 크고 D 작을 것)에 가장 가까운 저비용 후보다.
"""

import numpy as np
import pandas as pd

TARGET = "control_success"
MODES = ("middle", "ball", "reverse")
MIN_SHARE = 0.005          # 이보다 드문 셀은 성공 여부만 남기고 묶는다


def _pitch_labels(df, modes=MODES):
    """(투수 그룹 안에서) 한 투구 차분으로 실패모드 라벨을 복원한다.

    반환은 그 **행 자신의 투구** 기준이다. asof(t) 는 t 직전까지이므로
    차분은 투구 t 의 결과이고, 그것을 행 t 에 맞추려면 한 칸 당겨야 한다.
    """
    order = "row_id" if "row_id" in df.columns else None
    d = df.sort_values(order, kind="stable") if order else df
    pid = d["pitcher_id"].to_numpy()
    n = d["asof_pitcher_n"].to_numpy(np.float64)
    step = pd.Series(n).groupby(pid).diff().to_numpy()
    ok = step == 1                                  # 같은 투수의 연속 투구

    out = {}
    for m in modes:
        c = f"asof_pitcher_{m}_rate"
        if c not in d.columns:
            out[m] = pd.Series(np.nan, index=d.index)
            continue
        S = n * d[c].to_numpy(np.float64)
        v = pd.Series(S).groupby(pid).diff().to_numpy()
        v = np.where(ok, np.round(v), np.nan)
        v = np.where((v == 0) | (v == 1), v, np.nan)   # 0/1 아니면 버린다
        # 차분은 '앞 행의 투구' 결과이므로 한 칸 당겨 그 행의 라벨로 만든다
        out[m] = pd.Series(v, index=d.index).shift(-1)
    return pd.DataFrame(out).reindex(df.index)


def build_cells(train, modes=MODES, verbose=True, context="",
                min_share=MIN_SHARE):
    """(성공, 실투, 볼, 반대) 조합을 다중분류 셀로 만든다.

    셀에 **타깃 자신을 포함**하는 것이 핵심이다. 실패모드만으로는 타깃이
    재현되지 않기 때문이다(최선의 합집합이 84.1%). 타깃 비트를 넣으면
    P(성공) = sum_{셀의 성공비트=1} P(셀) 이 **정확히** 성립한다.

    반환: (셀 코드 Series, 셀 이름 리스트, 성공 셀 인덱스 집합)
    """
    lab = _pitch_labels(train, modes)
    y = train[TARGET].to_numpy(np.int8)
    parts = [pd.Series(y, index=train.index).astype(str)]
    known = pd.Series(True, index=train.index)
    for m in modes:
        v = lab[m]
        known &= v.notna()
        parts.append(v.fillna(9).astype(int).astype(str))
    cell = parts[0].str.cat(parts[1:], sep="")

    # E164: 셀에 **상황 맥락**을 붙인다.
    #
    # 손실 지도(tools/loss_map.py)에서 0-2 카운트의 BSS 가 277.6 이다 — 전체 878.8
    # 의 3분의 1도 안 된다. 2스트라이크에서는 포수가 존 밖 유인구를 요구하므로
    # **'제구 성공'의 의미가 뒤집힌다**: 존 안에 넣으면 실패다. 나머지 84% 카운트에서
    # 배운 "존 안에 잘 넣는 투수 = 좋은 제구" 를 여기에 그대로 적용하고 있었다.
    #
    # 맥락을 셀 코드에 붙이면 다중분류가 그 구간을 **다른 문제로** 다룬다.
    # P(성공) = sum(성공비트=1인 셀) 합산식은 그대로 성립한다 — 맥락이 뭐든
    # 성공 비트는 첫 글자에 남는다.
    for c in (x.strip() for x in context.split(",") if x.strip()):
        if c not in train.columns:
            raise KeyError(f"--fm-context 컬럼 없음: {c}")
        cell = cell.str.cat(train[c].astype(str), sep="|")

    # 드문 셀과 복원 실패 행은 **성공 비트만 남기고** 묶는다 → 합산식이 보존된다
    share = cell.value_counts(normalize=True)
    # 맥락을 붙이면 셀 수가 배로 늘어 기본 임계(0.5%)에서 대부분 뭉개진다.
    rare = set(share[share < min_share].index)
    cell = cell.where(~cell.isin(rare) & known,
                      pd.Series(y, index=train.index).astype(str) + "xxx")

    names = sorted(cell.unique())
    code = cell.map({v: i for i, v in enumerate(names)}).astype(np.int16)
    succ = {i for i, v in enumerate(names) if v[0] == "1"}
    if verbose:
        print(f"실패모드 셀 {len(names)}개 (복원 {known.mean() * 100:.2f}%) | "
              f"성공 셀 {len(succ)}개")
        vc = cell.value_counts(normalize=True)
        print("  " + " ".join(f"{v}:{vc[v] * 100:.1f}%" for v in names))
    return code, names, succ


def success_prob(proba, succ):
    """MultiClass 확률에서 P(제구 성공) 을 복원한다."""
    return proba[:, sorted(succ)].sum(axis=1)
