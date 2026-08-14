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

# ── 의도적 보조 라벨 잡음 (2026-08-14 사전등록) ────────────────────────────
#
# legacy 전역 shift 는 corrected 대비 +3.97 을 낸다. 두 가지가 동시에 바뀌어서
# 원인이 분리되지 않았다: (i) 3.87% 셀 오배정, (ii) taxonomy 12 → 14.
#
# tools/failmode_taxonomy_audit.py 로 (ii) 를 먼저 판정했다. legacy 만 가진 두 셀
# `1001`(8,968행)·`1100`(8,084행)은 **100% 가 오배정으로 도착한 행**이다. 버그를
# 없애면 그 셀의 모집단은 0 이므로 "legacy 의 14 에서 버그만 뺀 CLEAN14" 는
# 존재하지 않는다. 그래서 (i) 만 단독으로 재는 것이 유일하게 정의 가능한 실험이다.
#
# 아래 표는 legacy 자신의 전이 행렬에서 뽑았고, corrected-12 에 없는 목적지는
# 제거한 뒤 재정규화했다. 그 결과 legacy 이동의 29.74% 는 재현되지 않는다 —
# 이 팔이 답하는 질문이 "같은 셀 집합 안에서의 잡음만으로 충분한가" 인 이유다.
#
# 성공 비트는 절대 움직이지 않는다. 표의 모든 전이가 같은 성공 블록 안에 있고,
# 그것이 legacy 의 실제 성질이기도 하다(셀 첫 글자는 control_success 에서 직접
# 오므로 차분 오류가 닿지 않는다). 감사에서 양쪽 다 True 로 확인했다.
# 한 행이 옮겨질 확률. 그 행의 **셀과 row_id 만으로** 정해진다 — 프레임 구성에
# 의존하는 값이 하나라도 끼면 선택 뷰(전체)와 배치 뷰(테스트 제외)가 서로 다른
# 행을 오염시킨다. tests/test_fm_noise.py 가 그 결함을 실제로 잡았다.
#
# 값 = legacy 의 셀별 이동률 x 0.99649845. 계수는 2026-08-14 에 official train
# 1,475,092 행에서 한 번 구했고(총 이동률이 사전등록 3.873% 가 되도록), 이후
# 재유도하거나 탐색하지 않는다.
NOISE_SRC_RATE = {
    "0000": 0.058057, "0001": 0.075843, "0010": 0.006451, "0011": 0.008301,
    "0100": 0.086098, "0101": 0.098015, "0110": 0.007505, "0111": 0.006629,
    "0xxx": 0.559012, "1000": 0.039245, "1010": 0.006169, "1xxx": 0.547424,
}
NOISE_DEST = {
    "0000": {"0001": .421475, "0010": .058204, "0011": .030105, "0100": .350728,
             "0101": .100853, "0110": .007025, "0111": .003011, "0xxx": .028600},
    "0001": {"0000": .721001, "0010": .034308, "0011": .012847, "0100": .161188,
             "0101": .056299, "0110": .001965, "0111": .000453, "0xxx": .011940},
    "0010": {"0000": .562319, "0001": .180676, "0011": .019324, "0100": .169082,
             "0101": .043478, "0110": .006763, "0111": .000966, "0xxx": .017391},
    "0011": {"0000": .597458, "0001": .159958, "0010": .042373, "0100": .128178,
             "0101": .045551, "0110": .009534, "0111": .001059, "0xxx": .015890},
    "0100": {"0000": .701314, "0001": .184116, "0010": .031049, "0011": .011131,
             "0101": .056490, "0110": .003264, "0111": .000670, "0xxx": .011968},
    "0101": {"0000": .619499, "0001": .184537, "0010": .027231, "0011": .009725,
             "0100": .148796, "0110": .001702, "0111": .000729, "0xxx": .007780},
    "0110": {"0000": .612500, "0001": .141667, "0010": .041667, "0011": .016667,
             "0100": .129167, "0101": .037500, "0xxx": .020833},
    "0111": {"0000": .517857, "0001": .196429, "0010": .053571, "0011": .017857,
             "0100": .160714, "0101": .053571},
    "0xxx": {"0000": .554348, "0001": .180435, "0010": .034783, "0011": .010870,
             "0100": .184783, "0101": .030435, "0110": .004348},
    "1000": {"1010": .268547, "1xxx": .731453},
    "1010": {"1000": .897196, "1xxx": .102804},
    "1xxx": {"1000": .927500, "1010": .072500},
}
NOISE_RATE = 0.03873         # 사전등록. 스윕 금지 (§10)
_NOISE_SALT = np.uint64(0x9E3779B97F4A7C15)


def _pitch_labels(df, modes=MODES, legacy_shift=False):
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
        # 차분은 '앞 행의 투구' 결과이므로 한 칸 당겨 그 행의 라벨로 만든다.
        # **투수 안에서** 당겨야 한다. 전역 shift 였을 때 투수 경계 99,078행
        # (6.72%) 이 다음 투수의 첫 차분을 가져왔고, middle 라벨만 31,293행
        # (2.12%) 이 실제로 틀렸다 (2026-08-13 실측). groupby 를 빼먹으면
        # 셀 다중분류가 학습하는 보조 기하 전체가 조용히 어긋난다.
        ser = pd.Series(v, index=d.index)
        # legacy_shift reproduces the pre-2026-08-13 bug on purpose, so the
        # label correction can be attributed. Nothing else about the pipeline
        # changes, which is the point: P0S showed P1 is a null on this surface,
        # so whatever produced the +5.744 is inside P0, and this is the only
        # part of P0 large enough to be it.
        out[m] = ser.shift(-1) if legacy_shift else ser.groupby(pid).shift(-1)
    return pd.DataFrame(out).reindex(df.index)


def _unit_hash(row_id):
    """행마다 고정된 [0,1) 값. 프로세스·시드·행 순서와 무관해야 한다.

    파이썬 `hash()` 는 프로세스마다 소금이 달라 못 쓴다 — 같은 명령을 두 번
    돌리면 다른 행이 오염되고, 그러면 '결정적 손상' 이라는 전제가 무너진다.
    선택 뷰와 배치 뷰가 build_cells 를 따로 부르므로 두 호출이 **같은 행을**
    골라야 하고, 그래서 키는 row_id 하나뿐이다.
    """
    h = pd.util.hash_pandas_object(pd.Series(np.asarray(row_id)), index=False)
    return ((h.to_numpy(np.uint64) ^ _NOISE_SALT) >> np.uint64(11)) / float(1 << 53)


def _apply_noise(cell, row_id, rate, names):
    """사전정의 구조로 정확히 `rate` 비율의 보조 라벨을 결정적으로 옮긴다.

    taxonomy 가 정해진 **뒤에** 부른다. 목적지는 전부 corrected-12 안이므로
    셀 집합은 변하지 않는다 — 이 팔에서 바뀌는 건 라벨 하나뿐이다.

    `rate` 는 표가 이미 그 총량이 되도록 눈금이 맞춰져 있다는 **선언**이다.
    여기서 프레임을 보고 다시 맞추지 않는다 — 그러면 행의 운명이 같이 실린
    다른 행들에 좌우되고, 뷰마다 다른 행이 오염된다.
    """
    if rate <= 0:
        return cell
    if abs(rate - NOISE_RATE) > 1e-12:
        raise ValueError(f"표는 {NOISE_RATE} 에 맞춰져 있다. 다른 비율을 쓰려면 "
                         f"표를 다시 유도해야 하고, 그건 사전등록 위반이다(§10).")
    u = _unit_hash(row_id)
    src = cell.to_numpy()
    take = u < np.array([NOISE_SRC_RATE.get(c, 0.0) for c in src])

    out = cell.to_numpy().copy()
    # 두 번째 균등값. 선택과 목적지에 같은 난수를 쓰면 목적지가 셀 안에서
    # 단조로 배열돼 손상이 한쪽으로 쏠린다.
    v = _unit_hash(pd.Series(np.asarray(row_id)).astype(str) + "|dest")
    for s, dest in NOISE_DEST.items():
        m = take & (src == s)
        if not m.any():
            continue
        ds = [d for d in dest if d in names]
        if not ds:
            continue
        p = np.array([dest[d] for d in ds], dtype=np.float64)
        p /= p.sum()
        idx = np.searchsorted(np.cumsum(p), v[m], side="right")
        out[m] = np.asarray(ds)[np.clip(idx, 0, len(ds) - 1)]
    moved = (out != cell.to_numpy()).mean()
    # English on purpose: this line is captured by tools that decode the pipe
    # with the console's cp949, and a Korean byte here crashes the reader.
    print(f"auxiliary-label corruption {moved * 100:.4f}% "
          f"(pre-registered {rate * 100:.3f}%)")
    return pd.Series(out, index=cell.index)


def build_cells(train, modes=MODES, verbose=True, context="",
                min_share=MIN_SHARE, fit_mask=None, legacy_shift=False,
                noise_rate=0.0):
    """(성공, 실투, 볼, 반대) 조합을 다중분류 셀로 만든다.

    셀에 **타깃 자신을 포함**하는 것이 핵심이다. 실패모드만으로는 타깃이
    재현되지 않기 때문이다(최선의 합집합이 84.1%). 타깃 비트를 넣으면
    P(성공) = sum_{셀의 성공비트=1} P(셀) 이 **정확히** 성립한다.

    `fit_mask` 를 주면 **분할 안전 모드**로 동작한다:
      - 라벨을 fit / 나머지 파티션 안에서 각각 따로 복원한다. 안 그러면 fit 의
        마지막 행이 검증 시즌 첫 행의 asof 상태로 라벨을 받는다 (val2024 에서
        310행, val2023 에서 319행 — 2026-08-13 실측).
      - 희소 taxonomy 를 **fit 에서 동결**한 뒤 검증에 적용한다. 현재
        min_share=.005 에서는 fit-only 와 전체가 12셀로 같지만 구조적 결함이다.
      - fit taxonomy 에 없던 검증 셀은 성공 비트를 보존한 `0xxx`/`1xxx` 로
        보낸다. NaN 으로 두지 않는다.

    반환: (셀 코드 Series, 셀 이름 리스트, 성공 셀 인덱스 집합)
    """
    if fit_mask is not None:
        fit_mask = pd.Series(np.asarray(fit_mask, bool), index=train.index)
        lab = pd.concat([_pitch_labels(train[fit_mask], modes, legacy_shift),
                         _pitch_labels(train[~fit_mask], modes, legacy_shift)]
                        ).reindex(train.index)
    else:
        lab = _pitch_labels(train, modes, legacy_shift)
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
    # 맥락을 붙이면 셀 수가 배로 늘어 기본 임계(0.5%)에서 대부분 뭉개진다.
    fallback = pd.Series(y, index=train.index).astype(str) + "xxx"
    if fit_mask is None:
        share = cell.value_counts(normalize=True)
        rare = set(share[share < min_share].index)
        cell = cell.where(~cell.isin(rare) & known, fallback)
        names = sorted(cell.unique())
    else:
        # taxonomy 는 fit 에서만 정한다.
        share = cell[fit_mask].value_counts(normalize=True)
        keep = set(share[share >= min_share].index)
        cell = cell.where(cell.isin(keep) & known, fallback)
        # 이름 목록도 fit 기준. 검증에만 있는 셀은 위에서 이미 예약 셀로 갔다.
        names = sorted(set(cell[fit_mask].unique()) | {"0xxx", "1xxx"})
        unseen = ~cell.isin(names)
        if unseen.any():                       # 있으면 안 되지만 조용히 두지 않는다
            cell = cell.where(~unseen, fallback)

    if noise_rate:
        if "row_id" not in train.columns:
            raise KeyError("의도적 손상은 row_id 로 결정된다 — 컬럼이 없다")
        before = cell
        cell = _apply_noise(cell, train["row_id"], noise_rate, set(names))
        # 주 이진 의미는 절대 바뀌면 안 된다 (§8). 표가 같은 성공 블록 안에서만
        # 움직이도록 돼 있지만, 표를 고칠 사람을 위해 여기서 강제한다.
        if not (cell.str[0] == before.str[0]).all():
            raise RuntimeError("손상이 성공 비트를 움직였다 — 표가 잘못됐다")

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
