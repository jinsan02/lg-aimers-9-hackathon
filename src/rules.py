"""야구 규칙 기반 도메인 피처 + 자연어 서술 — 같은 지식의 두 표현.

가설: 투수의 '의도'는 규칙이 만든 상황 논리로 결정된다. 그 논리를 명시하면
      모델이 데이터에서 재발견할 부담이 줄어든다.

같은 규칙을 (a) 수치 피처 (b) 한국어 문장으로 각각 내보내, 어느 표현이
실제로 점수를 올리는지 **동일 조건에서 비교**한다. (a)가 무효면 (b)도 기대값이 낮다.
"""

import numpy as np


def add_fatigue_features(df):
    """체력·심리만 분리 — 규칙 피처(결정론적 함수) 없이.

    E49에서 규칙 11종은 -12였으나, 여기 4종을 더하니 +12.3이었다(E75).
    규칙과 달리 이들은 **비선형 결합**(로그·역수·곱)이라 트리가 분할로 근사하기 어렵다.
    """
    out = df.copy()
    new = []
    exp = out["asof_pitcher_n"].fillna(0)
    # 체력: 이닝 깊이 ÷ 경험 로그 — 신인일수록 이닝 부담이 크게 작용
    out["fatigue"] = out["inning"] * (1.0 / np.log1p(exp + 100))
    # 심리: 점수차 역수 (접전일수록 큼) × 상황 중요도
    out["tight_game"] = 1.0 / (1.0 + out["score_diff_pitcher_team"].abs())
    out["pressure_idx"] = out["tight_game"] * out["li"].fillna(1.0)
    out["late_tight_rookie"] = ((out["inning"] >= 7)
                                & (out["score_diff_pitcher_team"].abs() <= 2)
                                & (exp < 500)).astype(np.int8)
    new += ["fatigue", "tight_game", "pressure_idx", "late_tight_rookie"]
    return out, new


def add_rule_features(df):
    """야구 상황 논리 → 수치 피처. 반환 (df, 새 컬럼명)"""
    out = df.copy()
    new = []
    b, s = out["balls_before"], out["strikes_before"]
    o = out["outs_before"]
    r1, r2, r3 = out["runner_on_1b"], out["runner_on_2b"], out["runner_on_3b"]

    # 1) 카운트 압박 — 3볼은 반드시 스트라이크, 2스트라이크는 유인구 여유
    out["must_strike"] = ((b == 3) & (s < 2)).astype(np.int8)      # 볼넷 위기
    out["can_waste"] = ((s == 2) & (b < 2)).astype(np.int8)        # 유인구 여유
    out["full_count"] = ((b == 3) & (s == 2)).astype(np.int8)
    out["ahead_in_count"] = (s > b).astype(np.int8)
    new += ["must_strike", "can_waste", "full_count", "ahead_in_count"]

    # 2) 폭투 위험 — 3루 주자 + 2아웃 미만이면 낮은 유인구를 못 던진다
    out["wp_risk"] = ((r3 == 1) & (o < 2)).astype(np.int8)
    out["risky_low_pitch_blocked"] = ((r3 == 1) & (s == 2) & (o < 2)).astype(np.int8)
    new += ["wp_risk", "risky_low_pitch_blocked"]

    # 3) 주자 견제 부담 — 1루만 있으면 슬라이드스텝으로 딜리버리가 바뀜
    out["hold_runner"] = ((r1 == 1) & (r2 == 0)).astype(np.int8)
    out["scoring_position"] = ((r2 == 1) | (r3 == 1)).astype(np.int8)
    new += ["hold_runner", "scoring_position"]

    # 4) 승부처 — 늦은 이닝 + 접전 + 높은 li
    out["late_close"] = ((out["inning"] >= 7)
                         & (out["score_diff_pitcher_team"].abs() <= 2)).astype(np.int8)
    out["blowout"] = (out["score_diff_pitcher_team"].abs() >= 6).astype(np.int8)
    new += ["late_close", "blowout"]

    # 4-b) 체력·심리 (사용자 요청) — 이닝 깊이 × 경험, 접전 압박
    if "asof_pitcher_n" in out.columns:
        exp = out["asof_pitcher_n"].fillna(0)
        # 체력: 이닝이 깊을수록 부담, 경험이 적을수록 크게 작용
        out["fatigue"] = out["inning"] * (1.0 / np.log1p(exp + 100))
        # 심리: 점수차가 작을수록 · li가 클수록 압박
        out["tight_game"] = (1.0 / (1.0 + out["score_diff_pitcher_team"].abs()))
        out["pressure_idx"] = out["tight_game"] * out["li"].fillna(1.0)
        # 후반 접전 × 신인
        out["late_tight_rookie"] = ((out["inning"] >= 7)
                                    & (out["score_diff_pitcher_team"].abs() <= 2)
                                    & (exp < 500)).astype(np.int8)
        new += ["fatigue", "tight_game", "pressure_idx", "late_tight_rookie"]

    # 5) 타석 압박 지수 — 규칙 요소 결합
    out["pressure"] = (out["must_strike"] * 2 + out["wp_risk"]
                       + out["late_close"] + out["scoring_position"]
                       - out["can_waste"] - out["blowout"]).astype(np.int8)
    new.append("pressure")
    return out, new


# ---------- 같은 규칙의 자연어 표현 (LLM 트랙용) ----------
def verbalize(row):
    """한 행 → 야구 규칙 맥락이 담긴 한국어 서술."""
    parts = []
    b, s = int(row["balls_before"]), int(row["strikes_before"])
    o = int(row["outs_before"])
    parts.append(f"{int(row['inning'])}회 {'초' if row['top_bottom'] == 'T' else '말'}, "
                 f"{o}아웃, 볼카운트 {b}-{s}.")
    if b == 3 and s < 2:
        parts.append("볼넷 위기라 투수는 스트라이크를 넣어야 하는 압박을 받는다.")
    elif s == 2 and b < 2:
        parts.append("유리한 카운트라 존 밖 유인구를 던질 여유가 있다.")
    elif b == 3 and s == 2:
        parts.append("풀카운트로 물러설 곳이 없다.")

    bases = []
    if row["runner_on_1b"]:
        bases.append("1루")
    if row["runner_on_2b"]:
        bases.append("2루")
    if row["runner_on_3b"]:
        bases.append("3루")
    if bases:
        parts.append(f"{'·'.join(bases)}에 주자가 있다.")
        if row["runner_on_3b"] and o < 2:
            parts.append("3루 주자가 있어 폭투가 실점으로 직결되므로 낮은 공을 피한다.")
        if row["runner_on_1b"] and not row["runner_on_2b"]:
            parts.append("1루 주자 견제로 투구 동작이 빨라져 제구가 흔들리기 쉽다.")
    else:
        parts.append("주자가 없어 투수는 자유롭게 승부할 수 있다.")

    d = float(row["score_diff_pitcher_team"])
    if abs(d) <= 2 and int(row["inning"]) >= 7:
        parts.append("늦은 이닝 접전이라 한 구의 무게가 크다.")
    elif abs(d) >= 6:
        parts.append("점수 차가 크게 벌어져 있다.")

    hand = "우완" if row["pitcher_hand"] == 2 else "좌완"
    bhand = "우타자" if row["batter_hand"] == 2 else "좌타자"
    parts.append(f"{hand} 투수가 {bhand}를 상대한다.")
    sr = row.get("asof_pitcher_success_rate")
    if sr == sr and sr is not None:
        lvl = "제구가 좋은" if sr > 0.53 else ("제구가 불안한" if sr < 0.48 else "평균적인")
        parts.append(f"이 투수는 통산 제구 성공률 {sr:.3f}로 {lvl} 편이다.")
    return " ".join(parts)
