"""실제 train 행으로 직렬화 길이 실측 — 피처 수별 토큰 비용.

llm_serialization_handoff.md(팀 실측 207토큰/행)와 llm_feasibility.md(68토큰/행)의
차이가 '피처 개수' 때문인지 검증한다.
"""

import numpy as np
import pandas as pd

CHARS_PER_TOKEN = 3.5  # 영숫자 혼합 compact 표기 기준

test_cols = pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=0).columns
feats = [c for c in test_cols if c != "row_id"]
df = pd.read_csv("data/train.csv", encoding="utf-8-sig", usecols=feats, nrows=2000)

# 중요도 상위 피처 (EDA/gain 기준)
TOP20 = ["season", "game_month", "game_type", "inning", "top_bottom",
         "balls_before", "strikes_before", "outs_before", "base_state",
         "score_diff_pitcher_team", "li", "pitcher_hand", "batter_hand",
         "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
         "asof_pitcher_middle_rate", "asof_pitcher_prev5_game_success_rate",
         "asof_batter_success_rate", "asof_pitcher_fastball_rate"]


def ser(row, cols, short=False):
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3g}"
        return str(v)
    if short:  # 키를 약어로
        return " ".join(f"{c[:6]}={fmt(row[c])}" for c in cols)
    return " ".join(f"{c}={fmt(row[c])}" for c in cols)


for label, cols, short in [("전체 47피처 (풀네임)", feats, False),
                           ("전체 47피처 (키 약어)", feats, True),
                           ("상위 20피처 (풀네임)", TOP20, False),
                           ("상위 20피처 (키 약어)", TOP20, True)]:
    lens = np.array([len(ser(df.iloc[i], cols, short)) for i in range(len(df))])
    tok = lens / CHARS_PER_TOKEN
    print(f"{label:22s} 문자 중앙값 {np.median(lens):5.0f} "
          f"→ 토큰 추정 중앙값 {np.median(tok):5.0f} p99 {np.percentile(tok, 99):5.0f}")

print("\n예시(전체 47피처):")
print(" ", ser(df.iloc[0], feats)[:300], "...")
