"""컬럼별 범위/카디널리티 프로파일 — LLM 토큰화 설계용.

핵심 질문: 각 컬럼을 '단일 토큰'으로 표현하려면 몇 개의 bin이 필요한가?
그렇게 하면 행당 토큰이 몇 개가 되는가?

BPE가 숫자를 쪼개 수의 크기 관계를 파괴하는 문제(llm_serialization_handoff §3)를,
값을 미리 이산화해서 '컬럼별 전용 토큰'으로 만들면 회피할 수 있다는 가설의 검증.
"""

import numpy as np
import pandas as pd

test_cols = pd.read_csv("data/test.csv", encoding="utf-8-sig", nrows=0).columns
feats = [c for c in test_cols if c != "row_id"]
df = pd.read_csv("data/train.csv", encoding="utf-8-sig", usecols=feats)

rows = []
for c in feats:
    s = df[c]
    nun = s.nunique(dropna=True)
    if s.dtype == object:
        kind, need_bin = "cat", nun
    elif nun <= 32:
        kind, need_bin = "int-small", nun
    else:
        kind = "cont"
        # 분위 32bin으로 이산화했을 때 실제 구별되는 bin 수
        q = pd.qcut(s.dropna(), 32, duplicates="drop")
        need_bin = q.cat.categories.size
    rows.append({"col": c, "kind": kind, "nunique": nun,
                 "bins_needed": need_bin,
                 "min": None if s.dtype == object else float(s.min()),
                 "max": None if s.dtype == object else float(s.max())})

prof = pd.DataFrame(rows).sort_values("nunique", ascending=False)
print(prof.to_string(index=False))

total_vocab = int(prof.bins_needed.sum())
print(f"\n=== 어휘/토큰 예산 ===")
print(f"컬럼 수: {len(feats)}")
print(f"컬럼 전용 토큰 총합(어휘 크기): {total_vocab}")
print(f"  → 컬럼 순서 고정 시 **행당 {len(feats)}토큰** (값 토큰만, 컬럼명 불필요)")
print(f"  → 컬럼명까지 넣으면 행당 {len(feats) * 2}토큰")

# 고카디널리티 ID 처리
for c in ["pitcher_id", "batter_id"]:
    print(f"{c}: {df[c].nunique()}종 → 전용 토큰 시 어휘 +{df[c].nunique()}")

print("\n=== 참고: 현행 텍스트 직렬화 ===")
print("  47피처 풀네임 = 304토큰 / 키약어 = 140토큰 (tools/count_tokens.py 실측)")
print(f"  이산화 단일토큰 방식 = {len(feats)}토큰 → 6.5배 압축")
