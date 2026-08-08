"""1등과의 점수 격차를 '실제 예측 오차'로 환산."""

import numpy as np

LB = {"베이스라인(수료선)": 549.51, "우리 제출 최고": 845.91,
      "우리 대기중(blendv3 예상)": 907.0, "1등": 989.0}
N = 245_789


def brier(score, r):
    return r * (1 - r) * (1 - score / 100000)


print("가정: 2025 평균 제구 성공률 r (비공개) — 민감도 확인용으로 3개 값\n")
for r in [0.47, 0.48, 0.50]:
    base = r * (1 - r)
    print(f"── r = {r:.2f} (상수예측 Brier {base:.5f}) ──")
    for k, v in LB.items():
        print(f"   {k:24s} 점수 {v:7.2f} | Brier {brier(v, r):.6f}")
    gap = brier(LB["우리 제출 최고"], r) - brier(LB["1등"], r)
    print(f"   → 우리(845.91)와 1등의 Brier 차이: {gap:.6f}")
    print(f"   → 이 차이를 '평균 예측이 어긋난 정도'로만 설명하면 δ = {np.sqrt(gap):.4f} "
          f"({np.sqrt(gap) * 100:.2f}%p)\n")

r = 0.48
print("=== 해석 1: 상대적 위치 ===")
for k, v in LB.items():
    if k != "1등":
        print(f"  {k:24s} 1등 실력의 {v / LB['1등'] * 100:5.1f}% 확보")

print("\n=== 해석 2: 24.6만 투구 전체의 제곱오차 총량 ===")
for k in ["우리 제출 최고", "1등"]:
    print(f"  {k:24s} {brier(LB[k], r) * N:9.1f}")
print(f"  차이                     {(brier(LB['우리 제출 최고'], r) - brier(LB['1등'], r)) * N:9.1f}"
      f"  ({N:,}개 예측에 분산)")

print("\n=== 해석 3: 우리 실험에서 나온 개선 단위와 비교 ===")
print("  피처 v2 도입      로컬 +46  (LB 환산 ≈ +40)")
print("  CatBoost 전환     로컬 +105 (LB 환산 ≈ +90)")
print("  refit 수정        LB +180 (버그 수정)")
print(f"  1등까지 남은 격차  LB +{LB['1등'] - LB['우리 제출 최고']:.0f} "
      f"(대기중 제출 반영 시 +{LB['1등'] - LB['우리 대기중(blendv3 예상)']:.0f})")
