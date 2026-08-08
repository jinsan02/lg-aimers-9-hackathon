"""LB 예상값 산정 — 관측된 로컬↔LB 대응 3점으로 외삽."""
import numpy as np

# (로컬 2024 홀드아웃, 실제 LB) 관측점
OBS = [("RF", 415.57, 549.64), ("LGBM E06", 632.19, 764.44),
       ("CatBoost greedy 앙상블", 743.39, 845.91)]
CAND = [("blenddiv (다양성 균등 6종)", 797.45),
        ("blendv4 (greedy 8종)", 805.83),
        ("blendv3 (greedy 7종)", 804.70)]

x = np.array([o[1] for o in OBS])
y = np.array([o[2] for o in OBS])

print("=== 관측 대응점 ===")
for n, a, b in OBS:
    print(f"  {n:24s} 로컬 {a:7.2f} → LB {b:7.2f}  (차 {b - a:+6.1f}, 비 {b / a:.3f})")

seg = (y[2] - y[1]) / (x[2] - x[1])
print(f"\n구간 기울기: 1→2 {(y[1]-y[0])/(x[1]-x[0]):.3f} | 2→3 {seg:.3f}")
print("→ 기울기가 감소(수확 체감). 최근 구간(0.733)을 외삽에 사용.\n")

lin = np.polyfit(x, y, 1)
quad = np.polyfit(x, y, 2)
print("=== 후보별 LB 예상 ===")
for n, lo in CAND:
    a = y[2] + seg * (lo - x[2])          # 최근 구간 기울기
    b = np.polyval(lin, lo)               # 전체 선형
    c = np.polyval(quad, lo)              # 2차(3점 완전적합)
    print(f"  {n:26s} 로컬 {lo:7.2f}")
    print(f"     최근기울기 {a:7.1f} | 선형 {b:7.1f} | 2차 {c:7.1f}"
          f"  → **{min(a,b,c):.0f}~{max(a,b,c):.0f}**")

print("\n=== 1등(989) 도달에 필요한 로컬 점수 ===")
for name, f in [("최근기울기", lambda v: x[2] + (v - y[2]) / seg),
                ("선형", lambda v: (v - lin[1]) / lin[0])]:
    print(f"  {name:10s} 로컬 {f(989):.0f} 필요 "
          f"(현재 797 대비 +{f(989) - 797.45:.0f})")

print("\n⚠️ 불확실성: 관측점 3개뿐이고 모두 greedy 블렌드다. "
      "다양성 균등은 일반화가 다르므로 예상보다 높게 나올 수 있다.")
