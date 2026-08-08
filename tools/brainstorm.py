"""새 관점 브레인스토밍 — 아직 안 본 각도들을 데이터로 검증.

가설 5개:
 A. 타깃은 3가지 실패모드(가운데/존밖/반대방향)의 합 → asof rate 3종이 그걸 분해하는가?
 B. 시즌 drift는 연속적인가, 특정 시점의 계단인가? (월 단위 해상도)
 C. 선발/불펜 역할이 단일 행에서 추론 가능한가? 제구력이 다른가?
 D. balls+strikes = 이 타석 최소 투구수 → 타석 내 위치 정보인가?
 E. 투수의 '시즌 내 궤적'(첫 등판 vs 시즌 후반)이 신호인가?
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option("display.width", 200)
DATA, T = "./data", "control_success"
df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")

print("=== A. 타깃 = 3가지 실패모드의 합인가? ===")
fail = df[["asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
           "asof_pitcher_reverse_rate"]].sum(axis=1)
succ = df["asof_pitcher_success_rate"]
ok = succ.notna() & fail.notna()
print(f"  1 - (middle+ball+reverse) vs success_rate 상관: "
      f"{np.corrcoef(1 - fail[ok], succ[ok])[0, 1]:.4f}")
print(f"  평균 잔차: {((1 - fail[ok]) - succ[ok]).mean():.4f} "
      f"(표준편차 {((1 - fail[ok]) - succ[ok]).std():.4f})")
print("  → 상관이 1에 가까우면 3종이 타깃을 완전 분해 = 개별 모드 모델링 가능")

# 세 실패모드의 상대 구성비 (같은 성공률이라도 실패 방식이 다름)
tot = fail.replace(0, np.nan)
for c in ["middle", "ball", "reverse"]:
    df[f"mix_{c}"] = df[f"asof_pitcher_{c}_rate"] / tot
print("\n  실패 구성비와 타깃 상관 (같은 성공률에서도 실패 '방식'이 다른가):")
for c in ["middle", "ball", "reverse"]:
    v = df[f"mix_{c}"]
    m = v.notna()
    print(f"    mix_{c:8s} {np.corrcoef(v[m], df[T][m])[0, 1]:+.4f}")

print("\n=== B. drift는 연속인가 계단인가? (시즌×월) ===")
tab = df.pivot_table(index="season", columns="game_month", values=T, aggfunc="mean")
print(tab.round(4).to_string())
season_mean = df.groupby("season")[T].mean()
diffs = season_mean.diff().dropna()
print(f"  시즌별 변화량: {diffs.round(4).to_dict()}")
print(f"  변화량의 표준편차 {diffs.std():.4f} — 균일하면 연속, 튀면 계단")

print("\n=== C. 선발/불펜 역할 (단일 행 추론 가능) ===")
# 1~2회 등판 = 선발 가능성 높음. 행 단위로 inning만 보면 됨(다른 행 미사용)
df["early_inning"] = (df.inning <= 2).astype(int)
print(df.groupby("early_inning")[T].agg(["size", "mean"]).round(4).to_string())
print("\n  이닝별 성공률:")
print(df.groupby("inning")[T].agg(["size", "mean"]).round(4).head(12).to_string())

print("\n=== D. 타석 내 위치 (balls+strikes = 최소 투구수) ===")
df["pa_depth"] = df.balls_before + df.strikes_before
print(df.groupby("pa_depth")[T].agg(["size", "mean"]).round(4).to_string())

print("\n=== E. 투수 경험 구간 × 시즌 (신인 페널티가 시즌마다 다른가) ===")
df["exp_bin"] = pd.cut(df.asof_pitcher_n, [-1, 200, 1000, 5000, 99999],
                       labels=["<200", "200-1k", "1k-5k", "5k+"])
piv = df.pivot_table(index="exp_bin", columns="season", values=T,
                     aggfunc="mean", observed=True)
print(piv.round(4).to_string())

# 시각화
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
for s in sorted(df.season.unique()):
    sub = tab.loc[s].dropna()
    ax[0].plot(sub.index, sub.values, marker="o", label=str(s))
ax[0].set_title("B. 시즌×월 성공률 (계단 여부)")
ax[0].set_xlabel("월")
ax[0].legend(fontsize=7)

g = df.groupby("inning")[T].mean()
ax[1].plot(g.index, g.values, marker="s", color="crimson")
ax[1].set_title("C. 이닝별 성공률")
ax[1].set_xlabel("이닝")

piv.T.plot(ax=ax[2], marker="o")
ax[2].set_title("E. 투수 경험구간별 시즌 추이")
ax[2].set_xlabel("시즌")
ax[2].legend(fontsize=7)
plt.tight_layout()
plt.savefig("out/brainstorm.png", dpi=120)
print("\n저장: out/brainstorm.png")
