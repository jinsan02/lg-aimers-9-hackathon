"""F/R 분리 모델의 통합 BSS 계산 — 전체 2024 검증 기준으로 환산."""
import numpy as np
import pandas as pd

DATA, T = "./data", "control_success"
df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                 usecols=["season", "game_type", T])
v = df[df.season == 2024]
nR, nF = int((v.game_type == "R").sum()), int((v.game_type == "F").sum())
rR = float(v.loc[v.game_type == "R", T].mean())
rF = float(v.loc[v.game_type == "F", T].mean())
r = float(v[T].mean())

BSS_R, BSS_F, BSS_SINGLE = 717.44, 369.55, 783.53
bR = rR * (1 - rR) * (1 - BSS_R / 1e5)
bF = rF * (1 - rF) * (1 - BSS_F / 1e5)
b_split = (nR * bR + nF * bF) / (nR + nF)
b_single = r * (1 - r) * (1 - BSS_SINGLE / 1e5)
bss_split = 1e5 * (1 - b_split / (r * (1 - r)))

print(f"검증 2024: 전체 {nR + nF} (R {nR} r={rR:.4f} / F {nF} r={rF:.4f})")
print(f"R 전용 모델 BSS {BSS_R:.2f} → Brier {bR:.6f}")
print(f"F 전용 모델 BSS {BSS_F:.2f} → Brier {bF:.6f}")
print(f"분리 통합 Brier {b_split:.6f} → **통합 BSS {bss_split:.2f}**")
print(f"단일 모델   Brier {b_single:.6f} → BSS {BSS_SINGLE:.2f}")
print(f"\n판정: 분리 {bss_split - BSS_SINGLE:+.2f}")
