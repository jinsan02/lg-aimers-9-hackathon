"""컬럼 관련성 정밀 분석 — 시각화 포함.

핵심 질문 3개:
  A. 피처↔타깃 관계가 시즌을 넘어 **안정적인가** (안정 = 2025로 이전됨, 불안정 = 잡음)
  B. 피처끼리 얼마나 **중복**인가 (다중공선성 → 앙상블 다양성 한계의 원인)
  C. 표준화/정규화가 필요한 **분포 병리**가 있는가

산출: out/relations.png (4패널) + 콘솔 표
실행: ~/.venvs/aimers/bin/python tools/column_relations.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option("display.width", 200)
DATA, TARGET = "./data", "control_success"

test_cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
feats = [c for c in test_cols if c != "row_id"]
df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig", usecols=feats + [TARGET])
num = [c for c in feats if df[c].dtype != object]
seasons = sorted(df.season.unique())

# ---------- A. 시즌별 상관 안정성 ----------
rows = []
for c in num:
    if c == "season":
        continue
    cs = []
    for s in seasons:
        sub = df[df.season == s]
        v = sub[c]
        if v.nunique() < 2:
            cs.append(np.nan)
            continue
        cs.append(float(np.corrcoef(v.fillna(v.median()), sub[TARGET])[0, 1]))
    cs = np.array(cs, dtype=float)
    rows.append({"col": c, "mean_corr": np.nanmean(cs), "std_corr": np.nanstd(cs),
                 "min": np.nanmin(cs), "max": np.nanmax(cs),
                 "sign_flip": bool(np.nanmin(cs) * np.nanmax(cs) < 0)})
st = pd.DataFrame(rows)
st["abs_mean"] = st.mean_corr.abs()
st["stability"] = st.abs_mean / (st.std_corr + 1e-6)   # 클수록 신뢰
st = st.sort_values("abs_mean", ascending=False)

print("=== A. 시즌 간 상관 안정성 (상위 18) ===")
print("stability = |평균상관| / 상관표준편차  (높을수록 2025 이전 신뢰도↑)")
print(st.head(18)[["col", "mean_corr", "std_corr", "stability", "sign_flip"]]
      .round(4).to_string(index=False))
print("\n부호가 시즌마다 뒤집히는 피처(신뢰 불가):")
print(st[st.sign_flip].col.tolist())

# ---------- B. 피처 간 중복 ----------
corr = df[num].corr().abs()
np.fill_diagonal(corr.values, 0)
pairs = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
         .stack().sort_values(ascending=False))
print("\n=== B. 다중공선성 상위 12쌍 ===")
print(pairs.head(12).round(3).to_string())
print(f"|r|>0.9 쌍: {(pairs > 0.9).sum()}개 | >0.7: {(pairs > 0.7).sum()}개")

# ---------- C. 분포 병리 ----------
print("\n=== C. 표준화 필요성 진단 ===")
prof = []
for c in num:
    s = df[c].dropna()
    prof.append({"col": c, "skew": float(s.skew()), "kurt": float(s.kurtosis()),
                 "range": float(s.max() - s.min()),
                 "zero%": float((s == 0).mean() * 100)})
pf = pd.DataFrame(prof)
print(pf.reindex(pf["skew"].abs().sort_values(ascending=False).index)
      .head(10).round(2).to_string(index=False))
print(f"\n스케일 범위 최대/최소 비: "
      f"{pf['range'].max() / max(pf['range'][pf['range'] > 0].min(), 1e-9):.0f}배")

# ---------- 시각화 ----------
fig, ax = plt.subplots(2, 2, figsize=(16, 11))

top = st.head(12)
ax[0, 0].barh(top.col[::-1], top.mean_corr[::-1],
              xerr=top.std_corr[::-1], color="steelblue")
ax[0, 0].axvline(0, color="k", lw=.8)
ax[0, 0].set_title("A1. 피처↔타깃 상관 (막대=시즌평균, 오차=시즌표준편차)")

ax[0, 1].scatter(st.abs_mean, st.std_corr, s=28)
for _, r in st.head(8).iterrows():
    ax[0, 1].annotate(r.col[:22], (r.abs_mean, r.std_corr), fontsize=7)
ax[0, 1].set_xlabel("|평균 상관| (신호 크기)")
ax[0, 1].set_ylabel("상관 표준편차 (시즌 불안정성)")
ax[0, 1].set_title("A2. 신호 vs 불안정성 — 우하단이 좋은 피처")

for c in st.head(5).col:
    cs = [float(np.corrcoef(df[df.season == s][c].fillna(df[c].median()),
                            df[df.season == s][TARGET])[0, 1]) for s in seasons]
    ax[1, 0].plot(seasons, cs, marker="o", label=c[:26])
ax[1, 0].axhline(0, color="k", lw=.8)
ax[1, 0].set_title("A3. 상위 피처의 시즌별 상관 궤적")
ax[1, 0].legend(fontsize=7)

sel = [c for c in num if c.startswith("asof_")][:14]
im = ax[1, 1].imshow(df[sel].corr().abs(), cmap="magma", vmin=0, vmax=1)
ax[1, 1].set_xticks(range(len(sel)))
ax[1, 1].set_xticklabels([c[5:22] for c in sel], rotation=90, fontsize=6)
ax[1, 1].set_yticks(range(len(sel)))
ax[1, 1].set_yticklabels([c[5:22] for c in sel], fontsize=6)
ax[1, 1].set_title("B. asof_* 상호 상관 (밝을수록 중복)")
fig.colorbar(im, ax=ax[1, 1], fraction=.046)

plt.tight_layout()
plt.savefig("out/relations.png", dpi=120)
print("\n저장: out/relations.png")
