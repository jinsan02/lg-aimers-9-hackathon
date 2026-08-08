"""결측 감사 — 원본 컬럼 vs 우리가 만든 피처, 그리고 **결측이 이전되는가**.

두 가지가 다르다.
  원본 47개 : 대회가 준 것. 결측이 거의 없다고 알려져 있으나 실측한다.
  파생 피처 : std_*(E99), te_*(E95), tmx_*(E115) 는 **cold-start 결측**이 있다.
              신인/신규 투수는 이력이 없어서다.

핵심 질문은 '결측이 몇 %인가'가 아니라 **'결측률이 시즌마다 같은가'** 다.
학습 시즌과 검증/평가 시즌의 결측 패턴이 다르면, 트리가 결측을 '표본 적은 선수'
표지로 쓰고 그 규칙이 다음 시즌으로 이전되지 않는다. E115(구종x볼카운트)가
-9.5 로 진 이유의 유력한 후보다.

실행(4070/A100): python tools/missing_audit.py
"""

import os
import sys

import numpy as np
import pandas as pd

DATA = "./data"


def main():
    tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
    print(f"train {len(tr):,}행 x {len(tr.columns)}컬럼\n")

    print("=== ① 원본 컬럼 결측 ===")
    na = tr.isna().mean().sort_values(ascending=False)
    hit = na[na > 0]
    if len(hit) == 0:
        print("  없음 (전 컬럼 0.00%)")
    else:
        for c, v in hit.items():
            print(f"  {c:<42}{v * 100:7.3f}%")

    print("\n=== ② 결측률이 시즌마다 다른가 (상위 8개) ===")
    if len(hit):
        cols = list(hit.index[:8])
        t = tr.groupby("season")[cols].apply(lambda d: d.isna().mean())
        print((t * 100).round(2).to_string())

    # ③ 파생 피처의 cold-start
    print("\n=== ③ 파생 피처 cold-start (시즌별 결측률 %) ===")
    import season_std as ss
    anchors = ss.build_anchors(tr)
    pri = {c: float(tr[c].mean()) for c in tr.columns if c.startswith("asof_")}
    sp = ss.season_priors(tr)
    d2, std_cols = ss.add_std(tr, anchors, k=80.0, priors=pri,
                              to_career=False, season_prior=sp)
    show = [c for c in std_cols if c.endswith(("_n", "success_rate"))][:6]
    t = d2.groupby("season")[show].apply(lambda d: d.isna().mean())
    print((t * 100).round(2).to_string())
    # std_n == 0 = 그 시즌 첫 등판 구간 (결측은 아니지만 정보가 없는 상태)
    z = d2.groupby("season")["std_pitcher_n"].apply(lambda s: (s == 0).mean())
    print("\n  std_pitcher_n == 0 (그 시즌 이력 0) 비율 %:")
    print((z * 100).round(2).to_string())

    # ④ tmx (E115) - 링키지 커버리지 결측
    p = f"{DATA}/processed/tm_pitchmix.csv"
    if os.path.exists(p):
        tmf = pd.read_csv(p)
        keys = ["pitcher_id", "season", "balls_before", "strikes_before"]
        m = tr[keys + ["control_success"]].merge(tmf, on=keys, how="left")
        c0 = [c for c in tmf.columns if c.startswith("tmx_")][0]
        print(f"\n=== ④ tmx_* (E115) 결측률 - 링키지 커버리지 ===")
        t = m.groupby("season")[c0].apply(lambda s: s.isna().mean() * 100)
        print(t.round(2).to_string())
        # 결측 여부가 타깃과 상관있는가 = 트리가 오용할 여지
        mm = m[c0].isna()
        print(f"\n  결측 행 성공률 {m.loc[mm, 'control_success'].mean():.4f} / "
              f"비결측 {m.loc[~mm, 'control_success'].mean():.4f} "
              f"(차이 {m.loc[mm, 'control_success'].mean() - m.loc[~mm, 'control_success'].mean():+.4f})")
        print("  -> 차이가 크고 시즌마다 결측률이 다르면, 트리가 '결측 = 약한 투수'로")
        print("     학습하고 그 규칙이 다음 시즌으로 이전되지 않는다.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, "src")
    sys.exit(main())
