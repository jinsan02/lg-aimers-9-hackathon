"""피처별·그룹별 신호량 감사 - "우리가 놓친 정보가 어디에 있나"를 정량화.

배경: LB 1등 1334.96 / 2~5등 1063~1085 / 우리 865.13.
로컬 2024 홀드아웃에서 우리 최고는 804.70(블렌드), 단일 CatBoost 784.5.
격차가 로컬에서도 보여야 하므로 **정보(피처) 격차**로 추정된다.
이 스크립트는 어떤 피처군이 얼마나 신호를 갖는지 상한을 재서 탐색 방향을 정한다.

측정 방식:
  1) 단일 피처 BSS - 그 피처만으로 학습(구간별 평균, 즉 완전 비모수 상한에 가까움)
  2) 그룹 BSS - 피처군만으로 CatBoost 학습
  3) 오라클 - 특정 키(투수, 투수×카운트 등)의 **2024 실제 평균**을 그대로 예측(누수 상한)
     -> 그 키에 이론상 얼마나 신호가 있는지의 천장
"""

import sys

import numpy as np
import pandas as pd

DATA = "./data"
TARGET = "control_success"
VAL = 2024


def bss(p, y, base):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 100000 * (1 - ((p - y) ** 2).mean() / base)


def main():
    cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
    feats = [c for c in cols if c != "row_id"]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=list(dict.fromkeys(feats + [TARGET])))
    tr = df[df.season < VAL]
    va = df[df.season == VAL]
    y = va[TARGET].to_numpy()
    r = y.mean()
    base = r * (1 - r)
    prior = tr[TARGET].mean()
    # 학습 전체 평균(0.532)과 2024 실제(0.486)의 수준차만으로 -828 BSS가 난다.
    # 그건 season 피처가 처리하는 몫이므로, 여기서는 **수준을 맞춘 뒤** 순수 형태
    # 신호만 측정한다: 각 피처 매핑을 마지막 학습시즌(2023) 수준으로 재중심화.
    anchor = tr[tr.season == VAL - 1][TARGET].mean()
    print(f"학습 {len(tr):,}행 / 검증(2024) {len(va):,}행 | r={r:.4f} "
          f"학습평균={prior:.4f} 2023평균={anchor:.4f}")
    for nm, c in [("학습 전체 평균", prior), ("2023 평균", anchor), ("2024 실제(오라클)", r)]:
        print(f"  상수 {nm:16s} {c:.4f} -> BSS {bss(np.full(len(va), c), y, base):9.2f}")
    print()

    def recentre(p):
        """예측 평균을 2023 수준으로 이동 (수준 오차 제거, 형태 신호만 남김)."""
        return p - p.mean() + anchor

    # ---- 1) 단일 피처: 학습셋 구간별 평균(shrinkage k=200)을 검증에 적용
    print("=== 단일 피처 신호 (학습셋 구간평균 -> 2024 적용) ===")
    rows = []
    for c in feats:
        s = tr[c]
        if s.dtype == object or s.nunique() <= 60:
            key_tr, key_va = tr[c].astype(str), va[c].astype(str)
        else:                       # 연속형은 학습셋 분위수 30구간
            qs = np.unique(np.quantile(s.dropna(), np.linspace(0, 1, 31)))
            key_tr = pd.cut(tr[c], qs, include_lowest=True).astype(str)
            key_va = pd.cut(va[c], qs, include_lowest=True).astype(str)
        g = tr.groupby(key_tr)[TARGET].agg(["mean", "size"])
        sm = (g["mean"] * g["size"] + prior * 200) / (g["size"] + 200)
        p = key_va.map(sm).fillna(prior).to_numpy()
        rows.append((c, bss(recentre(p), y, base), tr[c].nunique()))
    rows.sort(key=lambda t: -t[1])
    for c, s, n in rows[:18]:
        print(f"  {c:34s} {s:8.2f}  (고유값 {n})")
    print("  ...")
    for c, s, n in rows[-4:]:
        print(f"  {c:34s} {s:8.2f}  (고유값 {n})")

    # ---- 2) 오라클: 키별 2024 실제 평균 (누수 - 그 키의 신호 천장)
    print("\n=== 오라클 상한 (2024 실제 평균을 그대로 예측 - 누수, 천장 측정용) ===")
    keys = [("투수", ["pitcher_id"]),
            ("타자", ["batter_id"]),
            ("투수×카운트", ["pitcher_id", "balls_before", "strikes_before"]),
            ("투수×타자손", ["pitcher_id", "batter_hand"]),
            ("카운트", ["balls_before", "strikes_before"]),
            ("투수×타자", ["pitcher_id", "batter_id"])]
    for name, k in keys:
        g = va.groupby(k)[TARGET].transform("mean").to_numpy()
        n = va.groupby(k)[TARGET].transform("size").to_numpy()
        # 표본 1개짜리 그룹은 자기 자신 = 완전 누수 -> shrinkage로 완화한 값도 같이
        sm = (g * n + r * 30) / (n + 30)
        print(f"  {name:14s} 그룹 {va.groupby(k).ngroups:>7,}개 | "
              f"생평균 {bss(g, y, base):9.1f} | k=30 수축 {bss(sm, y, base):9.1f}")

    print("\n* 오라클은 사용 불가(2024 라벨 사용). 그 키에 '이론상 얼마나' 신호가"
          " 있는지의 천장만 알려준다. 우리 모델 784.5와 비교해 어느 키가 미개발인지 판단.")


if __name__ == "__main__":
    sys.exit(main())
