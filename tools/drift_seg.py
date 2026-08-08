"""세그먼트별 드리프트 보정이 가능한가 (E153).

## 근거

지금 SHIFT 는 **모든 행에 같은 상수**를 뺀다. 이건 "드리프트가 균일하다"는 가정이다.
그런데 2024 는 ABS 도입 해라 존 경계 판정이 통째로 바뀌었고, 그 영향이 한복판
직구와 바깥쪽 유인구에 같을 이유가 없다.

드리프트가 세그먼트마다 다르면, 상수 시프트는 **평균만** 잡고 분산은 남긴다.
남는 손해는 정확히  1e5 x Var_세그먼트(편향) / base  이다:

    편향 sd 0.005 -> 10점,   0.010 -> 40점

## 합법성

세그먼트별 보정값은 **train 시즌들에서만** 만든다(미학습 시즌 S 에서 잰 편향이
S+1 에도 유지되는지 확인하고, 유지되면 그 값을 상수표로 고정). 각 행은 자기
자신의 피처로 세그먼트가 정해지므로 행 독립이다. 평가 데이터의 분포는 안 본다.

## 판정

미학습 2023 에서 잰 세그먼트 편향을 미학습 2024 에 적용해서 이득이 나오는지 본다.
시즌을 못 넘어가면 (세그먼트 가중이 그랬듯) 버린다.

실행: python tools/drift_seg.py 2023 2024
"""

import glob
import sys

import numpy as np
import pandas as pd

SEGS = {
    "count": ["balls_before", "strikes_before"],
    "hand": ["pitcher_hand", "batter_hand"],
    "inning": ["inning"],
    "base": ["base_state"],
    "outs": ["outs_before"],
}
MIN_N = 2000          # 이보다 작은 셀은 전체 평균으로 되돌린다


def load(tag):
    fs = sorted(glob.glob(f"./out/*{tag}_s*_test_preds.npz"))
    if not fs:
        return None
    z = [np.load(f, allow_pickle=True) for f in fs]
    return pd.DataFrame({"row_id": z[0]["row_id"],
                         "y": z[0]["y"].astype(np.float64),
                         "pred": np.mean([q["pred"] for q in z],
                                         0).astype(np.float64)})


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    sa, sb = sys.argv[1], sys.argv[2]
    A, B = load(f"BI{sa}"), load(f"BI{sb}")
    if A is None or B is None:
        print(f"예측 없음 (BI{sa} / BI{sb})")
        return 1

    cols = sorted({c for v in SEGS.values() for c in v})
    src = pd.read_csv("./data/train.csv", usecols=["row_id"] + cols)
    A = A.merge(src, on="row_id", how="left")
    B = B.merge(src, on="row_id", how="left")

    def bss(df, adj=0.0):
        r = df.y.mean()
        p = np.clip(df.pred - adj, 0, 1)
        return 1e5 * (1 - ((p - df.y) ** 2).mean() / (r * (1 - r)))

    gb, ga = float(B.pred.mean() - B.y.mean()), float(A.pred.mean() - A.y.mean())
    print(f"미학습 {sa}: {len(A):,}행 | 전역 편향 {ga:+.5f}")
    print(f"미학습 {sb}: {len(B):,}행 | 전역 편향 {gb:+.5f}")
    print(f"\n{sb} 기준선          {bss(B):8.2f}")
    print(f"  전역 시프트(자기)  {bss(B, gb):8.2f}  ({bss(B, gb) - bss(B):+.2f})")
    print(f"  전역 시프트({sa})  {bss(B, ga):8.2f}  ({bss(B, ga) - bss(B):+.2f})")

    print(f"\n{'세그먼트':<10}{'셀수':>6}{'편향sd':>10}{'자기적합':>11}"
          f"{f'{sa}→{sb}':>11}")
    for name, keys in SEGS.items():
        # 세그먼트 편향은 **전역 편향을 뺀 나머지**만 본다 (전역은 SHIFT 담당)
        bias_a = (A.assign(e=A.pred - A.y).groupby(keys).e.agg(["mean", "size"]))
        bias_a["mean"] = np.where(bias_a["size"] >= MIN_N, bias_a["mean"], ga)
        key_b = B[keys].apply(tuple, axis=1) if len(keys) > 1 else B[keys[0]]
        key_a = bias_a.index
        m = pd.Series(bias_a["mean"].to_numpy(), index=key_a)
        adj_a = key_b.map(m).fillna(ga).to_numpy() - ga      # 잔여 세그먼트 편향

        bias_b = (B.assign(e=B.pred - B.y).groupby(keys).e.agg(["mean", "size"]))
        bias_b["mean"] = np.where(bias_b["size"] >= MIN_N, bias_b["mean"], gb)
        mb = pd.Series(bias_b["mean"].to_numpy(), index=bias_b.index)
        adj_b = key_b.map(mb).fillna(gb).to_numpy() - gb

        self_fit = bss(B, gb + adj_b) - bss(B, gb)
        transfer = bss(B, gb + adj_a) - bss(B, gb)
        print(f"{name:<10}{len(bias_a):>6}{adj_b.std():>10.5f}"
              f"{self_fit:>+11.2f}{transfer:>+11.2f}")
    print("\n※ '자기적합'은 상한(낙관), 판정은 마지막 열이다. "
          "양수가 아니면 그 축은 시즌을 못 넘는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
