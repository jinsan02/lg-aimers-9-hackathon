"""코드 리뷰 — drift_seg.py 의 다중키 매핑이 실제로 붙었는가.

`bias_a` 는 여러 키로 groupby 해서 **MultiIndex** 가 되고, `key_b` 는 튜플 Series 다.
`Series.map(MultiIndex Series)` 가 조용히 NaN 을 내면 fillna(전역편향) 으로
보정값이 전부 0 이 되고, 그러면 "시즌을 못 넘는다"가 아니라 **아무것도 안 한 것**이
된다. 세그먼트 드리프트를 기각한 근거였으므로 확인해야 한다.

실행: python tools/audit_seg.py
"""

import glob

import numpy as np
import pandas as pd

SEGS = {"count": ["balls_before", "strikes_before"],
        "hand": ["pitcher_hand", "batter_hand"],
        "inning": ["inning"], "base": ["base_state"],
        "outs": ["outs_before"]}


def load(tag):
    fs = sorted(glob.glob(f"./out/*{tag}_s*_test_preds.npz"))
    z = [q for q in (np.load(f, allow_pickle=True) for f in fs)
         if "row_id" in q.files]
    return pd.DataFrame({"row_id": z[0]["row_id"],
                         "y": z[0]["y"].astype(np.float64),
                         "pred": np.mean([q["pred"] for q in z],
                                         0).astype(np.float64)})


def main():
    A, B = load("BI2023"), load("BI2024")
    cols = sorted({c for v in SEGS.values() for c in v})
    src = pd.read_csv("./data/train.csv", usecols=["row_id"] + cols)
    A = A.merge(src, on="row_id", how="left")
    B = B.merge(src, on="row_id", how="left")
    ga = float(A.pred.mean() - A.y.mean())

    print(f"{'세그먼트':<10}{'셀수':>6}{'매핑성공률':>12}{'보정 sd':>11}"
          f"{'보정 절대최대':>14}")
    for name, keys in SEGS.items():
        bias_a = A.assign(e=A.pred - A.y).groupby(keys).e.agg(["mean", "size"])
        m = pd.Series(bias_a["mean"].to_numpy(), index=bias_a.index)
        key_b = (B[keys].apply(tuple, axis=1) if len(keys) > 1 else B[keys[0]])
        mapped = key_b.map(m)
        hit = float(mapped.notna().mean())
        adj = mapped.fillna(ga).to_numpy() - ga
        flag = "" if hit > 0.95 else "   <-- 매핑 실패"
        print(f"{name:<10}{len(bias_a):>6}{hit:>11.1%}{adj.std():>11.5f}"
              f"{np.abs(adj).max():>14.5f}{flag}")
    print("\n※ 매핑성공률이 낮으면 그 축의 '이전 실패' 판정은 무효다 —")
    print("   보정을 안 한 것과 같기 때문.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
