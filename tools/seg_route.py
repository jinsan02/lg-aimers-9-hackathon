"""세그먼트 전용 모델이 라우팅으로 이득을 주는가 (E166).

`--row-filter` 로 그 구간만 학습한 모델(TS_only)과, 같은 설정으로 전체를 학습한
모델(TS_base)을 **그 구간 행에서만** 맞대본다. 라우팅은 그 행 자신의 컬럼으로
정해지므로 행 독립 원칙을 지킨다.

주의: 세그먼트 BSS 를 손실로 읽지 말 것 — F리그에서 한 번 속았다. 여기서는
**MSE 차이 x 행 비중**으로 전체 점수 기여를 낸다.

실행: python tools/seg_route.py TS_base TS_only strikes_before==2
"""

import glob
import sys

import numpy as np
import pandas as pd


def load(tag):
    fs = sorted(glob.glob(f"./out/*_{tag}_s*_test_preds.npz"))
    z = [q for q in (np.load(f, allow_pickle=True) for f in fs)
         if "row_id" in q.files]
    if not z:
        return None
    return pd.DataFrame({"row_id": z[0]["row_id"],
                         "y": z[0]["y"].astype(np.float64),
                         "pred": np.mean([q["pred"] for q in z],
                                         0).astype(np.float64)})


def main():
    base_tag = sys.argv[1] if len(sys.argv) > 1 else "TS_base"
    seg_tag = sys.argv[2] if len(sys.argv) > 2 else "TS_only"
    expr = sys.argv[3] if len(sys.argv) > 3 else "strikes_before==2"
    B, S = load(base_tag), load(seg_tag)
    if B is None or S is None:
        print(f"예측 없음 ({base_tag}: {B is not None} / "
              f"{seg_tag}: {S is not None})")
        return 1

    src = pd.read_csv("./data/train.csv",
                      usecols=["row_id", "balls_before", "strikes_before"])
    B = B.merge(src, on="row_id", how="left")
    n_all = len(B)
    r = float(B.y.mean())
    base = r * (1 - r)
    mse_all = float(((B.pred - B.y) ** 2).mean())
    print(f"기준 {base_tag}: 미학습 {n_all:,}행 | 전체 BSS "
          f"{1e5 * (1 - mse_all / base):.1f}")

    seg_ids = set(B.query(expr).row_id)
    m = S.row_id.isin(seg_ids)
    S = S[m]
    Bs = B[B.row_id.isin(set(S.row_id))].sort_values("row_id")
    Ss = S.sort_values("row_id")
    assert (Bs.row_id.to_numpy() == Ss.row_id.to_numpy()).all(), "행 정렬 불일치"
    y = Bs.y.to_numpy()
    mb = float(((Bs.pred.to_numpy() - y) ** 2).mean())
    ms = float(((Ss.pred.to_numpy() - y) ** 2).mean())
    share = len(Bs) / n_all
    print(f"\n세그먼트 [{expr}]  {len(Bs):,}행 ({share:.1%})")
    print(f"  {base_tag:<10} MSE {mb:.6f}")
    print(f"  {seg_tag:<10} MSE {ms:.6f}   차이 {ms - mb:+.6f}")
    gain = 1e5 * share * (mb - ms) / base
    print(f"\n  라우팅 시 전체 BSS 기여 {gain:+.2f}")

    # 가중 라우팅(세그먼트 안에서만 블렌드)도 본다 — 전량 교체가 최선이 아닐 수 있다
    best = (0.0, 0.0)
    for w in np.arange(0, 1.01, 0.1):
        p = (1 - w) * Bs.pred.to_numpy() + w * Ss.pred.to_numpy()
        g = 1e5 * share * (mb - float(((p - y) ** 2).mean())) / base
        if g > best[1]:
            best = (w, g)
    print(f"  세그먼트 내 최적 가중 w={best[0]:.1f} → 기여 {best[1]:+.2f}")
    print("\n※ 세그먼트 BSS 가 아니라 **MSE 차이 x 행 비중**으로 냈다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
