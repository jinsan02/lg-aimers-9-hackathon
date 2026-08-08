"""손실이 어디서 나는가 — 미학습 시즌 예측을 세그먼트로 쪼갠다 (E162).

## 왜 지금

밤새 축 재판정을 다 돌렸는데 채택된 게 하나도 없다. 하이퍼파라미터를 더 흔드는
것으로는 안 된다는 뜻이다. 그러면 **어디서 지고 있는지**를 보고 거기를 쳐야 한다.

각 세그먼트에서 두 가지를 본다:
  - 그 세그먼트 자체의 BSS (그 세그먼트의 실제 성공률을 기준선으로)
  - 전체 손실에서 그 세그먼트가 차지하는 몫 (행수 x 평균제곱오차)

BSS 가 낮은데 행수가 많은 칸이 곧 남은 먹거리다. 특히 F리그(11%)는 라벨 체제가
2023 에 바뀌었고 학습에 신체제가 2시즌뿐이라 따로 볼 값이 있다.

실행: python tools/loss_map.py RN1.5
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
    tag = sys.argv[1] if len(sys.argv) > 1 else "RN1.5"
    df = load(tag)
    if df is None:
        print(f"예측 없음: {tag}")
        return 1
    cols = ["row_id", "game_type", "balls_before", "strikes_before",
            "asof_pitcher_n", "inning", "pitcher_hand", "batter_hand"]
    src = pd.read_csv("./data/train.csv", usecols=cols)
    df = df.merge(src, on="row_id", how="left")
    df["se"] = (np.clip(df.pred, 0, 1) - df.y) ** 2
    total_se = df.se.sum()

    def seg_table(name, key):
        g = df.groupby(key)
        rows = []
        for k, s in g:
            r = s.y.mean()
            b = r * (1 - r)
            bss = 1e5 * (1 - s.se.mean() / b) if b > 1e-9 else float("nan")
            rows.append((k, len(s), r, bss, s.se.sum() / total_se))
        rows.sort(key=lambda x: -x[4])
        print(f"\n[{name}]  {'칸':<14}{'행수':>9}{'실제율':>8}"
              f"{'그칸 BSS':>10}{'손실몫':>8}")
        for k, n, r, bss, sh in rows[:8]:
            print(f"{'':<18}{str(k):<14}{n:>9,}{r:>8.4f}{bss:>10.1f}{sh:>8.1%}")

    print(f"{tag}: 미학습 {len(df):,}행 | 전체 BSS "
          f"{1e5 * (1 - df.se.mean() / (df.y.mean() * (1 - df.y.mean()))):.1f}")
    seg_table("리그", "game_type")
    df["cnt"] = df.balls_before.astype(str) + "-" + df.strikes_before.astype(str)
    seg_table("볼카운트", "cnt")
    df["exp"] = pd.cut(df.asof_pitcher_n, [-1, 200, 1000, 3000, 10000, 10 ** 9],
                       labels=["~200", "~1k", "~3k", "~10k", "10k+"])
    seg_table("투수 통산 투구수", "exp")
    df["mu"] = df.pitcher_hand.astype(str) + "vs" + df.batter_hand.astype(str)
    seg_table("매치업", "mu")
    print("\n※ '그칸 BSS' 가 낮고 '손실몫' 이 큰 칸이 남은 먹거리다.")
    print("   전체 BSS 보다 크게 낮은 칸은 그 칸 전용 처리가 필요하다는 신호.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
