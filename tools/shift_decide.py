"""SHIFT 를 쓸 근거가 있는가 — 기울기 적용 **후** 편향의 시즌별 분포 (E157).

지금까지의 SHIFT 는 "미래 시즌은 모델이 과대예측한다"를 전제했다. 그런데 배치와
같은 구조(refit -> 미학습 시즌)에서 재보니 부호가 시즌마다 뒤집힌다.

상수 c 의 기대이득은  1e5 (2 E[b] c - c^2) / base  이므로,
E[b] 가 0 근처면 **어떤 c>0 도 기대이득이 음수**다. 손실은 정확히 -1e5 c^2/base.

실행: python tools/shift_decide.py
"""

import glob

import numpy as np
import pandas as pd

SLOPE = 1.0513          # tools/slope_surf.py: 미학습 2024 적합값


def load(tag):
    fs = sorted(glob.glob(f"./out/*{tag}_s*_test_preds.npz"))
    z = [q for q in (np.load(f, allow_pickle=True) for f in fs)
         if "row_id" in q.files]
    if not z:
        return None
    return pd.DataFrame({"row_id": z[0]["row_id"],
                         "y": z[0]["y"].astype(np.float64),
                         "pred": np.mean([q["pred"] for q in z],
                                         0).astype(np.float64)})


def main():
    src = pd.read_csv("./data/train.csv", usecols=["row_id", "game_type"])
    rows = []
    for s in ("2022", "2023", "2024"):
        df = load(f"BI{s}")
        if df is None:
            continue
        df = df.merge(src, on="row_id", how="left")
        z = np.log(np.clip(df.pred, 1e-6, 1 - 1e-6)
                   / (1 - np.clip(df.pred, 1e-6, 1 - 1e-6)))
        q = 1 / (1 + np.exp(-SLOPE * z))
        R = df.game_type == "R"
        rows.append((s, float(q.mean() - df.y.mean()),
                     float(q[R].mean() - df.y[R].mean())))
    print(f"{'시즌':>6}{'기울기후 편향(전체)':>20}{'(R리그)':>12}")
    for s, a, r in rows:
        print(f"{s:>6}{a:>+20.5f}{r:>+12.5f}")
    b = np.array([r for _, _, r in rows])
    print(f"\nE[b] = {b.mean():+.5f}   sd {b.std(ddof=1):.5f}   "
          f"SE {b.std(ddof=1) / np.sqrt(len(b)):.5f}")

    r24 = 0.4861
    base = r24 * (1 - r24)
    print(f"\n{'c':>8}{'기대이득':>12}{'최악(b=최소)':>14}{'최선(b=최대)':>14}")
    for c in (0.0, 0.0010, 0.0016, 0.0026, 0.0033, 0.0052):
        def g(bb):
            return 1e5 * (2 * bb * c - c ** 2) / base
        print(f"{c:>8.4f}{g(b.mean()):>+12.2f}{g(b.min()):>+14.2f}"
              f"{g(b.max()):>+14.2f}")
    print("\n※ E[b] 가 0 근처면 c=0 이 최선이다. 부호가 안 맞을 때의 손실이")
    print("   맞을 때의 이득보다 크기 때문 — 2차항 -c^2 은 항상 손해다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
