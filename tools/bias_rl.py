"""1시즌 앞 편향을 **리그별로** 분리해서 본다 (E152b).

drift_seg 에서 전역 편향이 2023 −0.00500 / 2024 +0.00282 로 **부호가 뒤집혔다.**
그대로 믿으면 SHIFT 는 동전던지기가 된다.

그런데 BI 실행은 `--drop-f-pre 2022` 를 쓴다 — 학습에서 F리그가 사실상 다 빠지는데
평가 시즌(2023, 2024)에는 F 가 11% 남아 있다. 학습에 없던 리그를 평가에 넣은
셈이라 편향이 오염된다. 실제 제출은 F 를 안 빼므로 이 오염은 제출에 없다.

그래서 **R리그만으로** 다시 잰다. 이게 제출 구조에 가장 가까운 수치다.

실행: python tools/bias_rl.py 2022 2023 2024
"""

import glob
import sys

import numpy as np
import pandas as pd


def load(tag):
    fs = sorted(glob.glob(f"./out/*{tag}_s*_test_preds.npz"))
    if not fs:
        return None
    # row_id 저장은 오늘 추가했다 — 그 전에 만든 npz 는 건너뛴다
    z = [q for q in (np.load(f, allow_pickle=True) for f in fs)
         if "row_id" in q.files]
    if not z:
        return None
    return pd.DataFrame({"row_id": z[0]["row_id"],
                         "y": z[0]["y"].astype(np.float64),
                         "pred": np.mean([q["pred"] for q in z],
                                         0).astype(np.float64)}), len(fs)


def main():
    seasons = sys.argv[1:] or ["2021", "2022", "2023", "2024"]
    src = pd.read_csv("./data/train.csv", usecols=["row_id", "game_type"])
    print(f"{'시즌':>6}{'시드':>5}{'전체 편향':>12}{'R리그 편향':>12}"
          f"{'F 비율':>9}{'R행수':>10}")
    for s in seasons:
        got = load(f"BI{s}")
        if got is None:
            print(f"{s:>6}   (예측 없음)")
            continue
        df, n = got
        df = df.merge(src, on="row_id", how="left")
        allb = float(df.pred.mean() - df.y.mean())
        R = df[df.game_type == "R"]
        rb = float(R.pred.mean() - R.y.mean()) if len(R) else float("nan")
        print(f"{s:>6}{n:>5}{allb:>+12.5f}{rb:>+12.5f}"
              f"{1 - len(R) / len(df):>9.1%}{len(R):>10,}")
    print("\n※ R리그 편향의 **부호가 시즌마다 일정한지**가 SHIFT 의 존재 근거다.")
    print("   일정하지 않으면 c>0 은 기대이득이 음수다 — 상수 시프트를 빼야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
