"""ABS 도입(KBO 2024)이 **타깃의 정의**를 바꿨는가 (E160).

제구 실패는 셋이다: ① 존 중앙 ② 존 밖 ③ 포수 요구 반대.
이 중 ②는 '스트라이크존 밖'인데, KBO 는 2024 에 세계 최초로 1군에 ABS(자동 볼판정)를
도입했다. 그 전까지 존은 심판 판단이었고 2024 부터는 **기계가 정한 3D 도형**이다.
즉 ②의 라벨 의미가 2024 에서 바뀌었을 수 있다.

F리그 라벨 체제 변화(2022 0.71 -> 2023 0.47)를 못 보고 학습·검증을 가로질렀다가
조기종료가 15 iter 에서 죽은 적이 있다. 같은 종류의 함정이 **타깃 안에** 있는지 본다.

실패모드 라벨은 asof 한 투구 차분으로 복원한다 (train 전용, failmode.py).

실행: python tools/abs_check.py
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")


def main():
    import failmode as fm
    cols = ["row_id", "season", "game_type", "control_success", "pitcher_id",
            "asof_pitcher_n", "asof_pitcher_middle_rate",
            "asof_pitcher_ball_rate", "asof_pitcher_reverse_rate",
            "asof_pitcher_strike_rate"]
    df = pd.read_csv("./data/train.csv", usecols=cols)
    lab = fm._pitch_labels(df, modes=("middle", "ball", "reverse", "strike"))
    df = pd.concat([df, lab], axis=1)
    R = df[df.game_type == "R"]

    print("정규리그(R) 시즌별 기저율 — 복원 라벨\n")
    print(f"{'시즌':>6}{'행수':>10}{'성공':>9}{'실투(중앙)':>12}"
          f"{'존밖':>9}{'반대':>9}{'스트라이크':>12}{'복원률':>9}")
    prev = None
    for s, g in R.groupby("season"):
        ok = g[["middle", "ball", "reverse", "strike"]].notna().all(1)
        row = [g.control_success.mean()] + [g[m].mean() for m in
                                            ("middle", "ball", "reverse", "strike")]
        line = (f"{s:>6}{len(g):>10,}" + "".join(f"{v:>9.4f}" if i != 4
                                                 else f"{v:>12.4f}"
                                                 for i, v in enumerate(
                                                     [row[0], row[1], row[2],
                                                      row[3], row[4]]))
                + f"{ok.mean():>9.1%}")
        print(line)
        if prev is not None:
            d = [row[i] - prev[i] for i in range(5)]
            print(f"{'':>16}" + "".join(f"{v:>+9.4f}" if i != 4
                                        else f"{v:>+12.4f}"
                                        for i, v in enumerate(d)))
        prev = row
    print("\n※ 2023→2024 의 변화가 다른 해보다 크면 ABS 가 라벨 의미를 바꾼 것이다.")
    print("   특히 '존밖'(②)은 2024 부터 기계가 정한 존 기준이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
