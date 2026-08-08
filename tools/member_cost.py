"""멤버가 **무게만큼 값을 하는가** — 시드 수·가중을 비용 대비로 검토한다.

v14c(셀 다중분류)는 모델당 58MB 로 이진 멤버(5MB)의 11.5배다. 트리 4,400개 x
10클래스라 추론 비용도 그만큼 든다. 추론 10분 한도가 있으므로 "이득이 있다"만으로는
부족하고 **단위 비용당 이득**을 봐야 한다.

여기서 답하는 것:
  ① 시드를 6 -> 3 -> 1 로 줄이면 margin 이 얼마나 죽는가
     (가중이 0.20 이라 시드 평균 효과가 작을 수 있다 = 무료로 절반을 아낀다)
  ② 가중 w 를 다시 최적화하면 얼마나 달라지는가
  ③ 이진 멤버 시드를 늘리는 것과 비교하면 어느 쪽이 MB 당 이득이 큰가

실행: python tools/member_cost.py
"""

import glob
import sys

import numpy as np

MB_BIN = 5.09      # 이진 멤버 1개 (MB)
MB_CELL = 58.47    # 셀 다중분류 멤버 1개 (MB)


def preds(pat):
    fs = sorted(glob.glob(f"./out/*{pat}_val_preds.npz"))
    return [np.load(f)["pred"].astype(np.float64) for f in fs], fs


def main():
    F, ff = preds("v14f_s*")
    C, cf = preds("v14c_s*")
    if not F or not C:
        print("v14f / v14c 예측이 없다")
        return 1
    y = np.load(ff[0])["y"].astype(np.float64)
    r = y.mean()
    base = r * (1 - r)

    def bss(p):
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)

    pf = np.mean(F, 0)
    print(f"이진 멤버 {len(F)}개 ({len(F) * MB_BIN:.0f}MB) 단독 {bss(pf):.2f}")

    def best_w(p1, p2):
        A = float(((p1 - p2) ** 2).mean())
        if A <= 0:
            return 0.0, 0.0, 0.0
        m = 1e5 * 2 * float(((p1 - y) * (p1 - p2)).mean()) / base
        d = 1e5 * A / base
        return float(np.clip(m / (2 * d), 0, 1)), m, d

    print("\n=== ① 셀 멤버 시드 수를 줄이면 ===")
    print(f"{'시드':>4}{'MB':>7}{'단독':>9}{'rms':>8}{'margin':>9}"
          f"{'w*':>7}{'w=0.20 이득':>12}{'MB당':>9}")
    for n in (1, 2, 3, 4, 6):
        if n > len(C):
            continue
        pc = np.mean(C[:n], 0)
        w, m, d = best_w(pf, pc)
        g20 = bss(0.8 * pf + 0.2 * pc) - bss(pf)
        mb = n * MB_CELL
        print(f"{n:>4}{mb:>7.0f}{bss(pc):>9.2f}"
              f"{np.sqrt(d * base / 1e5):>8.4f}{m:>+9.1f}{w:>7.3f}"
              f"{g20:>12.2f}{g20 / mb:>9.4f}")

    print("\n=== ② 같은 MB 를 이진 멤버 시드에 쓰면 (비교군) ===")
    for n in (2, 4, 6, 8):
        if n > len(F):
            continue
        p = np.mean(F[:n], 0)
        print(f"  이진 {n}시드 ({n * MB_BIN:>5.0f}MB) {bss(p):>8.2f}"
              f"  8시드 대비 {bss(p) - bss(pf):+6.2f}")
    print(f"  -> 이진 시드 확장은 8개에서 이미 포화다. 셀 멤버 1개(58MB)는")
    print(f"     이진 11.5개분 용량이지만 그만큼의 시드 확장은 이득이 0 이다.")

    print("\n=== ③ 가중 민감도 (셀 6시드 기준) ===")
    pc = np.mean(C, 0)
    for w in (0.10, 0.15, 0.20, 0.25, 0.30):
        print(f"  w={w:.2f}  {bss((1 - w) * pf + w * pc) - bss(pf):+6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
