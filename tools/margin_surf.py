"""블렌드 margin 을 **미학습 시즌 표면**에서 잰다 (E155b).

지금까지 margin = 400,320 x rms^2 - D 의 D 를 2024 **자기검증** 표면에서 쟀다.
그 표면에서는 모든 모델이 2024 로 조기종료되므로 베이스가 그 시즌에 맞춘 이점이
D 에 그대로 들어간다. 시즌이 바뀌면 모델 간 격차는 압축되는 게 정상이고,
D 가 줄면 margin 은 그만큼 커진다.

이 도구는 같은 계산을 test_preds(미학습 시즌 예측)로 한다.

실행: python tools/margin_surf.py RN1.5 DV_sub7 DV_cell
"""

import glob
import sys

import numpy as np


def load(tag):
    fs = sorted(glob.glob(f"./out/*{tag}_s*_test_preds.npz"))
    if not fs:
        return None, None, 0
    z = [np.load(f, allow_pickle=True) for f in fs]
    return (np.mean([q["pred"] for q in z], 0).astype(np.float64),
            z[0]["y"].astype(np.float64), len(fs))


def main():
    tags = sys.argv[1:]
    if len(tags) < 2:
        print(__doc__)
        return 1
    pb, y, nb = load(tags[0])
    if pb is None:
        print(f"기준 없음: {tags[0]}")
        return 1
    r = float(y.mean())
    base = r * (1 - r)

    def center(p):
        return p - (p.mean() - r)       # 수준은 블렌드 뒤 SHIFT 가 따로 맡는다

    def bss(p):
        p = center(p)
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)

    b0 = bss(pb)
    print(f"기준 {tags[0]} ({nb}시드)  {b0:.2f}   미학습 {len(y):,}행\n")
    print(f"{'멤버':<12}{'시드':>5}{'단독':>9}{'D':>8}{'rms':>9}"
          f"{'Dmax':>8}{'margin':>9}{'최적w':>8}{'이득':>8}")
    for t in tags[1:]:
        p, _, n = load(t)
        if p is None:
            print(f"{t:<12}  (없음)")
            continue
        s = bss(p)
        d = b0 - s
        # D 는 편향제거 후로 재는데 rms 를 원본으로 재면 **수준 차이가 rms 에만**
        # 들어가 Dmax 가 부풀고 margin/이득이 과대평가된다. 같은 좌표에서 잰다.
        rms = float(np.sqrt(((center(pb) - center(p)) ** 2).mean()))
        dmax = 400320 * rms ** 2
        margin = dmax - d
        w = margin / (2 * dmax) if dmax > 0 else 0.0
        gain = margin ** 2 / (4 * dmax) if (dmax > 0 and margin > 0) else 0.0
        print(f"{t:<12}{n:>5}{s:>9.2f}{d:>+8.1f}{rms:>9.4f}{dmax:>8.1f}"
              f"{margin:>+9.1f}{max(w, 0):>8.3f}{gain:>8.2f}")
    print("\n※ margin>0 이어야 블렌드에 기여한다. 자기검증 표면에서 잰 값과")
    print("   비교하면 D 를 얼마나 부풀려 재고 있었는지가 드러난다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
