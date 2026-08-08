"""두 설정을 **같은 홀드아웃 시드**로 페어 비교한다.

오늘 O1(-7.44)이 가르친 것: 탐색·개발에 쓴 시드로 채점하면 부호까지 틀린다.
그래서 비교는 항상 (a) 시드를 짝지어 빼고 (b) t 값을 같이 본다.
채택 기준 t >= 2.4 / 기각 기준 95% 상한 < +3.

실행: python tools/cmp_tags.py cat_v14f cat_TM1 3,4,5,6,8,13
"""

import sys

import numpy as np


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    a, b, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split(",")

    def load(tag, s):
        return np.load(f"./out/{tag}_s{s}_val_preds.npz")

    y = load(a, seeds[0])["y"].astype(np.float64)
    r = float(y.mean())
    base = r * (1 - r)

    def raw(p):
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)

    def sc(p):
        return raw(p - (p.mean() - r))     # 수준은 SHIFT 가 따로 잡는다

    A = {s: load(a, s)["pred"].astype(np.float64) for s in seeds}
    B = {s: load(b, s)["pred"].astype(np.float64) for s in seeds}
    print(f"{'시드':>5}{a[-8:]:>12}{b[-8:]:>12}{'차이':>9}")
    d = []
    for s in seeds:
        x, z = sc(A[s]), sc(B[s])
        d.append(z - x)
        print(f"{s:>5}{x:>12.2f}{z:>12.2f}{z - x:>+9.2f}")
    d = np.array(d)
    se = d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
    print(f"\n페어 평균 {d.mean():+.2f} | SE {se:.2f} | t {d.mean() / se:+.2f} "
          f"| 95%CI [{lo:+.2f}, {hi:+.2f}]")

    pa = np.mean([A[s] for s in seeds], 0)
    pb = np.mean([B[s] for s in seeds], 0)
    print(f"{len(seeds)}시드 앙상블  {sc(pa):8.2f} -> {sc(pb):8.2f} "
          f"({sc(pb) - sc(pa):+.2f})")
    # D 를 편향제거 후로 재므로 rms 도 같은 좌표에서 재야 한다. 원본으로 재면
    # 멤버 간 **수준 차이가 rms 에만** 들어가 Dmax 가 부풀고 이득이 과대평가된다.
    ca, cb = pa - (pa.mean() - r), pb - (pb.mean() - r)
    rms = float(np.sqrt(((ca - cb) ** 2).mean()))
    dmax = 400320 * rms ** 2
    dd = sc(pa) - sc(pb)
    print(f"블렌드 여지: rms {rms:.4f} | Dmax {dmax:.1f} | D {dd:+.1f} | "
          f"margin {dmax - dd:+.1f} | 이득 "
          f"{max(0.0, dmax - dd) ** 2 / (4 * dmax) if dmax > 0 else 0:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
