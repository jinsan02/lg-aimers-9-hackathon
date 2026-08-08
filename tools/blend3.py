"""미학습 표면에서 **여러 멤버 동시 최적화** (E163).

셀 계열이 깊이 4·5·6 전부 base 보다 강한 것으로 나왔다(D −9 ~ −13.5).
그런데 셀끼리는 서로 얼마나 다른가? 같은 라벨·같은 피처·같은 알고리즘이라
서로 매우 닮았을 수 있고, 그러면 둘을 넣어도 하나만큼밖에 안 준다.

멤버 간 rms 행렬을 먼저 찍고, NNLS 로 동시 가중을 낸다.
가중은 **미학습 시즌 자기적합**이라 절대값은 낙관 — 조합 간 **차이**만 본다.

실행: python tools/blend3.py AB_base DW_cell CZ_d6 CZ_d4
"""

import glob
import itertools
import sys

import numpy as np


def load(tag):
    fs = sorted(glob.glob(f"./out/*_{tag}_s*_test_preds.npz"))
    z = [np.load(f, allow_pickle=True) for f in fs]
    if not z:
        return None, None
    return (np.mean([q["pred"] for q in z], 0).astype(np.float64),
            z[0]["y"].astype(np.float64))


def main():
    tags = sys.argv[1:]
    if len(tags) < 2:
        print(__doc__)
        return 1
    P, y = [], None
    keep = []
    for t in tags:
        p, yy = load(t)
        if p is None:
            print(f"없음: {t}")
            continue
        P.append(p)
        keep.append(t)
        y = yy
    r = float(y.mean())
    base = r * (1 - r)

    def sc(p):
        p = p - (p.mean() - r)
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)

    C = [p - (p.mean() - r) for p in P]      # 수준 제거 후 비교
    print(f"미학습 {len(y):,}행\n")
    print(f"{'멤버':<12}{'단독':>9}")
    for t, p in zip(keep, P):
        print(f"{t:<12}{sc(p):>9.2f}")
    print(f"\n멤버 간 rms (수준 제거 후)")
    print(f"{'':<12}" + "".join(f"{t[:9]:>10}" for t in keep))
    for i, t in enumerate(keep):
        row = "".join(f"{float(np.sqrt(((C[i] - C[j]) ** 2).mean())):>10.4f}"
                      for j in range(len(keep)))
        print(f"{t:<12}{row}")

    from scipy.optimize import nnls
    M = np.column_stack(C)
    big = 1e3 * np.abs(y).mean()
    A = np.vstack([M, np.full((1, M.shape[1]), big)])
    w, _ = nnls(A, np.concatenate([y, [big]]))
    w = w / w.sum()
    print(f"\n동시 최적 가중 (NNLS, 자기적합이라 낙관)")
    for t, v in zip(keep, w):
        print(f"  {t:<12}{v:>7.3f}")
    print(f"  → 블렌드 {sc(M @ w):.2f}  (base 단독 {sc(P[0]):.2f}, "
          f"{sc(M @ w) - sc(P[0]):+.2f})")

    print(f"\n부분집합 비교 (base 는 항상 포함)")
    idx = list(range(1, len(keep)))
    for k in range(1, len(idx) + 1):
        for combo in itertools.combinations(idx, k):
            sel = [0] + list(combo)
            Ms = M[:, sel]
            As = np.vstack([Ms, np.full((1, len(sel)), big)])
            ws, _ = nnls(As, np.concatenate([y, [big]]))
            if ws.sum() <= 0:
                continue
            ws = ws / ws.sum()
            nm = "+".join(keep[i] for i in sel)
            print(f"  {nm:<34}{sc(Ms @ ws):>9.2f}  "
                  f"{sc(Ms @ ws) - sc(P[0]):>+7.2f}  w="
                  + ",".join(f"{v:.2f}" for v in ws))
    print("\n※ 멤버를 더해도 점수가 거의 안 오르면 그 멤버는 이미 있는 것과 닮았다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
