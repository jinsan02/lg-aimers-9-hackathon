"""여러 멤버를 **동시에** 넣었을 때의 최적 가중과 이득 (GPU 0).

단일 멤버 공식은 이득 = margin^2/(4 Dmax) 이고 최대치가 Dmax/4 = 100,080 x rms^2 다.
즉 rms 가 이득의 천장을 정한다:

    rms 0.005 -> 최대 +2.5 / 0.011 -> +12.1 / 0.017 -> +28.9 / 0.030 -> +90.1

CatBoost + 우리 피처군 안에서 관측된 최대 rms 는 0.0165(nostd) 인데 D 가 103 이라
실이득이 +0.10 이다. 그래서 **여러 멤버를 합쳤을 때 이득이 개별 합보다 큰가**가
전략을 가른다. 멤버끼리도 서로 다르면(=서로의 잔차를 깎으면) 합이 커진다.

여기서는 음이 아닌 가중으로 Brier 를 직접 최소화한다(닫힌 해 + 사영). 그리고
**같은 시즌에서 가중을 적합하고 같은 시즌에서 보고하면 낙관 편향**이므로,
가중 개수를 표시해 과적합 여지를 함께 본다 (멤버 K개면 자유도 K).

실행: python tools/combine_members.py <기준패턴> <후보1> <후보2> ...
"""

import glob
import sys

import numpy as np


def load(pat):
    fs = sorted(glob.glob(f"./out/*{pat}_val_preds.npz"))
    if not fs:
        return None, None
    z = [np.load(f) for f in fs]
    return (np.mean([q["pred"] for q in z], 0).astype(np.float64),
            z[0]["y"].astype(np.float64))


def solve(P, y):
    """min_w ||P w - y||^2  s.t. w >= 0, sum w = 1.

    멤버끼리 상관이 0.99+ 라 P^T P 의 조건수가 크다. 사영 경사법은 이런 데서
    수천 번을 돌려도 시작점(1/K)에서 거의 안 움직인다 — 실제로 그렇게 나와서
    '조합 이득이 작다'는 잘못된 결론이 나올 뻔했다. NNLS 로 정확히 푼다.
    합=1 은 큰 가중의 등식 행 하나를 덧붙여 강제한다.
    """
    from scipy.optimize import nnls
    big = 1e3 * np.abs(y).mean()
    A = np.vstack([P, np.full((1, P.shape[1]), big)])
    b = np.concatenate([y, [big]])
    w, _ = nnls(A, b)
    s = w.sum()
    return w / s if s > 0 else np.full(P.shape[1], 1.0 / P.shape[1])


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    base_pat, cands = sys.argv[1], sys.argv[2:]
    p1, y = load(base_pat)
    if p1 is None:
        print(f"기준 없음: {base_pat}")
        return 1
    r = y.mean()
    base = r * (1 - r)

    def bss(p):
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)

    def deb(p):
        return p - (p.mean() - r)

    p1 = deb(p1)
    m0 = bss(p1)
    print(f"기준 {base_pat} {m0:.2f} | {len(y):,}행 (편향제거)\n")

    cols, names = [p1], ["base"]
    print(f"{'멤버':<16}{'단독':>9}{'D':>8}{'rms':>8}{'margin':>9}{'단독이득':>10}")
    for c in cands:
        p2, _ = load(c)
        if p2 is None or len(p2) != len(p1):
            print(f"{c:<16}  (없음/불일치)")
            continue
        p2 = deb(p2)
        A = float(((p1 - p2) ** 2).mean())
        d = 1e5 * A / base
        mg = 1e5 * 2 * float(((p1 - y) * (p1 - p2)).mean()) / base
        g = mg * mg / (4 * d) if mg > 0 and d > 0 else 0.0
        print(f"{c:<16}{bss(p2):>9.2f}{m0 - bss(p2):>8.1f}"
              f"{np.sqrt(A):>8.4f}{mg:>+9.1f}{g:>10.2f}")
        cols.append(p2)
        names.append(c)

    if len(cols) < 2:
        return 0
    P = np.column_stack(cols)
    w = solve(P, y)
    tot = bss(P @ w)
    print(f"\n=== 동시 최적화 (가중 {len(cols)}개, 2024 자기적합) ===")
    for n, v in sorted(zip(names, w), key=lambda x: -x[1]):
        if v > 0.005:
            print(f"  {n:<16}{v:6.3f}")
    print(f"  합계 이득 {tot - m0:+.2f}  (개별 이득 단순합 대비 확인용)")
    print("\n※ 가중을 2024 에서 적합하고 2024 에서 보고했으므로 낙관 편향이다.")
    print("   멤버가 늘수록 편향도 커진다 — 실제 채택은 다른 시즌 확인 후에.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
