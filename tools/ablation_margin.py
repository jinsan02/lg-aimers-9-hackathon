"""절제 모델들을 **블렌드 멤버**로 재채점한다 (GPU 0).

지금까지 절제(LOO)는 "그 레버를 빼면 얼마나 나빠지나"만 봤다. 그건 **교체** 기준이다.
블렌드 **멤버** 기준은 다르다:

    Dmax   = 1e5 A / base = 400,320 x rms^2
    margin = Dmax - D                     (D = 기준 대비 성능 격차)
    이득    = margin^2 / (4 Dmax)

즉 **약해도(D 큼) 계통적으로 다르면(rms 큼) 값을 한다.** 피처군을 하나씩 뺀 모델은
정의상 '같은 데이터·같은 알고리즘·다른 귀납편향'이라 이 조건에 맞을 수 있다.
한 번도 이 기준으로 본 적이 없다.

특히 note(TE 제거)와 nostd(시즌내 복원 제거)는 신호의 큰 축을 통째로 뺀 것이라
예측이 크게 달라진다 — rms 가 크면 D 가 커도 margin 이 양수일 수 있다.

실행: python tools/ablation_margin.py A24
"""

import glob
import re
import sys

import numpy as np


def load(pat):
    fs = sorted(glob.glob(f"./out/*{pat}_val_preds.npz"))
    if not fs:
        return None, None
    z = [np.load(f) for f in fs]
    return (np.mean([q["pred"] for q in z], 0).astype(np.float64),
            z[0]["y"].astype(np.float64))


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "A24"
    p1, y = load(f"{prefix}_full_s*")
    if p1 is None:
        print(f"{prefix}_full 예측 없음")
        return 1
    r = y.mean()
    base = r * (1 - r)

    def bss(p):
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)

    def deb(p):
        return p - (p.mean() - r)          # 수준 편향 제거 (LEVERS 측정 원칙)

    p1 = deb(p1)
    m0 = bss(p1)
    print(f"{prefix} 기준 {m0:.2f} (편향제거) | {len(y):,}행\n")
    print(f"{'절제':<10}{'단독':>9}{'D':>8}{'rms':>8}{'Dmax':>9}"
          f"{'margin':>9}{'w*':>7}{'이득':>8}")

    arms = sorted({re.search(rf"{prefix}_(\w+?)_s\d+_val", f.replace("\\", "/")).group(1)
                   for f in glob.glob(f"./out/*{prefix}_*_val_preds.npz")
                   if re.search(rf"{prefix}_(\w+?)_s\d+_val", f.replace("\\", "/"))})
    rows = []
    for a in arms:
        if a == "full":
            continue
        p2, _ = load(f"{prefix}_{a}_s*")
        if p2 is None or len(p2) != len(p1):
            continue
        p2 = deb(p2)
        A = float(((p1 - p2) ** 2).mean())
        if A <= 0:
            continue
        d = 1e5 * A / base
        mg = 1e5 * 2 * float(((p1 - y) * (p1 - p2)).mean()) / base
        g = mg * mg / (4 * d) if mg > 0 else 0.0
        w = float(np.clip(mg / (2 * d), 0, 1))
        rows.append((a, bss(p2), m0 - bss(p2), np.sqrt(A), d, mg, w, g))
    for a, s, D, rms, d, mg, w, g in sorted(rows, key=lambda x: -x[7]):
        print(f"{a:<10}{s:>9.2f}{D:>8.1f}{rms:>8.4f}{d:>9.1f}"
              f"{mg:>+9.1f}{w:>7.3f}{g:>8.2f}")

    # 양수인 것들을 **함께** 넣으면 (서로 다를수록 합이 커진다)
    pos = [a for a, *_ , g in rows if g > 0.2]
    if len(pos) > 1:
        print(f"\n양수 멤버 {pos} 를 동시 블렌드")
        best = (0.0, None)
        for wtot in (0.1, 0.2, 0.3, 0.4, 0.5):
            mix = np.mean([deb(load(f"{prefix}_{a}_s*")[0]) for a in pos], 0)
            v = bss((1 - wtot) * p1 + wtot * mix) - m0
            if v > best[0]:
                best = (v, wtot)
            print(f"  w={wtot:.1f}  {v:+6.2f}")
        print(f"  -> 최선 w={best[1]} 이득 {best[0]:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
