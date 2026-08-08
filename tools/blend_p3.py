"""P3 — 3멤버(base / cell / 증류) 가중을 2024 자기검증 표면에서 고른다.

w 는 **편향제거 후** 점수로 고른다. 파이프라인이 블렌드 뒤에 전역 SHIFT 를 걸어
수준을 따로 처리하기 때문이다 (redecide_w.py 와 같은 규약).

실행: python tools/blend_p3.py
"""

import glob

import numpy as np

SEEDS_SHIP = ["42", "7", "13", "3", "4", "5"]      # ZD5 제출 시드
R = None


def load(tag, seeds=None):
    fs = sorted(glob.glob(f"./out/cat_{tag}_s*_val_preds.npz"))
    if seeds is not None:
        fs = [f for f in fs
              if f.split("_s")[-1].split("_")[0] in seeds]
    if not fs:
        return None, None, 0
    z = [np.load(f) for f in fs]
    return (np.mean([q["pred"] for q in z], 0).astype(np.float64),
            z[0]["y"].astype(np.float64), len(fs))


def bss(p, y, r, base):
    return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)


def cen(p, y, r, base):
    return bss(p - (p.mean() - r), y, r, base)


def main():
    b, y, nb = load("VB_base")
    c, _, nc = load("ZD5", SEEDS_SHIP)
    c8, _, nc8 = load("ZD5")
    d, _, nd = load("DX_seq")
    for nm, v in (("VB_base", b), ("ZD5", c), ("DX_seq", d)):
        if v is None:
            print(f"없음: {nm}")
            return 1
    r = float(y.mean())
    base = r * (1 - r)
    print(f"검증 2024  {len(y):,}행  r={r:.4f}")
    print(f"멤버 시드수: base {nb} / cell {nc}(제출)·{nc8}(전체) / 증류 {nd}\n")

    for nm, v in (("base ", b), ("cell ", c), ("cell8", c8), ("dist ", d)):
        print(f"  {nm}  원점수 {bss(v, y, r, base):8.2f}   "
              f"편향제거 {cen(v, y, r, base):8.2f}   평균 {v.mean():.4f}")

    print("\n=== 2멤버 현행 재현 (base+cell) ===")
    for w in (0.45, 0.50, 0.55, 0.60, 0.65):
        print(f"  cell {w:.2f}   편향제거 {cen((1-w)*b + w*c, y, r, base):8.2f}")

    print("\n=== 3멤버 격자 (행=cell, 열=dist, 편향제거) ===")
    ws = np.round(np.arange(0.0, 0.81, 0.05), 2)
    best = (-1e9, 0, 0)
    hdr = "      " + "".join(f"{w:>8.2f}" for w in ws[:13])
    print(hdr)
    for wc in ws[:13]:
        row = f"{wc:>5.2f} "
        for wd in ws[:13]:
            if wc + wd > 1.0:
                row += "       ."
                continue
            s = cen((1 - wc - wd) * b + wc * c + wd * d, y, r, base)
            if s > best[0]:
                best = (s, wc, wd)
            row += f"{s:>8.1f}"
        print(row)
    s, wc, wd = best
    print(f"\n최적 (자기적합·낙관):  base {1-wc-wd:.2f} / cell {wc:.2f} / dist {wd:.2f}"
          f"   편향제거 {s:.2f}")
    cur = cen(0.45 * b + 0.55 * c, y, r, base)
    print(f"현행 2멤버 0.45/0.55:  {cur:.2f}      차이 {s - cur:+.2f}")
    print(f"\nLB 환산(편향제거 + 128.4):  현행 {cur + 128.4:.1f} -> 신규 {s + 128.4:.1f}")
    print("※ 자기적합 최적이라 낙관치다. 실제 이득은 이보다 작다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
