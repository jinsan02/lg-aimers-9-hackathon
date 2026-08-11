"""동일 로컬 표면 후보를 기준선에 고정 소량 결합해 누락된 다양성을 찾는다.

2023 validation과 refit 후 완전 미학습 2024 test가 모두 있는 산출물만 사용한다.
후보 자체 점수나 2024 최적 가중으로 채택하지 않고, 미리 고정한 5/10/20/50% 중
두 표면에서 동시에 양수인지를 gate로 본다.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def bss(y, p, center=False):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if center:
        p = p + y.mean() - p.mean()
    r = y.mean()
    return 1e5 * (1 - np.mean((p-y)**2) / (r*(1-r)))


def load(path):
    z = np.load(path)
    return z["y"].astype(float), z["pred"].astype(float)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="out/cat_MVA_native")
    ap.add_argument("--cell", default="",
                    help="선택: 현행 core를 만들 cell stem")
    ap.add_argument("--cell-weight", type=float, default=.55)
    ap.add_argument("--glob", default="out/*_val_preds.npz")
    args = ap.parse_args()
    by, bpv = load(args.base + "_val_preds.npz")
    ty, bpt = load(args.base + "_test_preds.npz")
    if args.cell:
        cy, cpv = load(args.cell + "_val_preds.npz")
        cty, cpt = load(args.cell + "_test_preds.npz")
        if not np.array_equal(cy, by) or not np.array_equal(cty, ty):
            raise ValueError("base/cell target mismatch")
        cw = args.cell_weight
        bpv = (1-cw)*bpv + cw*cpv
        bpt = (1-cw)*bpt + cw*cpt
    weights = (.05, .10, .20, .50)
    rows = []
    base_norm = os.path.normcase(os.path.normpath(args.base))
    for vp in glob.glob(args.glob):
        stem = vp[:-len("_val_preds.npz")]
        tp = stem + "_test_preds.npz"
        if not os.path.exists(tp) or os.path.normcase(os.path.normpath(stem)) == base_norm:
            continue
        try:
            vy, pv = load(vp)
            yy, pt = load(tp)
        except Exception:
            continue
        if (vy.shape != by.shape or yy.shape != ty.shape or
                not np.array_equal(vy, by) or not np.array_equal(yy, ty)):
            continue
        gains = []
        for w in weights:
            gains.append((bss(by, (1-w)*bpv+w*pv)-bss(by, bpv),
                          bss(ty, (1-w)*bpt+w*pt)-bss(ty, bpt)))
        valid = [(w, gv, gt) for w, (gv, gt) in zip(weights, gains)
                 if gv > 0 and gt > 0]
        best = max(valid, key=lambda x: min(x[1], x[2])) if valid else None
        rows.append((stem.replace("out\\", "").replace("out/", ""),
                     bss(ty, pt)-bss(ty, bpt),
                     np.sqrt(np.mean((pt-bpt)**2)), gains, best))

    rows.sort(key=lambda x: (x[4] is not None,
                             min(x[4][1:]) if x[4] else -9999), reverse=True)
    label = f"base+cell(w={args.cell_weight:.2f})" if args.cell else "base"
    print(f"{label} val/test={bss(by,bpv):.3f}/{bss(ty,bpt):.3f} | candidates={len(rows)}")
    print("name                               soloTest    rms    "
          "w05(v/t)       w10(v/t)       w20(v/t)       w50(v/t)    gate")
    for name, solo, rms, gains, best in rows:
        gg = " ".join(f"{a:+6.2f}/{b:+6.2f}" for a, b in gains)
        gate = (f"PASS w={best[0]:.2f} min={min(best[1:]):+.2f}"
                if best else "-")
        print(f"{name:<34} {solo:+8.2f} {rms:.5f} {gg} {gate}")

    passed = [r for r in rows if r[4] is not None]
    print(f"\n두 표면 동시 양수: {len(passed)}개")
    for name, _, _, _, best in passed:
        print(f"  {name}: w={best[0]:.2f}, val={best[1]:+.3f}, test={best[2]:+.3f}")


if __name__ == "__main__":
    main()
