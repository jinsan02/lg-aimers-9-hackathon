"""코드 리뷰 — 셀 멤버 가중 0.38 이 **수준 보정을 대신하고 있지 않은가**.

decide_v15.sweep() 은 원점수(bss, 편향제거 없음)로 w 를 골랐다. 그런데 두 멤버의
예측 평균이 다르면, w 를 키우는 것만으로 블렌드 평균이 내려간다. 2024 홀드아웃은
모델이 과대예측하는 쪽이라 **평균을 내리는 것 자체가 원점수를 올린다** —
판별력이 좋아져서가 아니다.

그리고 그 수준 보정은 뒤에 오는 SHIFT 가 이미 맡는 일이다. 즉 같은 이득을
두 번 세고, w 는 그만큼 과대 선택된다.

여기서는 **편향제거 후**로 다시 골라 두 값을 비교한다. 차이가 크면 v15 의 0.38 은
잘못 고른 값이다.

실행: python tools/audit_w.py
"""

import glob

import numpy as np


def load(pat):
    fs = sorted(glob.glob(f"./out/*{pat}_val_preds.npz"))
    z = [np.load(f) for f in fs]
    return (np.mean([q["pred"] for q in z], 0).astype(np.float64),
            z[0]["y"].astype(np.float64), len(fs))


def main():
    pf, y, nf = load("v14f_s*")
    pc, _, nc = load("ZD5_s*")
    r = float(y.mean())
    base = r * (1 - r)

    def raw(p):
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)

    def cen(p):                       # 수준은 SHIFT 가 따로 맡는다
        return raw(p - (p.mean() - r))

    print(f"2024 {len(y):,}행 | 실제 {r:.4f}")
    print(f"  base {nf}시드  평균 {pf.mean():.4f} (편향 {pf.mean() - r:+.5f})  "
          f"원점수 {raw(pf):.2f}  편향제거 {cen(pf):.2f}")
    print(f"  cell {nc}시드  평균 {pc.mean():.4f} (편향 {pc.mean() - r:+.5f})  "
          f"원점수 {raw(pc):.2f}  편향제거 {cen(pc):.2f}")
    print(f"  두 멤버 평균 차 {pf.mean() - pc.mean():+.5f}\n")

    ws = np.round(np.arange(0.0, 0.81, 0.05), 2)
    vr = np.array([raw((1 - w) * pf + w * pc) for w in ws])
    vc = np.array([cen((1 - w) * pf + w * pc) for w in ws])
    print(f"{'w':>6}{'원점수':>10}{'Δ원':>8}{'편향제거':>11}{'Δ편향제거':>11}")
    for w, a, b in zip(ws, vr, vc):
        print(f"{w:>6.2f}{a:>10.2f}{a - vr[0]:>+8.2f}{b:>11.2f}{b - vc[0]:>+11.2f}")
    print(f"\n원점수 최적 w {ws[vr.argmax()]:.2f} (이득 {vr.max() - vr[0]:+.2f})")
    print(f"편향제거 최적 w {ws[vc.argmax()]:.2f} (이득 {vc.max() - vc[0]:+.2f})")
    print(f"현행 채택 0.38 → 편향제거 이득 "
          f"{cen(0.62 * pf + 0.38 * pc) - vc[0]:+.2f}")
    print("\n※ 두 최적이 크게 다르면 w 가 수준 보정을 대신 하고 있었다는 뜻이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
