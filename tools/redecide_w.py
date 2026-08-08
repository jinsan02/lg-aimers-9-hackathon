"""가중 w 를 **올바른 지표**로 다시 고른다.

decide_v15 는 원점수로 골랐다. 그런데 파이프라인은 블렌드 뒤에 전역 SHIFT 를
적용하므로 수준은 이미 따로 처리된다 — w 는 **편향제거 후** 점수로 골라야 한다.

두 시즌에서 각각 고르고(2023R 결정 / 2024 확인), 오늘 세운 시즌 이전 원칙대로
두 최적의 중점을 쓴다. 재학습은 필요 없다 — script_blend 의 상수 하나다.

실행: python tools/redecide_w.py
"""

import glob

import numpy as np


def load(pat):
    fs = sorted(glob.glob(f"./out/*{pat}_val_preds.npz"))
    if not fs:
        return None, None, 0
    z = [np.load(f) for f in fs]
    return (np.mean([q["pred"] for q in z], 0).astype(np.float64),
            z[0]["y"].astype(np.float64), len(fs))


def curve(pb, pc, y, ws):
    r = float(y.mean())
    base = r * (1 - r)

    def cen(p):
        p = p - (p.mean() - r)
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)
    return np.array([cen((1 - w) * pb + w * pc) for w in ws])


def main():
    ws = np.round(np.arange(0.0, 0.81, 0.05), 2)
    b23, y23, n23b = load("R23B_s*")
    c23, _, n23c = load("R23C_s*")
    b24, y24, n24b = load("v14f_s*")
    c24, _, n24c = load("ZD5_s*")
    for nm, v in (("R23B", b23), ("R23C", c23), ("v14f", b24), ("ZD5", c24)):
        if v is None:
            print(f"없음: {nm}")
            return 1

    v23 = curve(b23, c23, y23, ws)
    v24 = curve(b24, c24, y24, ws)
    w23, w24 = float(ws[v23.argmax()]), float(ws[v24.argmax()])
    print(f"{'w':>6}{'2023R Δ':>11}{'2024 Δ':>11}")
    for w, a, b in zip(ws, v23 - v23[0], v24 - v24[0]):
        mark = ""
        if w == w23:
            mark += "  <-2023R최적"
        if w == w24:
            mark += "  <-2024최적"
        if w == 0.38:
            mark += "  <-현행"
        print(f"{w:>6.2f}{a:>+11.2f}{b:>+11.2f}{mark}")

    W = round((w23 + w24) / 2 / 0.05) * 0.05
    cur = float(np.interp(0.38, ws, v24) - v24[0])
    new = float(np.interp(W, ws, v24) - v24[0])
    print(f"\n2023R 최적 {w23:.2f} → 2024 에 적용 "
          f"{np.interp(w23, ws, v24) - v24[0]:+.2f}")
    print(f"2024 자기적합 최적 {w24:.2f} (낙관)")
    print(f"채택 w = {W:.2f} (중점)   2024 이득 {new:+.2f}  "
          f"vs 현행 0.38 의 {cur:+.2f}   →  차이 {new - cur:+.2f}")
    print("\n※ 이 차이가 코드 수정으로 얻는 전부다. 재학습은 없다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
