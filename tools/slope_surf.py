"""SLOPE 를 **올바른 표면**에서 다시 검정한다 (E156).

SLOPE 1.0416 은 *검증* 모델(≤2022 학습, 2023 검증)의 예측에 적합해서 *검증* 모델의
2024 예측에 적용해 +3.51 을 얻은 값이다. 그런데 제출은 **refit 모델**의 블렌드다.
refit 모델은 데이터를 한 시즌 더 봐서 분산 구조가 다르다 — 실제로 편향이 3배 달랐다.

여기서는 미학습 시즌 예측끼리 검정한다:
  미학습 2023 예측에 기울기를 적합 → 미학습 2024 예측에 적용 → 이득이 남는가

실행: python tools/slope_surf.py
"""

import glob

import numpy as np


def load(tag):
    fs = sorted(glob.glob(f"./out/*{tag}_s*_test_preds.npz"))
    z = [q for q in (np.load(f, allow_pickle=True) for f in fs)
         if "row_id" in q.files]
    if not z:
        return None, None
    return (np.mean([q["pred"] for q in z], 0).astype(np.float64),
            z[0]["y"].astype(np.float64))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_slope(p, y):
    X = np.column_stack([np.ones(len(p)), _logit(p)])
    w = np.array([0.0, 1.0])
    for _ in range(30):
        q = 1 / (1 + np.exp(-X @ w))
        W = q * (1 - q)
        H = (X * W[:, None]).T @ X + 1e-9 * np.eye(2)
        w = w + np.linalg.solve(H, X.T @ (y - q))
    return float(w[0]), float(w[1])


def main():
    for a, b in (("2022", "2023"), ("2023", "2024")):
        pa, ya = load(f"BI{a}")
        pb, yb = load(f"BI{b}")
        if pa is None or pb is None:
            print(f"{a}->{b}: 예측 없음")
            continue
        r = yb.mean()
        base = r * (1 - r)

        def bss(p):
            return 1e5 * (1 - ((np.clip(p, 0, 1) - yb) ** 2).mean() / base)

        i_a, s_a = fit_slope(pa, ya)
        i_b, s_b = fit_slope(pb, yb)
        z = _logit(pb)
        print(f"\n미학습 {a} 적합 기울기 {s_a:.4f} (절편 {i_a:+.4f}) | "
              f"미학습 {b} 자기적합 {s_b:.4f}")
        print(f"  {b} 기준선            {bss(pb):8.2f}")
        for nm, s in ((f"{a} 기울기 이전", s_a), (f"{b} 자기적합(상한)", s_b),
                      ("현행 1.0416", 1.0416)):
            q = 1 / (1 + np.exp(-s * z))
            print(f"  {nm:<20} {bss(q):8.2f}  ({bss(q) - bss(pb):+.2f})")
    print("\n※ '기울기 이전' 이 양수여야 SLOPE 를 쓸 근거가 된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
