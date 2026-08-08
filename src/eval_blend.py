"""LGBM × TabM 검증 예측 블렌딩 평가.

실행: python src/eval_blend.py out/lgbm_v3_val_preds.npz out/tabm_val_preds.npz
(두 파일 모두 2024 검증셋, 같은 행 순서 전제 — prep/train 스크립트가 동일 순서 유지)
"""

import sys

import numpy as np


def bss(y, p):
    r = y.mean()
    return float(max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))))


def main():
    lgbm_path, tabm_path = sys.argv[1], sys.argv[2]
    a = np.load(lgbm_path)
    b = np.load(tabm_path)
    y = a["y"]
    assert len(y) == len(b["y"]) and np.allclose(y, b["y"]), "검증셋 불일치"
    p_lgbm, p_tabm = a["pred"], b["ensemble"]

    print(f"LGBM  BSS {bss(y, p_lgbm):.2f}")
    print(f"TabM  BSS {bss(y, p_tabm):.2f}")
    print("\nalpha·LGBM + (1-alpha)·TabM:")
    best = (None, -1)
    for alpha in np.arange(0, 1.01, 0.05):
        s = bss(y, alpha * p_lgbm + (1 - alpha) * p_tabm)
        mark = ""
        if s > best[1]:
            best = (alpha, s)
            mark = " *"
        print(f"  a={alpha:.2f}  BSS {s:8.2f}{mark}")
    print(f"\nBEST: alpha={best[0]:.2f} → BSS {best[1]:.2f}")

    # 로짓 평균 (확률 평균보다 나은 경우가 많음)
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    for alpha in [0.3, 0.5, 0.7]:
        z = alpha * logit(p_lgbm) + (1 - alpha) * logit(p_tabm)
        s = bss(y, 1 / (1 + np.exp(-z)))
        print(f"logit blend a={alpha:.1f}: BSS {s:.2f}")


if __name__ == "__main__":
    main()
