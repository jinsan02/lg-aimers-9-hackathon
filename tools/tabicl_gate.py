"""TabICLv2 게이트 — 추론시간부터 잰다. 10분 초과면 성능은 볼 필요도 없음.

ICL은 학습셋을 컨텍스트로 들고 추론하므로 비용 구조가 GBDT와 다르다.
컨텍스트 크기별로 24.6만 행 예측 시간을 재고 L4로 환산한다.

실행(A100): ~/.venvs/tabicl/bin/python tools/tabicl_gate.py
"""

import time

import numpy as np
import torch

N_TEST = 245_789
LIMIT_S = 600.0
L4_FACTOR = 3.5   # A100 → L4 보수적 환산
CONTEXT_SIZES = [10_000, 30_000, 60_000]


def bss(y, p):
    r = y.mean()
    return float(max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))))


def main():
    from tabicl import TabICLClassifier

    d = np.load("data/processed/train_v1.npz")
    is_val = d["season"] == 2024
    X = np.hstack([d["X_num"], d["X_cat"].astype(np.float32)])
    y = d["y"].astype(np.int64)
    Xtr_all, ytr_all = X[~is_val], y[~is_val]
    Xva, yva = X[is_val], y[is_val].astype(np.float64)
    print(f"train {Xtr_all.shape} | val {Xva.shape} | 피처 {X.shape[1]}")

    rng = np.random.default_rng(0)
    # 실제 평가 규모(24.6만)로 시간 측정하되, 우선 소규모로 스케일 확인
    probe_n = 20_000
    for ctx in CONTEXT_SIZES:
        idx = rng.choice(len(Xtr_all), size=min(ctx, len(Xtr_all)), replace=False)
        clf = TabICLClassifier(device="cuda", random_state=42)
        t0 = time.perf_counter()
        clf.fit(Xtr_all[idx], ytr_all[idx])
        t_fit = time.perf_counter() - t0

        t0 = time.perf_counter()
        p = clf.predict_proba(Xva[:probe_n])[:, 1]
        torch.cuda.synchronize()
        t_pred = time.perf_counter() - t0

        per_row = t_pred / probe_n
        full_a100 = per_row * N_TEST
        full_l4 = full_a100 * L4_FACTOR
        score = bss(yva[:probe_n], p)
        verdict = "✅" if full_l4 < LIMIT_S else "❌"
        print(f"ctx={ctx:6d} | fit {t_fit:5.1f}s | {probe_n}행 예측 {t_pred:6.1f}s "
              f"| 24.6만 환산 A100 {full_a100 / 60:5.1f}분 / L4 {full_l4 / 60:5.1f}분 "
              f"{verdict} | BSS(부분) {score:7.2f}")
        del clf
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
