"""추론 가속 벤치마크 — 평가 서버 조건(245,789행) 기준.

측정 대상:
  CatBoost(현 제출 모델) / TabM fp32 / TabM fp16 / TabM int8 동적양자화(torch)
KV-cache 양자화는 자기회귀 트랜스포머 전용이라 본 파이프라인에 해당 없음(§리포트 참조).

실행: PYTHONPATH=src python tools/bench_speed.py
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
N = 245_789
LIMIT = 600.0


def timeit(fn, warmup=1, rep=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(rep):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def bench_catboost():
    import joblib
    import pandas as pd
    from features import add_features
    pack = joblib.load("model/cat_fv2.pkl")
    df = pd.read_csv("data/train.csv", encoding="utf-8-sig", nrows=300_000)
    df, _ = add_features(df, pack["priors"])
    X = df[pack["features"]].iloc[:N].copy()
    for c in pack["cat_cols"]:
        X[c] = X[c].astype(str)
    el = timeit(lambda: pack["model"].predict_proba(X), warmup=1, rep=2)
    print(f"CatBoost(제출 모델)      {el:6.2f}s  ({el / LIMIT * 100:.2f}% of 10min)")
    return el


def bench_tabm():
    from eval_tabm_ensemble import CAT_NAMES
    from tabm_reference import Model
    ck = torch.load("model/tabm_v4s0_best.pt", map_location="cpu",
                    weights_only=False)
    cfg = ck["config"]
    m = Model(n_num_features=cfg["n_num_features"],
              cat_cardinalities=cfg["cat_cardinalities"], n_classes=None,
              backbone=cfg["backbone"], bins=ck["bins"],
              num_embeddings=cfg["num_embeddings"],
              arch_type=cfg["arch_type"], k=cfg["k"]).eval()
    kept = cfg.get("cat_cols_kept") or CAT_NAMES
    idx = [CAT_NAMES.index(c) for c in kept]

    d = np.load("data/processed/train_v1.npz")
    xn = torch.as_tensor(d["X_num"][:N])
    xc = torch.as_tensor(d["X_cat"][:N].astype(np.int64)[:, idx])

    @torch.no_grad()
    def run(model, dtype=None):
        def f():
            for i in range(0, N, 16384):
                a = xn[i:i + 16384]
                if dtype is not None:
                    a = a.to(dtype)
                model(a, xc[i:i + 16384])
        return f

    el = timeit(run(m), rep=2)
    print(f"TabM fp32               {el:6.2f}s  ({el / LIMIT * 100:.2f}%)")

    mq = torch.ao.quantization.quantize_dynamic(
        m, {torch.nn.Linear}, dtype=torch.qint8)
    elq = timeit(run(mq), rep=2)
    print(f"TabM int8 동적양자화     {elq:6.2f}s  ({elq / el:.2f}× vs fp32)")
    return el, elq


if __name__ == "__main__":
    print(f"평가 서버 조건: {N:,}행 / 추론 한도 {LIMIT:.0f}초 (CPU 6 vCPU 가정)\n")
    torch.set_num_threads(6)
    bench_tabm()
    bench_catboost()
