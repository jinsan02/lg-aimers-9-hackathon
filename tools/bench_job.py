"""CPU/GPU 동시 실행 시 자원 분배 벤치마크의 '한 작업' (E151).

A100 서버는 물리 20코어 x 2소켓 x HT = 논리 80이고, **GPU 는 NUMA node0 에
붙어 있다**(nvidia-smi topo: CPU Affinity 0-19,40-59).

  node0: 0-19, 40-59   <- GPU 가 여기
  node1: 20-39, 60-79

앞서 LightGBM 을 num_threads=0(=77스레드)으로 돌렸다가 동시 실행 중이던
CatBoost GPU 작업이 3.2배 느려졌다. GPU 학습도 데이터 공급에 CPU 스레드를
쓰기 때문이다. 이 스크립트는 그 간섭을 실제로 재기 위한 것이다.

실행 (드라이버가 taskset 으로 감싸서 호출한다):
    python tools/bench_job.py --device GPU --iters 1000 --threads 8
"""

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="./out/bench.npz")
    ap.add_argument("--device", choices=["GPU", "CPU"], required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import pandas as pd
    z = np.load(args.npz, allow_pickle=True)
    Xn, Xc, y = z["Xn"], z["Xc"], z["y"]
    # CatBoost 는 범주형 열이 float 이면 거부한다 — 정수 열로 따로 붙인다
    X = pd.DataFrame(Xn, columns=[f"n{i}" for i in range(Xn.shape[1])])
    for j in range(Xc.shape[1]):
        X[f"c{j}"] = Xc[:, j].astype(np.int32)
    cat_idx = [f"c{j}" for j in range(Xc.shape[1])]

    from catboost import CatBoostClassifier, Pool
    t0 = time.time()
    pool = Pool(X, y, cat_features=cat_idx, thread_count=args.threads)
    t_pool = time.time() - t0

    kw = dict(iterations=args.iters, learning_rate=0.05, depth=8,
              l2_leaf_reg=10, border_count=128, task_type=args.device,
              thread_count=args.threads, random_seed=42, verbose=0)
    if args.device == "GPU":
        kw["devices"] = "0"
    t1 = time.time()
    CatBoostClassifier(**kw).fit(pool)
    t_fit = time.time() - t1
    print(f"RESULT {args.label or args.device} device={args.device} "
          f"threads={args.threads} iters={args.iters} "
          f"pool={t_pool:.1f}s fit={t_fit:.1f}s", flush=True)


if __name__ == "__main__":
    main()
