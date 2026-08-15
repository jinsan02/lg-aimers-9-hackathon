"""Is a GPU-trained PairLogit ranker scoreable at all? And how slow is CPU?

`FLAG --model rank full-refit` recorded that the group16 refit crashes natively
on the 4070 and the A100, and inferred that constructing a second ranker while
the first one's buffers were alive was the problem. A night of staging the work
across four processes showed that reading is too narrow: the fitted model itself
segfaults on `predict`, on a different machine, on CPU, from a saved .cbm, on
100 rows. The refit was simply the first thing that ever touched it.

This isolates the one variable that matters -- `task_type` at fit time -- on a
frame small enough to run in a minute, and then times a CPU fit so the cost of
the only remaining route is a measurement rather than a guess.

  python tools/rank_gpu_inference_probe.py            (run on the 5070)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

N = 40_000
GROUP = 16


def frame(seed=0):
    rng = np.random.default_rng(seed)
    hand = rng.choice(["L", "R"], size=N)
    team = rng.choice([f"T{i}" for i in range(10)], size=N)
    X = pd.DataFrame({f"f{i}": rng.normal(size=N) for i in range(12)})
    X["hand"] = hand
    X["team"] = team
    lin = 0.8 * X.f0 - 0.5 * X.f1 + 0.3 * (hand == "L")
    y = (rng.random(N) < 1 / (1 + np.exp(-lin))).astype(float)
    return X, y, ["hand", "team"]


def child(device, path):
    """Fit in a subprocess so a segfault is reported, not inherited."""
    from catboost import CatBoostRanker, Pool

    X, y, cats = frame()
    g = np.arange(len(X)) // GROUP
    kw = dict(iterations=60, depth=6, learning_rate=0.1,
              loss_function="PairLogitPairwise", verbose=0,
              random_seed=1, task_type=device)
    if device == "GPU":
        kw["devices"] = "0"
    t0 = time.time()
    m = CatBoostRanker(**kw)
    m.fit(Pool(X, y, cat_features=cats, group_id=g))
    fit_s = time.time() - t0
    m.save_model(path)
    print(f"FIT_OK {device} {fit_s:.1f}s trees={m.tree_count_}", flush=True)


def score(path):
    from catboost import CatBoostRanker

    X, y, cats = frame()
    m = CatBoostRanker()
    m.load_model(path)
    p = m.predict(X.head(100))
    print(f"PREDICT_OK mean={float(p.mean()):+.6f}", flush=True)


def main():
    if len(sys.argv) > 2:
        {"fit": lambda: child(sys.argv[2], sys.argv[3]),
         "score": lambda: score(sys.argv[3])}[sys.argv[1]]()
        return

    tmp = tempfile.gettempdir()
    for device in ("CPU", "GPU"):
        path = os.path.join(tmp, f"probe_{device}.cbm")
        print(f"\n=== fit on {device}, then score in a fresh process")
        for phase in ("fit", "score"):
            args = [sys.executable, os.path.abspath(__file__), phase, device,
                    path]
            r = subprocess.run(args, capture_output=True, text=True)
            out = (r.stdout or "").strip().splitlines()
            print(f"  {phase:6} exit={r.returncode}  "
                  + (out[-1] if out else
                     (r.stderr or "").strip().splitlines()[-1:] or ["(silent)"])[0]
                  if not out else f"  {phase:6} exit={r.returncode}  {out[-1]}")
            if r.returncode != 0 and phase == "fit":
                break

    print("\n=== CPU fit rate on a realistic width, for costing the only "
          "remaining route")
    from catboost import CatBoostRanker, Pool
    X, y, cats = frame()
    g = np.arange(len(X)) // GROUP
    t0 = time.time()
    m = CatBoostRanker(iterations=50, depth=8, learning_rate=0.01,
                       loss_function="PairLogitPairwise", verbose=0,
                       random_seed=1, task_type="CPU", border_count=254)
    m.fit(Pool(X, y, cat_features=cats, group_id=g))
    dt = time.time() - t0
    per = dt / 50
    print(f"  {N:,} rows, depth 8, 50 iterations: {dt:.1f}s "
          f"({per:.3f}s/iter)")
    print(f"  extrapolated to 870,752 rows x 1500 iterations: "
          f"{per * (870752 / N) * 1500 / 3600:.1f} h")


if __name__ == "__main__":
    main()
