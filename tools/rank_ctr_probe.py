"""Which part of the real recipe makes a GPU ranker unscoreable?

A 40k-row GPU ranker with 12 numeric features and 2 categoricals fits, saves and
predicts fine. The real one -- 121 features, 9 categoricals, border_count 254 --
segfaults on `predict` from a saved .cbm, on two machines, on CPU, on 100 rows.
So the fault is not `task_type=GPU` on its own.

This holds rows and trees small and varies only the categorical handling, which
is what changes the CTR structure a GPU model has to serialise:

    full        121 features,  9 categoricals   (the champion's set)
    onehot      same, but one_hot_max_size high enough to skip CTRs entirely
    ctr1        same, max_ctr_complexity 1 (no categorical combinations)
    nocat       the 112 numeric features only

Each is fitted in one subprocess and scored in another, so a segfault is a
reported exit code rather than something that takes this process with it.

  python tools/rank_ctr_probe.py        (run on a machine with the GPU)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

N_ROWS = 60_000
ITERS = 30
GROUP = 16
ARMS = {
    "full":   {},
    "onehot": {"one_hot_max_size": 1024},
    "ctr1":   {"max_ctr_complexity": 1},
    "nocat":  {"_drop_cat": True},
}


def build():
    import fpipe
    pack = joblib.load(os.path.join(ROOT, "model", "cat_B1SMOKE_base.pkl"))
    header = list(pd.read_csv(os.path.join(ROOT, "data", "test.csv"), nrows=0,
                              encoding="utf-8-sig").columns)
    raw = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                      encoding="utf-8-sig",
                      usecols=header + ["control_success"])
    raw = raw[raw.season == 2022].head(N_ROWS).reset_index(drop=True)
    fr = fpipe.transform(raw, pack["fpipe"])
    X = fr[pack["features"]].copy()
    for c in pack["cat_cols"]:
        X[c] = X[c].astype(str)
    return X, fr["control_success"].to_numpy(float), list(pack["cat_cols"])


def fit(arm, path):
    from catboost import CatBoostRanker, Pool
    X, y, cats = build()
    kw = dict(ARMS[arm])
    if kw.pop("_drop_cat", False):
        X = X.drop(columns=cats)
        cats = []
    g = np.arange(len(X)) // GROUP
    m = CatBoostRanker(iterations=ITERS, depth=8, learning_rate=0.01,
                       border_count=254, loss_function="PairLogitPairwise",
                       task_type="GPU", devices="0", random_seed=3, verbose=0,
                       **kw)
    m.fit(Pool(X, y, cat_features=cats or None, group_id=g))
    m.save_model(path)
    print(f"FIT_OK {arm} trees={m.tree_count_} "
          f"size={os.path.getsize(path):,}B", flush=True)


def score(arm, path):
    from catboost import CatBoostRanker
    X, y, cats = build()
    if ARMS[arm].get("_drop_cat"):
        X = X.drop(columns=cats)
    m = CatBoostRanker()
    m.load_model(path)
    print(f"PREDICT_OK {arm} mean="
          f"{float(m.predict(X.head(100)).mean()):+.6f}", flush=True)


def main():
    if len(sys.argv) > 1:
        {"fit": fit, "score": score}[sys.argv[1]](sys.argv[2], sys.argv[3])
        return
    tmp = tempfile.gettempdir()
    print(f"{N_ROWS:,} rows, {ITERS} iterations, depth 8, border_count 254, "
          f"group{GROUP}\n")
    for arm in ARMS:
        path = os.path.join(tmp, f"ctrprobe_{arm}.cbm")
        line = f"{arm:8}"
        for phase in ("fit", "score"):
            r = subprocess.run([sys.executable, os.path.abspath(__file__),
                                phase, arm, path],
                               capture_output=True, text=True)
            tail = [l for l in (r.stdout or "").strip().splitlines() if l]
            line += f"  {phase} exit={r.returncode:<5}"
            if tail:
                line += tail[-1].split(" ", 1)[1] if " " in tail[-1] else ""
            if r.returncode != 0:
                line += " <-- died"
                break
        print(line, flush=True)


if __name__ == "__main__":
    main()
