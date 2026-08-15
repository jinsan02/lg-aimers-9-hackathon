"""Which ingredient makes a GPU PairLogit model unscoreable? At the scale that fails.

Established: the fitted ranker cannot be scored. A saved `.cbm` from the real
run segfaults on `predict` with 100 rows, on CPU, on two machines, from a
DataFrame and from a Pool, with 22 GB free. Small models are fine. So the
trigger is scale, and this separates the ingredients **at the scale that
reproduces the failure** rather than at a size where everything passes.

Earlier arms at 30 trees told us nothing: 30 trees does not fail, so "ctr1
passed" and "nocat passed" were vacuous. The reproducing scale is ~300 trees,
where the model does not segfault but **hangs** in `predict` -- so the score
phase runs under a timeout and a hang is reported, not waited on.

Arms, one ingredient at a time:

    full254    the current recipe: border_count 254, default max_ctr_complexity
    border32   border_count 32 -- CatBoost's own default for GPU
               PairLogitPairwise, which the general recipe overrode to 254
    ctr1       max_ctr_complexity 1: no categorical combinations
    ctr1_b32   both
    nocat      the 112 numeric features only, no categoricals at all

Reading:

    nocat passes, full254 fails      -> the CTR applier
    ctr1 passes                      -> combined CTRs specifically
    border32 passes                  -> model growth from the 254 borders
    everything fails                 -> large GPU PairLogit ensembles as such

`one_hot_max_size` is deliberately not an arm: pairwise scoring does not support
one-hot, so its failure to fit is a documented constraint and not evidence.

  python tools/rank_ctr_probe.py [--iters 300] [--rows 60000]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

GROUP = 16
SCORE_TIMEOUT = 300          # a hang is the failure mode, not just a crash

ARMS = {
    "full254":  {"border_count": 254},
    "border32": {"border_count": 32},
    "ctr1":     {"border_count": 254, "max_ctr_complexity": 1},
    "ctr1_b32": {"border_count": 32, "max_ctr_complexity": 1},
    "nocat":    {"border_count": 254, "_drop_cat": True},
}


def build(rows):
    import fpipe
    pack = joblib.load(os.path.join(ROOT, "model", "cat_B1SMOKE_base.pkl"))
    header = list(pd.read_csv(os.path.join(ROOT, "data", "test.csv"), nrows=0,
                              encoding="utf-8-sig").columns)
    raw = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                      encoding="utf-8-sig",
                      usecols=header + ["control_success"])
    raw = raw[raw.season == 2022].head(rows).reset_index(drop=True)
    fr = fpipe.transform(raw, pack["fpipe"])
    X = fr[pack["features"]].copy()
    for c in pack["cat_cols"]:
        X[c] = X[c].astype(str)
    return X, fr["control_success"].to_numpy(float), list(pack["cat_cols"])


def fit(arm, path, iters, rows):
    from catboost import CatBoostRanker, Pool
    X, y, cats = build(rows)
    kw = dict(ARMS[arm])
    if kw.pop("_drop_cat", False):
        X = X.drop(columns=cats)
        cats = []
    g = np.arange(len(X)) // GROUP
    t0 = time.time()
    m = CatBoostRanker(iterations=iters, depth=8, learning_rate=0.01,
                       loss_function="PairLogitPairwise", task_type="GPU",
                       devices="0", random_seed=3, verbose=0, **kw)
    m.fit(Pool(X, y, cat_features=cats or None, group_id=g))
    m.save_model(path)
    print(f"trees={m.tree_count_} size={os.path.getsize(path):,}B "
          f"fit={time.time() - t0:.0f}s", flush=True)


def score(arm, path, iters, rows):
    from catboost import CatBoostRanker
    X, y, cats = build(rows)
    if ARMS[arm].get("_drop_cat"):
        X = X.drop(columns=cats)
    m = CatBoostRanker()
    m.load_model(path)
    t0 = time.time()
    p = m.predict(X.head(100))
    print(f"mean={float(p.mean()):+.6f} in {time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--rows", type=int, default=60_000)
    ap.add_argument("--only", default="")
    a, rest = ap.parse_known_args()
    if rest:
        phase, arm, path = rest[0], rest[1], rest[2]
        {"fit": fit, "score": score}[phase](arm, path, a.iters, a.rows)
        return

    tmp = tempfile.gettempdir()
    arms = [x for x in ARMS if not a.only or x in a.only.split(",")]
    print(f"{a.rows:,} rows, {a.iters} iterations, depth 8, group{GROUP}, "
          f"GPU fit -> fresh-process CPU predict of 100 rows "
          f"(timeout {SCORE_TIMEOUT}s)\n")
    print(f"{'arm':10}{'fit':>36}   {'score':<40}")
    for arm in arms:
        path = os.path.join(tmp, f"ctrprobe_{a.iters}_{arm}.cbm")
        row = f"{arm:10}"
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--iters", str(a.iters), "--rows", str(a.rows),
                            "fit", arm, path], capture_output=True, text=True)
        out = [l for l in (r.stdout or "").strip().splitlines() if l]
        row += f"{(out[-1] if out else 'exit ' + str(r.returncode)):>36}"
        if r.returncode != 0:
            print(row + "   FIT FAILED", flush=True)
            continue
        try:
            r2 = subprocess.run([sys.executable, os.path.abspath(__file__),
                                 "--iters", str(a.iters), "--rows",
                                 str(a.rows), "score", arm, path],
                                capture_output=True, text=True,
                                timeout=SCORE_TIMEOUT)
            o2 = [l for l in (r2.stdout or "").strip().splitlines() if l]
            verdict = ("OK  " + o2[-1]) if r2.returncode == 0 else \
                      f"DIED exit={r2.returncode}"
        except subprocess.TimeoutExpired:
            verdict = f"HUNG >{SCORE_TIMEOUT}s"
        print(row + f"   {verdict}", flush=True)


if __name__ == "__main__":
    main()
