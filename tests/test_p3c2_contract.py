"""P3-C2 weighted-cell training and shipped analytic-deweight contract."""

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
import fpipe                                                     # noqa: E402
import train_gbdt2 as trainer                                    # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def frame(n=120):
    cols = list(pd.read_csv(os.path.join(ROOT, "data", "test.csv"), nrows=0,
                            encoding="utf-8-sig").columns)
    return pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                       encoding="utf-8-sig", usecols=cols,
                       nrows=2000).tail(n).reset_index(drop=True)


def main():
    print("P3-C2 contract")

    labels = np.concatenate([
        np.arange(12, dtype=np.int16),
        np.full(337393, 9, dtype=np.int16),
        np.full(138840, 10, dtype=np.int16),
        np.full(768, 11, dtype=np.int16),
    ])
    # Remove the one seed row per class from the reported cells so the fixture
    # exactly reproduces the pre-registered counts.
    labels = np.delete(labels, [9, 10, 11])
    meta = trainer._p3c2_contract(
        labels, [str(i) for i in range(12)], {9, 10, 11}, "test")
    w = np.asarray(meta["weights"], np.float64)
    check("1 frozen weights match the pre-registration",
          np.allclose(w[[9, 10, 11]],
                      [0.70575412, 1.71504249, 1.0], atol=1e-8),
          str(w[[9, 10, 11]]))
    check("2 residual and every failure weight stay exactly one",
          w[11] == 1.0 and np.array_equal(w[:9], np.ones(9)))
    check("3 weighted success mass is preserved",
          meta["mass_error"] <= 1e-9, f"error {meta['mass_error']:.3e}")

    from catboost import CatBoostClassifier
    sx = np.arange(240, dtype=np.float64).reshape(-1, 1)
    sy = np.tile(np.arange(12, dtype=np.int16), 20)
    sm = CatBoostClassifier(
        iterations=2, depth=2, loss_function="MultiClass", classes_count=12,
        class_weights=w.tolist(), task_type="CPU", verbose=0, random_seed=3)
    sm.fit(sx, sy)
    check("4 CatBoost accepts the positional vector as classes 0..11",
          [int(x) for x in sm.classes_.tolist()] == list(range(12)))

    cell = joblib.load(os.path.join(ROOT, "model", "cat_B1S_cell_s3.pkl"))
    test = frame()
    src = fpipe.transform(test.copy(), cell["fpipe"])
    X = fpipe.design(src, cell)
    q = cell["model"].predict_proba(X)

    identity = fpipe.deweight_multiclass(q, np.ones(12), list(range(12)))
    check("5 all-one correction is bit-identical",
          np.array_equal(identity, q))

    p = fpipe.deweight_multiclass(q, w, list(range(12)))
    check("6 deweighted rows are a finite probability simplex",
          np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
          and np.allclose(p.sum(axis=1), 1.0, atol=1e-12),
          f"max sum error {np.max(np.abs(p.sum(axis=1) - 1)):.3e}")

    pack = dict(cell)
    pack.update({
        "class_weights": w.tolist(),
        "classes_order": list(range(12)),
        "balanced_cells": [9, 10],
        "residual_cells": [11],
        "analytic_deweight": True,
        "fm_success": [9, 10, 11],
    })
    got = fpipe.predict(pack, test.copy())
    want = p[:, [9, 10, 11]].sum(axis=1)
    check("7 trainer algebra equals the shipped fpipe path",
          np.array_equal(got, want), f"max diff {np.max(np.abs(got-want)):.3e}")
    check("8 cell 11 remains in P(success)",
          not np.array_equal(got, p[:, [9, 10]].sum(axis=1)),
          f"mean cell11 mass {p[:, 11].mean():.8f}")

    bad = dict(pack)
    bad["classes_order"] = list(reversed(range(12)))
    try:
        fpipe.predict(bad, test.head(1))
        check("9 class-order mismatch raises", False, "returned a prediction")
    except ValueError:
        check("9 class-order mismatch raises", True)

    rev = fpipe.predict(pack, test.iloc[::-1].reset_index(drop=True))[::-1]
    half = fpipe.predict(pack, test.iloc[:60].copy())
    one = fpipe.predict(pack, test.iloc[[17]].copy())
    check("10 subset/reversal/single-row independence",
          np.array_equal(got, rev) and np.array_equal(got[:60], half)
          and got[17] == one[0])

    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "out")) as td:
        pp = os.path.join(td, "pack.pkl")
        fp = os.path.join(td, "frame.pkl")
        op = os.path.join(td, "pred.npy")
        joblib.dump(pack, pp)
        test.to_pickle(fp)
        code = (
            "import sys,joblib,numpy as np,pandas as pd;"
            f"sys.path.insert(0,{os.path.join(ROOT, 'src')!r});"
            "import fpipe;"
            f"p=joblib.load({pp!r});x=pd.read_pickle({fp!r});"
            f"np.save({op!r},fpipe.predict(p,x))")
        run = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                             capture_output=True, text=True, timeout=300)
        fresh = np.load(op) if run.returncode == 0 and os.path.exists(op) else None
        check("11 save -> fresh process -> predict parity",
              fresh is not None and np.array_equal(got, fresh),
              (run.stderr or "")[-200:])

    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
