"""Algebraic and row-independence contract for the F1 adapter."""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import f_adapter_gate as f1


def main():
    rng = np.random.default_rng(7)
    n = 1000
    x = pd.DataFrame({c: rng.normal(size=n) for c in f1.FEATURES})
    p0 = np.clip(rng.normal(.49, .04, n), .1, .9)
    true_beta = np.array([.08, -.05, .04, -.03, .02, -.01])
    y = rng.binomial(1, f1.expit(f1.logit(p0) + x.to_numpy() @ true_beta))
    art = f1.fit_adapter(x, p0, y)
    q = f1.apply_adapter(art, x, p0)
    assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
    assert art["intercept"] == 0.0 and art["l2"] == 100.0
    rev = f1.apply_adapter(art, x.iloc[::-1], p0[::-1])[::-1]
    half = f1.apply_adapter(art, x.iloc[:500], p0[:500])
    one = f1.apply_adapter(art, x.iloc[[0]], p0[[0]])
    assert np.array_equal(q, rev)
    assert np.array_equal(q[:500], half)
    assert q[0] == one[0]
    print("F1 adapter contract PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
