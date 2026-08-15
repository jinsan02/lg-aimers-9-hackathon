"""RMSE deployment refit must use the deployment frame and target algebra."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import train_gbdt2 as trainer


def main():
    dep = pd.DataFrame({
        "control_success": [0.0, 1.0, 1.0],
        "base": [0.2, 0.4, 0.8],
    }, index=[10, 20, 30])
    args = SimpleNamespace(label_smooth=0.0)
    y = trainer._rmse_refit_target(args, dep)
    assert np.array_equal(y.index, dep.index)
    assert np.array_equal(y.to_numpy(), np.array([0., 1., 1.]))
    r = trainer._rmse_refit_target(args, dep, "base")
    assert np.allclose(r, [-.2, .6, .2])
    args.label_smooth = 0.1
    s = trainer._rmse_refit_target(args, dep)
    assert np.allclose(s, [.1, .9, .9])

    source = open(os.path.join(ROOT, "src", "train_gbdt2.py"),
                  encoding="utf-8").read()
    block = source[source.index('if args.loss == "RMSE":'):
                   source.index("if args.transfer_split:")]
    assert "Pool(train_dep[features]" in block
    assert "_refit_weights(args, train_dep)" in block
    assert "Pool(train[features]" not in block
    assert 'eval_metric="RMSE"' in block
    assert '"CrossEntropy", "RMSE"' in source
    print("RMSE deployment refit contract PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
