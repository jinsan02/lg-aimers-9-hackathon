import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import graph_features as gf


def frame(future_y=0):
    return pd.DataFrame({
        "season": [2021, 2021, 2022, 2022, 2023],
        "pitcher_id": [1, 1, 1, 2, 1],
        "batter_id": [10, 11, 10, 10, 11],
        "control_success": [1, 0, 1, 0, future_y],
    })


def main():
    a, cols, art = gf.add_expanding(frame(0))
    b, _, _ = gf.add_expanding(frame(1))
    # A target-season label cannot alter that season's graph features.
    np.testing.assert_allclose(a.loc[a.season == 2023, cols],
                               b.loc[b.season == 2023, cols], equal_nan=True)
    # Inference is a frozen lookup: changing batch composition/order cannot matter.
    q = frame().iloc[[4, 3]].drop(columns="control_success")
    x, _ = gf.apply_tables(q.iloc[[0]], art)
    y, _ = gf.apply_tables(q, art)
    np.testing.assert_allclose(x[cols], y.iloc[[0]][cols], equal_nan=True)
    print("graph feature cutoff/row-independence tests: OK")


if __name__ == "__main__":
    main()
