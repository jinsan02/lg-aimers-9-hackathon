"""Interrogate the ranker that cannot be scored, beside one that can.

The scale probes settled that the *configuration* is innocent: 870,752 rows,
1200 trees, border_count 254, all nine categoricals, fitted fresh, scores in
0.0s. The shipped stage-1 model has the same rows, 1224 trees and the same size
class, and segfaults on 100 rows. So the difference is in the artifact, not the
recipe, and the artifact is still on disk.

Every accessor runs in its own subprocess with a marker printed before it, so a
native abort names the call that died instead of taking the report with it.

  python tools/rank_model_autopsy.py <bad.cbm> <good.cbm>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

CALLS = [
    "load", "tree_count", "feature_names", "cat_indices", "all_params",
    "scale_and_bias", "json_dump", "predict_1", "predict_100",
]


def child(path, call):
    from catboost import CatBoostRanker
    import numpy as np
    import pandas as pd

    m = CatBoostRanker()
    m.load_model(path)
    if call == "load":
        print(f"loaded {os.path.getsize(path):,}B")
    elif call == "tree_count":
        print(f"trees={m.tree_count_}")
    elif call == "feature_names":
        print(f"n_features={len(m.feature_names_)}")
    elif call == "cat_indices":
        print(f"cat={m.get_cat_feature_indices()}")
    elif call == "all_params":
        p = m.get_all_params()
        print(json.dumps({k: str(v) for k, v in sorted(p.items())}))
    elif call == "scale_and_bias":
        print(f"scale_and_bias={m.get_scale_and_bias()}")
    elif call == "json_dump":
        out = os.path.join(tempfile.gettempdir(),
                           os.path.basename(path) + ".json")
        m.save_model(out, format="json")
        d = json.load(open(out, encoding="utf-8"))
        fi = d.get("features_info", {})
        print(json.dumps({
            "size": os.path.getsize(out),
            "oblivious_trees": len(d.get("oblivious_trees", [])),
            "float_features": len(fi.get("float_features", [])),
            "cat_features": len(fi.get("cat_features", [])),
            "ctrs": len(d.get("ctr_data", {}).get("ctrs", {})
                        if isinstance(d.get("ctr_data"), dict) else []),
            "one_hot": len(fi.get("one_hot_features", [])),
        }))
    elif call.startswith("predict"):
        n = int(call.split("_")[1])
        names = list(m.feature_names_)
        cats = set(m.get_cat_feature_indices())
        rng = np.random.default_rng(0)
        X = pd.DataFrame({c: rng.normal(size=n) for c in names})
        for i in cats:
            X[names[i]] = "R"
        print(f"pred_mean={float(m.predict(X).mean()):+.6f}")


def main():
    if len(sys.argv) == 4:
        child(sys.argv[2], sys.argv[3])
        return
    bad, good = sys.argv[1], sys.argv[2]
    params = {}
    for label, path in (("BAD ", bad), ("GOOD", good)):
        print(f"\n=== {label} {path}")
        for call in CALLS:
            r = subprocess.run([sys.executable, os.path.abspath(__file__),
                                "--child", path, call],
                               capture_output=True, text=True, timeout=600)
            out = (r.stdout or "").strip().splitlines()
            tail = out[-1] if out else (r.stderr or "").strip()[-160:]
            if call == "all_params" and r.returncode == 0 and out:
                params[label] = json.loads(out[-1])
                tail = f"{len(params[label])} parameters"
            mark = "ok  " if r.returncode == 0 else f"DIED exit={r.returncode} "
            print(f"  {call:15} {mark}{tail[:150]}")
    if len(params) == 2:
        a, b = params["BAD "], params["GOOD"]
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        print(f"\n=== parameter differences ({len(diff)})")
        for k in diff:
            print(f"  {k:28} BAD={a.get(k)!r:28} GOOD={b.get(k)!r}")


if __name__ == "__main__":
    main()
