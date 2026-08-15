"""Fresh-process artifact and subset-independence audit for P3-C2."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fpipe  # noqa: E402
import invalidated  # noqa: E402
import train_gbdt2 as trainer  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(tag):
    path = os.path.join(ROOT, "out", f"cat_{tag}_test_preds.npz")
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    a = ap.parse_args()
    invalidated.guard(a.tags)
    os.chdir(ROOT)

    raw, _ = trainer.load(drop_f_pre=2022)[:2]
    test = raw.loc[raw["season"] == 2024].copy()
    if len(test) != 253_507:
        raise SystemExit(f"unexpected 2024 rows: {len(test)}")

    for tag in a.tags:
        model_path = os.path.join(ROOT, "model", f"cat_{tag}.pkl")
        pack = joblib.load(model_path)
        saved = load_npz(tag)
        if not np.array_equal(test["row_id"].to_numpy(), saved["row_id"]):
            raise SystemExit(f"{tag}: row order differs from training frame")
        pred = np.asarray(fpipe.predict(pack, test), np.float64)
        maxdiff = float(np.max(np.abs(pred - saved["pred"])))

        probe = test.iloc[:800].copy()
        p_full = np.asarray(fpipe.predict(pack, probe), np.float64)
        p_rev = np.asarray(fpipe.predict(pack, probe.iloc[::-1]), np.float64)[::-1]
        p_half = np.asarray(fpipe.predict(pack, probe.iloc[:400]), np.float64)
        p_one = np.asarray(fpipe.predict(pack, probe.iloc[[0]]), np.float64)
        drift = max(float(np.max(np.abs(p_full - p_rev))),
                    float(np.max(np.abs(p_full[:400] - p_half))),
                    float(abs(p_full[0] - p_one[0])))
        if maxdiff != 0.0 or drift != 0.0:
            raise SystemExit(
                f"{tag}: parity maxdiff={maxdiff:.3e}, subset drift={drift:.3e}")
        print(f"{tag}: sha256={sha256(model_path)} rows={len(pred):,} "
              f"fresh_process_maxdiff={maxdiff:.3e} subset_drift={drift:.3e} "
              f"features={len(pack['features'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
