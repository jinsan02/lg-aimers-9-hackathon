"""Materialise the next-season side of the BND22/BND23 clean boundaries.

The trainer saved selection-model predictions for the validation season and a
deployment pack refitted through that season.  Applying that frozen pack to the
following official-train season completes the honest source->target pair:

    BND22: val 2022 (fit <=2021) -> pack fit <=2022 predicts 2023
    BND23: val 2023 (fit <=2022) -> pack fit <=2023 predicts 2024

Only official train rows are used.  The generated arrays are research evidence,
never submission artefacts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fpipe  # noqa: E402

SEEDS = (3, 4, 5, 6, 8, 13)
SPECS = {
    "BND22_base": (2022, 2023),
    "BND22_cell": (2022, 2023),
    "BND23_base": (2023, 2024),
    "BND23_cell": (2023, 2024),
}


def _same_or_raise(path: Path, expected: dict[str, np.ndarray]) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=True) as z:
        for key in ("y", "pred", "row_id"):
            if key not in z.files or not np.array_equal(z[key], expected[key]):
                raise ValueError(f"refuse to overwrite mismatching {path} ({key})")
    return True


def materialize(data: pd.DataFrame, model_dir: Path, out_dir: Path) -> list[dict]:
    rows = []
    for tag, (source, target) in SPECS.items():
        target_rows = data.loc[data.season == target].reset_index(drop=True)
        if not len(target_rows):
            raise ValueError(f"no official train rows for target season {target}")
        y = target_rows.control_success.to_numpy(np.float64)
        row_id = target_rows.row_id.astype(str).to_numpy()
        for seed in SEEDS:
            pack_path = model_dir / f"cat_{tag}_s{seed}.pkl"
            if not pack_path.exists():
                raise FileNotFoundError(pack_path)
            pack = joblib.load(pack_path)
            lin = pack.get("lineage") or {}
            # Lineage ``rows.fit_max_season`` describes the selection fit
            # (source-1), while the saved deployment model is refitted through
            # the validation/source season.  The latter contract is recorded in
            # the frozen training flag, not in ``rows.fit_max_season``.
            flags = lin.get("flags") or {}
            if int(flags.get("max_train_season", -1)) != source:
                raise ValueError(f"{pack_path}: max_train_season is not {source}")
            lr = lin.get("rows") or {}
            if int(lr.get("fit_max_season", -1)) != source - 1:
                raise ValueError(f"{pack_path}: selection fit is not <= {source-1}")
            pred = np.asarray(fpipe.predict(pack, target_rows), np.float64)
            if pred.shape != y.shape or not np.isfinite(pred).all():
                raise ValueError(f"{pack_path}: invalid prediction shape/value")
            payload = {"y": y, "pred": pred, "row_id": row_id}
            dest = out_dir / f"cat_{tag}_s{seed}_test_preds.npz"
            existed = _same_or_raise(dest, payload)
            if not existed:
                np.savez_compressed(
                    dest, **payload, seed=seed, host=lin.get("host", ""),
                    surface=f"val{source}->test{target}",
                    argv=f"frozen refit pack {tag} applied to official train season {target}")
            rows.append({"tag": tag, "seed": seed, "source": source,
                         "target": target, "rows": len(y),
                         "mean": float(pred.mean()), "existing": existed})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args(argv)
    data = pd.read_csv(args.data)
    rows = materialize(data, Path(args.model_dir), Path(args.out_dir))
    for r in rows:
        print(f"{r['tag']} s{r['seed']} {r['source']}->{r['target']} "
              f"n={r['rows']:,} mean={r['mean']:.6f} "
              f"{'verified' if r['existing'] else 'written'}")
    print(f"materialized {len(rows)}/{len(SPECS)*len(SEEDS)} next-season arrays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
