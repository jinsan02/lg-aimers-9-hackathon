"""Integrity gate for the pre-registered BND22/BND23 boundary artefacts.

This tool deliberately does *not* rank or score the seven downgraded axes.  It
only establishes whether the 24 shared prerequisite members are safe to spend:
six seeds for base/cell on target seasons 2022 and 2023, all from one host and
with identical row/target arrays inside each boundary.

The historical judging contract is asymmetric: val-2023 requires
``--drop-f-pre 2022`` (SETTLED ``drop-f-pre-omitted``), while val-2022 does
not.  An artefact-presence check that ignores this can bless the exact banned
12--21-tree failure signature, so the flag is part of integrity.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


SEEDS = (3, 4, 5, 6, 8, 13)
HOST = "DESKTOP-053T952"
SPECS = {
    "BND22_base": {"year": 2022, "features": 121, "cell": False},
    "BND22_cell": {"year": 2022, "features": 123, "cell": True},
    "BND23_base": {"year": 2023, "features": 121, "cell": False},
    "BND23_cell": {"year": 2023, "features": 123, "cell": True},
}


def _required_drop_f_pre(year: int) -> int:
    return 2022 if year == 2023 else 0


def _sha_array(a: np.ndarray) -> str:
    a = np.asarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    if a.dtype.kind in "OUS":
        for v in a.astype(str).reshape(-1):
            h.update(v.encode("utf-8")); h.update(b"\0")
    else:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _scalar(z, key):
    v = z[key]
    return v.item() if np.ndim(v) == 0 else v


def audit(model_dir: Path, out_dir: Path, ledger_path: Path) -> dict:
    errors: list[str] = []
    tags: dict[str, dict] = {}
    boundary_identity: dict[int, tuple[str, str]] = {}

    ledger = pd.read_csv(ledger_path, sep="\t", header=None, dtype=str)
    for tag, spec in SPECS.items():
        packs = {s: model_dir / f"cat_{tag}_s{s}.pkl" for s in SEEDS}
        preds = {s: out_dir / f"cat_{tag}_s{s}_val_preds.npz" for s in SEEDS}
        missing = [str(p) for p in [*packs.values(), *preds.values()] if not p.exists()]
        if missing:
            errors.append(f"{tag}: missing {len(missing)} artefact(s)")
            tags[tag] = {"missing": missing}
            continue

        row_hashes, y_hashes, feature_hashes, fit_hashes = set(), set(), set(), set()
        hosts, surfaces, seeds_seen = set(), set(), set()
        finite = True
        pack_rows = []
        for seed in SEEDS:
            with np.load(preds[seed], allow_pickle=True) as z:
                required = {"y", "pred", "row_id", "seed", "host", "surface", "argv"}
                absent = required.difference(z.files)
                if absent:
                    errors.append(f"{tag} s{seed}: npz missing {sorted(absent)}")
                    continue
                y, pred, row_id = z["y"], z["pred"], z["row_id"]
                if not (len(y) == len(pred) == len(row_id)):
                    errors.append(f"{tag} s{seed}: row length mismatch")
                finite &= bool(np.isfinite(pred).all() and np.isfinite(y).all())
                row_hashes.add(_sha_array(row_id)); y_hashes.add(_sha_array(y))
                hosts.add(str(_scalar(z, "host"))); surfaces.add(str(_scalar(z, "surface")))
                seeds_seen.add(int(_scalar(z, "seed")))
                argv = str(_scalar(z, "argv"))
                for token in (f"--val-season {spec['year']}",
                              f"--max-train-season {spec['year']}", "--p1"):
                    if token not in argv:
                        errors.append(f"{tag} s{seed}: argv lacks {token}")
                req_drop = _required_drop_f_pre(spec["year"])
                has_drop = f"--drop-f-pre {req_drop}" in argv if req_drop else "--drop-f-pre" in argv
                if req_drop and not has_drop:
                    errors.append(f"{tag} s{seed}: argv lacks --drop-f-pre {req_drop}")
                if not req_drop and has_drop:
                    errors.append(f"{tag} s{seed}: unexpected --drop-f-pre present")

            pack = joblib.load(packs[seed])
            lin = pack.get("lineage") or {}
            flags = lin.get("flags") or {}
            rows = lin.get("rows") or {}
            feature_hashes.add(str(lin.get("features_sha")))
            fit_hashes.add(str(lin.get("fit_rowid_sha")))
            pack_rows.append(rows)
            if int(lin.get("n_features", -1)) != spec["features"]:
                errors.append(f"{tag} s{seed}: n_features={lin.get('n_features')}")
            if int(flags.get("val_season", -1)) != spec["year"]:
                errors.append(f"{tag} s{seed}: lineage val season mismatch")
            if int(flags.get("max_train_season", -1)) != spec["year"]:
                errors.append(f"{tag} s{seed}: lineage max season mismatch")
            req_drop = _required_drop_f_pre(spec["year"])
            if int(flags.get("drop_f_pre", -1)) != req_drop:
                errors.append(f"{tag} s{seed}: lineage drop_f_pre != {req_drop}")
            if bool(flags.get("failmode_cells")) != spec["cell"]:
                errors.append(f"{tag} s{seed}: cell flag mismatch")
            if spec["cell"] and pack.get("fm_success") != [9, 10, 11]:
                errors.append(f"{tag} s{seed}: success cells mismatch")
            if not spec["cell"] and pack.get("fm_success"):
                errors.append(f"{tag} s{seed}: base unexpectedly has success cells")
            model = pack.get("model")
            if model is None or len(pack.get("features", [])) != spec["features"]:
                errors.append(f"{tag} s{seed}: unscoreable pack structure")

        if row_hashes and y_hashes:
            ident = (next(iter(row_hashes)), next(iter(y_hashes)))
            prior = boundary_identity.get(spec["year"])
            if prior is not None and prior != ident:
                errors.append(f"boundary {spec['year']}: base/cell row or target mismatch")
            boundary_identity[spec["year"]] = ident
        for name, values in (("row_id", row_hashes), ("target", y_hashes),
                             ("features", feature_hashes), ("fit", fit_hashes)):
            if len(values) != 1:
                errors.append(f"{tag}: seed-level {name} disagreement ({len(values)})")
        if hosts != {HOST}:
            errors.append(f"{tag}: hosts={sorted(hosts)}")
        if seeds_seen != set(SEEDS):
            errors.append(f"{tag}: seeds={sorted(seeds_seen)}")
        if not finite:
            errors.append(f"{tag}: non-finite y/pred")

        # The ledger uses the complete seed tag (e.g. BND22_base_s3).
        ledger_rows = ledger[ledger.apply(
            lambda r: r.astype(str).str.contains(fr"{tag}_s", regex=True).any(), axis=1)]
        tags[tag] = {
            "year": spec["year"], "seeds": sorted(seeds_seen),
            "row_id_sha": sorted(row_hashes), "target_sha": sorted(y_hashes),
            "fit_rowid_sha": sorted(fit_hashes), "features_sha": sorted(feature_hashes),
            "hosts": sorted(hosts), "surfaces": sorted(surfaces),
            "finite": finite, "ledger_rows": int(len(ledger_rows)),
            "pack_rows": pack_rows[:1],
        }

    return {
        "ok": not errors,
        "expected_models": len(SPECS) * len(SEEDS),
        "expected_prediction_arrays": len(SPECS) * len(SEEDS),
        "tags": tags,
        "errors": errors,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--ledger", default="LEDGER.tsv")
    ap.add_argument("--report", default="out/bnd_integrity.json")
    args = ap.parse_args(argv)
    report = audit(Path(args.model_dir), Path(args.out_dir), Path(args.ledger))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
