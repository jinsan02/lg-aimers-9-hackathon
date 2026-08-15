"""Write the provenance record a run needs to be comparable later.

The audit's reproducibility bar is deliberately not byte-identity: it is
"regenerate to within run-to-run variation, and know exactly what was compared
to what". That needs a small fixed set of facts, and we have repeatedly lost
runs because one of them was missing -- which machine, which training rows,
which feature order, which taxonomy.

`out/lineage_<tag>.json` is written next to the predictions. Cheap enough to
write on every run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def rowid_sha(row_ids) -> str:
    """Hash the *set* of training rows, order-independently.

    The priors float in `member_fingerprint.py` is a mean, and a mean is a weak
    fingerprint: two different row sets can share one, and it says nothing about
    *which* rows moved. This hashes the sorted row_id bytes, so a stray
    `--drop-f-pre`, `--min-season` or `--max-train-season` changes it and a
    reordering does not -- which matches the competition's own set-membership
    criterion for what counts as the same data.
    """
    import numpy as np
    a = np.sort(np.asarray(row_ids).astype(np.int64))
    h = hashlib.sha256()
    h.update(str(a.size).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:16]


def _git() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    commit = run("git", "rev-parse", "HEAD")
    if commit:
        return {"commit": commit, "dirty": bool(run("git", "status", "--porcelain")),
                "source": "git"}
    # Remote workers are deployed by scp, so C:\aimers there is not a checkout
    # and git returns nothing. Measured 2026-08-13: the whole B0-JL baseline
    # landed with commit "". `deploy_5070.sh` stamps the file below with the
    # revision it copied, so the record still says which source produced the run.
    for src, val in (("env", os.environ.get("AIMERS_SOURCE_COMMIT", "")),
                     ("file", _read(".deployed_commit"))):
        if val:
            return {"commit": val, "dirty": None, "source": src}
    return {"commit": "", "dirty": None, "source": "unknown"}


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn", "catboost", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


# Flags that change what the model learns. A difference in any of these means
# two members are not the same experiment and their predictions must not be
# blended or compared -- the v16 (-6.15) and v17 (-53.6) mistakes were both a
# single stray flag in a handoff command. Flags that only change *reporting*
# (--tag, --out, --quiet) are deliberately absent: they must not trip a check.
TRAINING_RELEVANT = (
    "model", "val_season", "test_season", "max_train_season", "min_season",
    "drop_f_pre", "drop_cols", "feat_v2", "feat_v4", "feat_std", "std_k",
    "std_to_prior", "std_season_prior", "feat_domain", "feat_skill",
    "feat_skill_pc", "feat_h1", "feat_k", "te", "te_dev", "te_halflife",
    "failmode_cells", "fm_modes", "fm_min_share", "fm_multilabel",
    "missing_strategy", "extra_feats", "soft_target", "resid_col",
    "baseline_col", "lr", "depth", "l2", "es", "iters", "refit_mult",
    "no_refit", "border_count", "one_hot_max_size", "rank_group",
    "loss_function", "subsample", "rsm", "min_data_in_leaf",
)


def training_flags(args) -> dict:
    """The subset of resolved args that can move a training number."""
    out = {}
    for k in TRAINING_RELEVANT:
        if hasattr(args, k):
            v = getattr(args, k)
            out[k] = (v if isinstance(v, (int, float, str, bool, type(None)))
                      else str(v))
    return out


def pack_record(args, features, train, is_fit, is_val) -> dict:
    """The compact fingerprint stamped into every `.pkl`.

    Small on purpose -- it rides inside the artifact, so it holds identity, not
    narrative. `out/lineage_<tag>.json` keeps the full record.
    """
    season = train["season"]
    has_id = "row_id" in train.columns
    return {
        "schema": 1,
        "host": socket.gethostname(),
        "git": _git(),
        "rows": {
            "total": int(len(train)),
            "fit": int(is_fit.sum()),
            "val": int(is_val.sum()),
            "fit_min_season": int(season[is_fit].min()) if is_fit.any() else None,
            "fit_max_season": int(season[is_fit].max()) if is_fit.any() else None,
            "val_season": args.val_season,
            "test_season": args.test_season or None,
        },
        "fit_rowid_sha": (rowid_sha(train.loc[is_fit, "row_id"])
                          if has_id and is_fit.any() else None),
        "features_sha": _sha("|".join(features)),
        "n_features": len(features),
        "flags": training_flags(args),
    }


def write(tag, args, features, train, is_val, is_fit, seeds_done,
          cell_names=None, extra=None):
    """`seeds_done` maps seed -> {best_iter, refit_trees, val_raw_bss, ...}."""
    season = train["season"]
    rec = {
        "tag": tag,
        "written": None,                      # stamped by the caller if wanted
        "git": _git(),
        "host": socket.gethostname(),
        "versions": _versions(),
        "argv": " ".join(sys.argv[1:]),
        "resolved_args": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                              else str(v)) for k, v in sorted(vars(args).items())},
        "rows": {
            "total": int(len(train)),
            "fit": int(is_fit.sum()),
            "val": int(is_val.sum()),
            # Both ends of the fit era. Only the max was recorded before, and
            # the 2026-08-15 P0 audit needed the min to tell an honest
            # `--min-season` run from a contaminated one.
            "fit_min_season": int(season[is_fit].min()) if is_fit.any() else None,
            "fit_max_season": int(season[is_fit].max()) if is_fit.any() else None,
            "val_season": args.val_season,
            "test_season": args.test_season or None,
            "max_train_season": getattr(args, "max_train_season", None),
            "seasons": {int(k): int(v) for k, v in season.value_counts().items()},
            # The strong fingerprint: which rows actually trained this model.
            "fit_rowid_sha": (rowid_sha(train.loc[is_fit, "row_id"])
                              if "row_id" in train.columns and is_fit.any()
                              else None),
            "val_rowid_sha": (rowid_sha(train.loc[is_val, "row_id"])
                              if "row_id" in train.columns and is_val.any()
                              else None),
        },
        "features": {"n": len(features), "fingerprint": _sha("|".join(features)),
                     "feat_k": getattr(args, "feat_k", None)},
        "cell_taxonomy": (None if cell_names is None else
                          {"n": len(cell_names), "fingerprint": _sha("|".join(cell_names))}),
        "seeds": seeds_done,
    }
    if extra:
        rec.update(extra)
    os.makedirs("out", exist_ok=True)
    path = f"./out/lineage_{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2, sort_keys=False)
    print(f"lineage: {path}")
    return path
