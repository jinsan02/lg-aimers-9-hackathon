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


def _git() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn", "catboost", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


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
            "fit_max_season": int(season[is_fit].max()) if is_fit.any() else None,
            "val_season": args.val_season,
            "test_season": args.test_season or None,
            "seasons": {int(k): int(v) for k, v in season.value_counts().items()},
        },
        "features": {"n": len(features), "fingerprint": _sha("|".join(features))},
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
