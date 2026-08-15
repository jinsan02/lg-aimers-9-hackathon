"""Check that blend members came from **the same training data and hyperparameters**.

On 2026-08-08 we treated `VB_base` as a reproduction of `cat_v14f` and were about
to pick blend weights from its predictions. It was a different, 52-point-weaker
model — a `--drop-f-pre 2022` had leaked into the handoff command, so the training
data differed. We made that mistake in v16 (−6.15), again in v17 (−53.6), and
again there.

**The old check was one float.** `fpipe['priors']['asof_pitcher_success_rate']`
is a mean over the fit rows: it moves when the training set moves, so it catches
the gross case, but a mean is not an identity. It cannot say *which* rows
differ, two different row sets can share one, and it says nothing about the fit
era, the feature order, or the flags. The 2026-08-15 P0 audit had to reconstruct
fit eras from ledger timestamps and a commit date because the artifacts carried
nothing.

So packs written from 2026-08-15 carry a `lineage` record and this tool prefers
it:

  fit_rowid_sha   sha256 of the sorted row_id of the fit partition — the set of
                  rows that trained the model, order-independent, matching the
                  competition's own set-membership criterion
  rows            total / fit / val counts, fit era **both ends**, val and test
                  season
  features_sha    the ordered feature list
  flags           only the training-relevant flags (tools/lineage.py), so a
                  different --tag or --out never trips a check

Older packs have no `lineage`; they are compared on the legacy float and
reported as **WEAK** so a green line cannot be mistaken for the strong check.

  python tools/member_fingerprint.py v14f ZD5 DX_seq
  python tools/member_fingerprint.py --verbose B1S_base B1S_cell

Exit 2 on any mismatch, on a tag that could not be loaded, or on seeds within a
tag that disagree.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

import joblib

try:                                   # survive a cp949 Windows console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

KEYS = ("learning_rate", "depth", "l2_leaf_reg", "loss_function", "border_count")

# Fields of the lineage record that must agree across members of one blend.
# `host` is not here: the same experiment run on two machines is still the same
# experiment, and the machine-effect rule lives in the judging tools, not here.
STRONG = ("fit_rowid_sha", "features_sha", "n_features")
ROWS = ("total", "fit", "val", "fit_min_season", "fit_max_season",
        "val_season", "test_season")


def _load(tag):
    fs = sorted(glob.glob(f"./model/cat_{tag}_s*.pkl")) or \
         sorted(glob.glob(f"./model/cat_{tag}.pkl"))
    return fs


def _ident(pack):
    """(strong?, comparable identity dict) for one pack."""
    lin = pack.get("lineage")
    if isinstance(lin, dict) and lin.get("fit_rowid_sha"):
        d = {k: lin.get(k) for k in STRONG}
        d.update({f"rows.{k}": (lin.get("rows") or {}).get(k) for k in ROWS})
        d["flags"] = json.dumps(lin.get("flags") or {}, sort_keys=True)
        return True, d
    p = pack["model"].get_all_params()
    return False, {
        "priors_mean": round(
            pack["fpipe"]["priors"]["asof_pitcher_success_rate"], 12),
        "n_features": len(pack["features"]),
        "hyper": tuple(round(p[k], 6) if isinstance(p.get(k), float)
                       else p.get(k) for k in KEYS),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--verbose", action="store_true",
                    help="print the full identity of every tag")
    a = ap.parse_args(argv)

    tags = a.tags or sorted({f.split("cat_")[1].rsplit("_s", 1)[0]
                             for f in glob.glob("./model/cat_*_s*.pkl")})
    rows, bad = [], []
    for t in tags:
        fs = _load(t)
        if not fs:
            print(f"not found: {t}")
            bad.append(t)
            continue
        # Every seed is checked, not just the first. A single mismatched seed
        # is exactly the case this tool exists to catch, and reading fs[0]
        # alone would let it exit 0.
        idents, strongs = [], set()
        for f in fs:
            pack = joblib.load(f)
            s, d = _ident(pack)
            strongs.add(s)
            idents.append(json.dumps(d, sort_keys=True, default=str))
        if len(set(idents)) > 1:
            print(f"!! {t}: its own seeds disagree on the training set")
            for i in sorted(set(idents)):
                print(f"     {i[:160]}")
            bad.append(t)
        rows.append((t, len(fs), all(strongs), json.loads(idents[0]),
                     pack.get("best_iteration"), pack.get("val_bss")))

    print(f"\n{'tag':<14}{'seeds':>6}  {'check':<8}{'identity':<26}"
          f"{'feat':>6}{'best_it':>9}{'val_bss':>10}")
    for t, n, strong, d, bi, vb in rows:
        if strong:
            ident = str(d.get("fit_rowid_sha"))
            era = (f"  fit {d.get('rows.fit_min_season')}-"
                   f"{d.get('rows.fit_max_season')} "
                   f"n={d.get('rows.fit')}")
        else:
            ident = f"{d['priors_mean']:.10f}"
            era = "  (no lineage)"
        nf = d.get("n_features", "?")
        print(f"{t:<14}{n:>6}  {'STRONG' if strong else 'WEAK':<8}{ident:<26}"
              f"{nf:>6}{(bi if bi is not None else -1):>9}"
              f"{(vb if vb is not None else float('nan')):>10.2f}{era}")
        if a.verbose:
            for k, v in sorted(d.items()):
                print(f"                {k}: {str(v)[:150]}")

    # Cross-tag agreement, field by field, so the message names the difference.
    print()
    strong_rows = [r for r in rows if r[2]]
    weak_rows = [r for r in rows if not r[2]]
    mism = []
    if len(rows) > 1:
        keys = set()
        for _, _, _, d, _, _ in rows:
            keys |= set(d)
        common = {k for k in keys
                  if all(k in d for _, _, _, d, _, _ in rows)}
        for k in sorted(common):
            vals = {json.dumps(d[k], sort_keys=True, default=str)
                    for _, _, _, d, _, _ in rows}
            if len(vals) > 1:
                mism.append(k)

    if mism:
        # loss_function is expected to differ between a binary base and a
        # multiclass cell arm; a row set or a feature list is never expected to.
        hard = [k for k in mism if k != "flags" and "hyper" not in k]
        print("!! members differ on: " + ", ".join(mism))
        for k in mism:
            print(f"   {k}")
            for t, _, _, d, _, _ in rows:
                print(f"      {t:<14}{str(d[k])[:130]}")
        if hard:
            print("\n!! that includes a training-set or feature difference. "
                  "These members were not trained on the same data; do not "
                  "blend them or pick weights across them.")
            bad.extend(t for t, *_ in rows)
    elif len(rows) > 1:
        print("identity matches across all members"
              + (" (STRONG)" if not weak_rows else ""))

    if weak_rows:
        print(f"\nWEAK for {len(weak_rows)} tag(s): "
              + ", ".join(t for t, *_ in weak_rows)
              + " -- written before 2026-08-15, no lineage in the pack. The "
                "float is a mean over the fit rows, not an identity: it catches "
                "a changed training set but cannot prove an unchanged one.")

    if bad:
        print(f"\n!! could not verify or mismatched: {', '.join(sorted(set(bad)))}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
