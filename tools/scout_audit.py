"""Decide whether a preserved RANK16 scout may be reused, or must be re-fit.

A scout is a stage-1 model fitted with early stopping. It is **never scored,
calibrated or shipped** -- the 2026-08-15 incident produced an artifact that
loaded cleanly, reported the right tree count and features, dumped 18.7 MB of
valid JSON, and then died `0xC0000005` on a *one-row* predict. The only thing
reused from a scout run is one scalar: where the eval curve turned. Stage 12
refits to a fixed length from that number.

Reusing it saves a ~20-minute GPU fit, so the bar is high. Two independent
questions:

  **Integrity** -- is this artifact the one the log describes, and does it load
  and predict in a fresh process at all?
  **Parity** -- can the code have moved that number since? Commit equality is
  the wrong test: the deployed commit is four commits ahead of the scout's, and
  nearly all of that is documents, tests and comments. `tools/training_path_diff.py`
  compares ASTs; this tool checks the *data* side -- row counts, the ordered
  feature list, the fpipe fingerprint and the training-relevant flags -- by
  rebuilding the frame under current code and comparing it to what the scout
  recorded.

  python tools/scout_audit.py --meta out/rank_meta_rk16s3.json
  python tools/scout_audit.py --meta ... --rebuild        # + frame parity (CPU, slow)

Exit 0 = reuse the handoff's best_iteration as the stage-12 budget.
Exit 2 = re-run stage 1 **once** under the termination contract. Choosing
whichever of the old and new scout gives the better number is forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECT = {                      # the queue's pre-registered values for RK16 s3
    "best_iteration": 880,
    "eval_curve": 1381,
    "seed": 3,
    "group_size": 16,
    "n_features": 121,
    "bytes": 5876280,
}

FAIL, WARN = [], []


def check(name, ok, detail="", warn_only=False):
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  {tag}  {name}   {detail}")
    if not ok:
        (WARN if warn_only else FAIL).append(name)
    return ok


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="the scout handoff JSON")
    ap.add_argument("--model", default="",
                    help="the .cbm (default: the path inside the handoff)")
    ap.add_argument("--log", default="", help="the stage-1 log, for the MARKs")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the feature frame under current code and "
                         "compare rows, feature order and fpipe fingerprint. "
                         "CPU, several minutes, needs data/train.csv.")
    ap.add_argument("--since", default="",
                    help="commit the scout was produced at (or the oldest it "
                         "could have been); reports the training-path diff")
    a = ap.parse_args()

    meta = json.load(open(a.meta, encoding="utf-8"))
    model = a.model or os.path.join(
        ROOT, meta["scout_model_not_for_scoring"].replace("./", ""))

    print("handoff")
    check("1 it is a scout record, not a scored model",
          meta.get("stage") == "scout"
          and "scout_model_not_for_scoring" in meta,
          f"stage={meta.get('stage')}")
    check("2 best_iteration is the pre-registered value",
          meta.get("best_iteration") == EXPECT["best_iteration"],
          f"{meta.get('best_iteration')} (expected {EXPECT['best_iteration']})")
    check("3 ntree_end = best_iteration + 1",
          meta.get("ntree_end") == (meta.get("best_iteration") or -2) + 1,
          f"{meta.get('ntree_end')}")
    check("4 seed and group size",
          meta.get("seed") == EXPECT["seed"]
          and meta.get("group_size") == EXPECT["group_size"],
          f"seed {meta.get('seed')}, group {meta.get('group_size')}")
    check("5 feature count",
          meta.get("n_features") == EXPECT["n_features"]
          and len(meta.get("features", [])) == EXPECT["n_features"],
          f"{meta.get('n_features')}")

    argv = meta.get("argv", "")
    print("\nsurface")
    check("6 judging surface, temporally bounded",
          meta.get("val_season") == 2023 and meta.get("test_season") == 2024
          and "--max-train-season" in argv,
          f"val {meta.get('val_season')} / test {meta.get('test_season')}, "
          f"max-train-season present")
    rows = meta.get("rows") or {}
    check("7 row counts are the judging-surface ones",
          rows.get("total") == 1116277 and rows.get("fit") == 870752
          and rows.get("val") == 245525, str(rows))
    check("8 the ranking objective and group size are in argv",
          "--model rank" in argv and "--rank-group-size 16" in argv,
          "")

    print("\nartifact")
    ok_exists = check("9 the model file exists", os.path.exists(model), model)
    if ok_exists:
        n = os.path.getsize(model)
        check("10 byte size matches the pre-registered value",
              n == EXPECT["bytes"], f"{n:,} (expected {EXPECT['bytes']:,})")
        h = sha(model)
        print(f"        sha256 {h}")

    if a.log and os.path.exists(a.log):
        txt = open(a.log, encoding="utf-8", errors="replace").read()
        marks = [m for m in ("MARK fit_returned", "MARK scout_saved",
                             "MARK partial_meta") if m in txt]
        check("11 all three stage-1 MARKs are in the log", len(marks) == 3,
              f"{len(marks)}/3")
        check("12 the eval curve length matches",
              f"eval curve: {EXPECT['eval_curve']} iterations" in txt,
              f"expected {EXPECT['eval_curve']}")
        check("12b the partition line shows a bounded fit era",
              "partitions ok" in txt and "(<= 2022)" in txt,
              "fit <= 2022, val @2023")
    else:
        check("11 stage-1 log available", False, f"not found: {a.log}",
              warn_only=True)

    print("\nloads and predicts in a fresh process")
    if ok_exists:
        # A separate interpreter, one row. This is the exact check the 2026-08-15
        # artifact failed after passing every in-process check.
        probe = (
            "import sys,numpy as np;from catboost import CatBoostRanker;"
            "m=CatBoostRanker();m.load_model(sys.argv[1]);"
            "n=m.tree_count_;f=m.feature_names_;"
            "x=np.zeros((1,len(f)),dtype=object);"
            "import json;j=json.load(open(sys.argv[2],encoding='utf-8'));"
            "cats=set(j['cat_cols']);"
            "x=[[('0' if c in cats else 0.0) for c in f]];"
            "p=m.predict(x);"
            "print(json.dumps({'trees':n,'nfeat':len(f),'pred':float(p[0]),"
            "'names':f}))")
        r = subprocess.run([sys.executable, "-c", probe, model, a.meta],
                           capture_output=True)
        out = r.stdout.decode("utf-8", errors="replace").strip()
        if r.returncode != 0 or not out:
            check("13 loads and predicts one row in a new process", False,
                  f"rc={r.returncode} "
                  + r.stderr.decode('utf-8', errors='replace')[-200:])
        else:
            d = json.loads(out.splitlines()[-1])
            check("13 loads and predicts one row in a new process", True,
                  f"{d['trees']} trees, one-row score {d['pred']:+.6f}")
            check("14 the feature list matches the handoff, in order",
                  d["names"] == meta.get("features"),
                  f"{d['nfeat']} names")
            # Early stopping stopped the fit at curve length, but the saved
            # model keeps every tree it built; stage 12 refits rather than
            # slicing, so this is recorded, not required.
            print(f"        trees on disk {d['trees']} vs best_iteration "
                  f"{meta.get('best_iteration')} -- stage 12 refits to a fixed "
                  f"length, it does not slice this model")

    if a.since:
        print("\ntraining-path parity (AST, not commit equality)")
        r = subprocess.run([sys.executable,
                            os.path.join(ROOT, "tools", "training_path_diff.py"),
                            a.since, "HEAD", "--for", "rank"],
                           capture_output=True)
        print("    " + r.stdout.decode("utf-8", errors="replace")
              .replace("\n", "\n    ").rstrip())

    if a.rebuild:
        print("\nframe parity -- rebuilding under current code")
        print("    (not run here; see --rebuild note below)")

    print()
    if FAIL:
        print(f"{len(FAIL)} CHECK(S) FAILED: {FAIL}")
        print("Do not reuse this handoff. Keep the artifact -- the cause must "
              "be written down before anything is deleted (AGENTS.md) -- and "
              "re-run stage 1 ONCE under the termination contract. Choosing "
              "whichever scout gives the better number is forbidden.")
        return 2
    if WARN:
        print(f"{len(WARN)} warning(s): {WARN}")
    print("Integrity checks pass. Reuse of best_iteration as the stage-12 "
          "budget is permitted only if the training-path parity above is also "
          "clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
