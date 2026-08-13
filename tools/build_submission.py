"""Single builder: one command turns a named member set into a verified zip.

Audit section 8 asked for exactly this. Until now packaging was manual, and the
manual steps were where things went wrong -- the constants generator wrote to
`out/` while the script read `model/`, so a hand copy sat in the middle of every
build and would have shipped last week's constants the first time it was
skipped.

What it refuses to do:
  * package a member set with a seed missing -- the seed list is declared, not
    discovered by glob;
  * package a `failmode.py`, which is train-only (the same algebra on test rows
    recovers 2025 targets, 96.79% verified);
  * package a pkl whose training-set fingerprint disagrees with its siblings;
  * finish without running the script end to end and checking its output.

  python tools/build_submission.py --script src/script_blend_b1s.py \
      --members B1S_base:3,4,5,6,8,13 B1S_cell:3,4,5,6,8,13 \
      --out submissions/b1s_YYYYMMDD.zip
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Everything fpipe.transform can reach. failmode is deliberately absent.
MODULES = ["fpipe.py", "features.py", "season_std.py", "skill.py", "target_enc.py"]
BANNED = {"failmode.py"}
REQUIREMENTS = "catboost==1.2.10\n"
LIMIT_BYTES = 10 * 1024 ** 3          # the platform's 10GB submission cap


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def collect(members):
    """[(tag, [seeds])] -> [pkl paths], asserting every named seed exists."""
    paths, fps = [], {}
    for spec in members:
        tag, _, seeds = spec.partition(":")
        want = [s.strip() for s in seeds.split(",") if s.strip()]
        if not want:
            raise SystemExit(f"{tag}: no seeds given. The seed set is declared, not globbed.")
        for s in want:
            p = os.path.join(ROOT, "model", f"cat_{tag}_s{s}.pkl")
            if not os.path.exists(p):
                raise SystemExit(f"missing member: {p}")
            pack = joblib.load(p)
            art = pack.get("fpipe") or {}
            fp = (art.get("priors") or {}).get("asof_pitcher_success_rate")
            fps.setdefault(tag, {})[s] = fp
            paths.append(p)
    for tag, d in fps.items():
        uniq = {round(v, 10) for v in d.values() if v is not None}
        if len(uniq) > 1:
            raise SystemExit(f"{tag}: seeds disagree on the training-set "
                             f"fingerprint {sorted(uniq)}")
        print(f"  {tag}: {len(d)} seeds, fingerprint {uniq.pop() if uniq else 'n/a'}")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--constants", default="model/matchup_constants_2024.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-smoke", action="store_true",
                    help="package without running the script. Use only when the "
                         "same script already passed a smoke on this member set.")
    a = ap.parse_args()

    print("members")
    pkls = collect(a.members)

    stage = os.path.join(ROOT, "out", "_pkg")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, "model"))
    shutil.copy(os.path.join(ROOT, a.script), os.path.join(stage, "script.py"))
    for m in MODULES:
        shutil.copy(os.path.join(ROOT, "src", m), os.path.join(stage, m))
    for p in pkls:
        shutil.copy(p, os.path.join(stage, "model", os.path.basename(p)))
    cpath = os.path.join(ROOT, a.constants)
    if not os.path.exists(cpath):
        raise SystemExit(f"constants missing: {cpath}. The generator writes to "
                         f"model/; if it is only in out/, regenerate rather than copy.")
    shutil.copy(cpath, os.path.join(stage, "model", os.path.basename(cpath)))
    with open(os.path.join(stage, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(REQUIREMENTS)

    for root, _, files in os.walk(stage):
        for f in files:
            if f in BANNED:
                raise SystemExit(f"refusing to package {f}: train-only module")

    if not a.skip_smoke:
        print("\nsmoke: running the packaged script exactly as the server would")
        # The server mounts ./data next to script.py. Stage the two small public
        # files so the run exercises the real path resolution rather than a
        # rewritten one; they are removed before the zip is written.
        os.makedirs(os.path.join(stage, "data"), exist_ok=True)
        for f in ("test.csv", "sample_submission.csv"):
            shutil.copy(os.path.join(ROOT, "data", f),
                        os.path.join(stage, "data", f))
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, "script.py"], cwd=stage,
                           capture_output=True, text=True, env=env)
        sys.stdout.write(r.stdout[-2000:])
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-4000:])
            raise SystemExit(f"smoke failed, return code {r.returncode}")
        out = os.path.join(stage, "output", "submission.csv")
        if not os.path.exists(out):
            raise SystemExit("smoke produced no ./output/submission.csv")
        sub = pd.read_csv(out)
        samp = pd.read_csv(os.path.join(ROOT, "data", "sample_submission.csv"),
                           encoding="utf-8-sig")
        v = sub.iloc[:, 1].to_numpy(float)
        if len(sub) != len(samp):
            raise SystemExit(f"row count {len(sub)} != sample {len(samp)}")
        if set(sub.iloc[:, 0]) != set(samp.iloc[:, 0]):
            raise SystemExit("submission ids do not match the sample")
        if not np.isfinite(v).all() or v.min() < 0 or v.max() > 1:
            raise SystemExit(f"values out of range: min {v.min()} max {v.max()}")
        print(f"  rows {len(sub):,} | mean {v.mean():.6f} | "
              f"min {v.min():.6f} max {v.max():.6f} | finite ok")
        first = dict(zip(sub.iloc[:, 0], v))

        # Row independence, checked on the packaged artefact rather than the
        # source tree. A feature built from other rows of test.csv is
        # disqualifying, and this is the only place that property can be
        # verified on what actually ships.
        #
        # Two checks, because they catch different things. Reversing the rows
        # catches anything order-dependent (rolling, lag, shift). It does NOT
        # catch an aggregate over the whole frame -- a groupby mean, a global
        # quantile, a frequency count are all order-invariant and would sail
        # through. Dacon's 2026-08-13 notice states the criterion as *set*
        # membership, not order:
        #
        #     the prediction for a row must be identical whether test.csv holds
        #     that row alone or the whole evaluation set
        #
        # so the second check drops rows instead of reordering them. The script
        # refuses an id set that disagrees with the sample, which is correct
        # behaviour, so the sample is subset alongside it.
        t0 = pd.read_csv(os.path.join(stage, "data", "test.csv"), encoding="utf-8-sig")
        s0 = pd.read_csv(os.path.join(stage, "data", "sample_submission.csv"),
                         encoding="utf-8-sig")
        id_col = t0.columns[0]

        def rerun(frame, label):
            """Write `frame` as test.csv (+ matching sample), rerun, read back."""
            frame.to_csv(os.path.join(stage, "data", "test.csv"), index=False,
                         encoding="utf-8")
            s0[s0.iloc[:, 0].isin(set(frame[id_col]))].to_csv(
                os.path.join(stage, "data", "sample_submission.csv"),
                index=False, encoding="utf-8")
            rr = subprocess.run([sys.executable, "script.py"], cwd=stage,
                                capture_output=True, text=True, env=env)
            if rr.returncode != 0:
                sys.stderr.write(rr.stderr[-4000:])
                raise SystemExit(f"row-independence rerun failed ({label})")
            d = pd.read_csv(out)
            return dict(zip(d.iloc[:, 0], d.iloc[:, 1].to_numpy(float)))

        cases = [("reversed order", t0.iloc[::-1])]
        if len(t0) >= 4:
            cases.append(("half the rows", t0.iloc[::2]))
        if len(t0) >= 2:
            cases.append(("one row alone", t0.iloc[[0]]))

        for label, frame in cases:
            got = rerun(frame, label)
            common = set(first) & set(got)
            if len(common) != len(frame):
                raise SystemExit(f"row-independence rerun ({label}) returned "
                                 f"{len(common)} of {len(frame)} expected ids")
            drift = max(abs(first[k] - got[k]) for k in common)
            if drift > 1e-12:
                raise SystemExit(
                    f"NOT row independent: with {label}, a prediction moved by "
                    f"{drift:.3e}. Another row of test.csv is reaching the "
                    f"inference -- that is a disqualifying feature.")
            print(f"  row independence ok ({label}, {len(common)} rows, "
                  f"max drift {drift:.1e})")

        if len(t0) < 200:
            print(f"  note: the public test.csv is {len(t0)} rows, too few for a "
                  f"per-player aggregate to show. Run "
                  f"tools/audit_subset_independence.py for the strong version.")
        shutil.rmtree(os.path.join(stage, "output"), ignore_errors=True)
        shutil.rmtree(os.path.join(stage, "data"), ignore_errors=True)

    dest = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(stage):
            # The smoke run leaves __pycache__ behind. Shipping it is dead
            # weight at best and a stale bytecode shadow at worst.
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in sorted(files):
                if f.endswith(".pyc"):
                    continue
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, stage))
    size = os.path.getsize(dest)
    if size > LIMIT_BYTES:
        os.remove(dest)
        raise SystemExit(f"zip is {size / 1e9:.2f} GB, over the 10 GB limit")
    names = zipfile.ZipFile(dest).namelist()
    if "script.py" not in names:
        raise SystemExit("script.py is not at the zip root")
    print(f"\nwrote {a.out}  {size / 1e6:.1f} MB, {len(names)} entries")
    print(f"  sha256 {sha(dest)}")
    shutil.rmtree(stage, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
