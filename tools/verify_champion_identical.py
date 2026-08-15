"""Prove the shipped champion still predicts **bit-identically** after a change.

`fpipe.predict` is the submission path. On 2026-08-15 it was rewritten to
dispatch on model kind (a multilabel pack had been returning P(middle) as
P(success), and a ranker pack raised AttributeError), and `fpipe.transform`
began reading `feat_k` from the artifact. Both are meant to be no-ops for the
champion's 14 members. "Meant to be" is not a check.

This runs each member twice in **separate processes** -- once against the code
inside `submissions/<zip>` exactly as the organiser will run it, once against
the working tree -- and requires `np.array_equal`, not `allclose`. Two
processes because the zip ships its own `fpipe.py`, `features.py`,
`season_std.py`, `skill.py` and `target_enc.py`, and importing both copies into
one interpreter silently resolves to whichever landed on `sys.path` first.

  python tools/verify_champion_identical.py
  python tools/verify_champion_identical.py --zip submissions/b1s8_20260813.zip --rows 400

Exit 2 if any member differs by a single bit.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs inside each subprocess. `src_dir` holds the fpipe.py under test and
# `model_dir` the packs; both come from the same source so a member is never
# scored by the other side's pipeline.
WORKER = r'''
import json, os, sys
import joblib, numpy as np, pandas as pd
src_dir, model_dir, out_path, rows = sys.argv[1:5]
sys.path.insert(0, src_dir)
import fpipe
hdr = list(pd.read_csv(os.path.join(r"{root}", "data", "test.csv"), nrows=0,
                       encoding="utf-8-sig").columns)
raw = pd.read_csv(os.path.join(r"{root}", "data", "train.csv"),
                  encoding="utf-8-sig", usecols=hdr)
test = raw[raw.season == 2024].head(int(rows))[hdr].reset_index(drop=True)
out = {{}}
for fn in sorted(os.listdir(model_dir)):
    if not fn.endswith(".pkl"):
        continue
    pack = joblib.load(os.path.join(model_dir, fn))
    out[fn] = fpipe.predict(dict(pack), test.copy()).astype(np.float64)
    del pack
np.savez(out_path, **out)
'''


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="submissions/b1s8_20260813.zip")
    ap.add_argument("--rows", type=int, default=400)
    a = ap.parse_args()

    zpath = os.path.join(ROOT, a.zip)
    if not os.path.exists(zpath):
        raise SystemExit(f"no such package: {zpath}")

    tmp = tempfile.mkdtemp(prefix="champver_")
    try:
        zsrc, zmod = os.path.join(tmp, "zsrc"), os.path.join(tmp, "zsrc", "model")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(zsrc)
        members = sorted(f for f in os.listdir(zmod) if f.endswith(".pkl"))
        print(f"{a.zip}: {len(members)} model members, {a.rows} rows\n")

        # The working tree must hold the same artifacts, byte for byte, or the
        # comparison is between two different models rather than two pipelines.
        print(f"{'member':<30}{'zip sha':>18}{'working tree':>18}  same")
        same_files = True
        for m in members:
            zh = sha(os.path.join(zmod, m))
            wp = os.path.join(ROOT, "model", m)
            wh = sha(wp) if os.path.exists(wp) else "(absent)"
            ok = zh == wh
            same_files &= ok
            print(f"{m:<30}{zh:>18}{wh:>18}  {'yes' if ok else 'NO'}")

        worker = WORKER.format(root=ROOT)
        wpath = os.path.join(tmp, "worker.py")
        with open(wpath, "w", encoding="utf-8") as f:
            f.write(worker)

        runs = {}
        for label, src_dir in (("zip", zsrc), ("tree", os.path.join(ROOT, "src"))):
            out = os.path.join(tmp, f"{label}.npz")
            r = subprocess.run([sys.executable, wpath, src_dir, zmod, out,
                                str(a.rows)], cwd=ROOT, capture_output=True,
                               text=True)
            if r.returncode != 0:
                print(f"\n{label} worker failed:\n{r.stdout}\n{r.stderr}")
                return 2
            runs[label] = np.load(out)

        print(f"\n{'member':<30}{'zip mean':>12}{'tree mean':>12}"
              f"{'max |diff|':>13}  identical")
        bad = []
        for m in members:
            zp, tp = runs["zip"][m], runs["tree"][m]
            ident = np.array_equal(zp, tp)
            if not ident:
                bad.append(m)
            print(f"{m:<30}{zp.mean():>12.8f}{tp.mean():>12.8f}"
                  f"{np.abs(zp - tp).max():>13.2e}  "
                  f"{'yes' if ident else 'NO'}")

        # The blend is what actually ships, so check it too: a per-member
        # rounding difference that cancels in the mean would still change it.
        def blend(z):
            b = [z[m] for m in members if "_base_" in m]
            c = [z[m] for m in members if "_cell_" in m]
            return 0.45 * np.mean(b, 0) + 0.55 * np.mean(c, 0)

        bz, bt = blend(runs["zip"]), blend(runs["tree"])
        bl_ok = np.array_equal(bz, bt)
        print(f"\n0.45*base + 0.55*cell   zip {bz.mean():.10f}   "
              f"tree {bt.mean():.10f}   identical {'yes' if bl_ok else 'NO'}")

        if bad or not bl_ok:
            print(f"\n!! {len(bad)} member(s) differ: {bad}")
            print("   The submission path changed the champion. Do not ship.")
            return 2
        if not same_files:
            print("\n!! artifacts in model/ differ from the zip -- the pipelines "
                  "agree but they were compared on the zip's copies only.")
            return 2
        print(f"\nall {len(members)} members and the blend are bit-identical "
              f"through the current fpipe. The champion is unchanged.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
