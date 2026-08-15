"""Do `season_std` and `skill` still produce the *same numbers* at old flags?

`tools/training_path_diff.py` reports these two modules as changed since
`0169d8d`, the commit the `B1J6_base` judging-surface control was fitted under.
Both changes look inert at default arguments -- `build_anchors` gained
`last_pitch=False` and `skill.build` gained `neutral_mode="missing"`, and the
new code sits inside those branches -- but `season_std` also rewrote its
shrinkage denominators (`n0` -> `n0c`, `sn` -> `snc`), which is *not* inside a
branch. Reading it is not enough: reusing a control whose features have quietly
moved is the v16 (-6.15) and v17 (-53.6) failure.

So both revisions are imported side by side and their outputs compared
numerically on real rows. Byte-identical is not required; **numerically
identical** is.

Run: python tools/feature_path_parity.py [old_rev]
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = sys.argv[1] if len(sys.argv) > 1 else "0169d8d"
MODULES = ("season_std", "skill")
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def load_rev(rev, tmp):
    """Materialise src/*.py at `rev` and import the modules from there."""
    d = os.path.join(tmp, rev.replace("/", "_"))
    os.makedirs(d, exist_ok=True)
    names = subprocess.run(["git", "ls-tree", "--name-only", f"{rev}:src"],
                           capture_output=True, cwd=ROOT).stdout.decode().split()
    for n in names:
        if not n.endswith(".py"):
            continue
        b = subprocess.run(["git", "show", f"{rev}:src/{n}"],
                           capture_output=True, cwd=ROOT).stdout
        with open(os.path.join(d, n), "wb") as f:
            f.write(b)
    mods = {}
    for m in MODULES:
        spec = importlib.util.spec_from_file_location(f"{rev}_{m}",
                                                      os.path.join(d, f"{m}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mods[m] = mod
    return mods


def frame(n=60000):
    cols = ["row_id", "season", "pitcher_id", "batter_id", "game_month",
            "control_success", "balls_before", "strikes_before"]
    cols += [c for c in pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                                    nrows=0, encoding="utf-8-sig").columns
             if c.startswith("asof_")]
    df = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols,
                     encoding="utf-8-sig")
    return df.iloc[::max(1, len(df) // n)].reset_index(drop=True)


def main():
    print(f"comparing src at {OLD} against the working tree\n")
    tmp = tempfile.mkdtemp(prefix="parity_")
    try:
        old = load_rev(OLD, tmp)
        sys.path.insert(0, os.path.join(ROOT, "src"))
        new = {m: __import__(m) for m in MODULES}
        df = frame()
        print(f"  {len(df):,} sampled rows, seasons "
              f"{df.season.min()}-{df.season.max()}\n")

        print("season_std.build_anchors + add_std at the control's flags")
        # B1J6_base: --feat-std --std-k 80 --std-to-prior --std-season-prior
        a_old = old["season_std"].build_anchors(df.copy())
        a_new = new["season_std"].build_anchors(df.copy())
        check("1 build_anchors returns the same keys",
              sorted(a_old) == sorted(a_new), f"{len(a_new)} keys")
        worst, where = 0.0, ""
        for k in sorted(set(a_old) & set(a_new)):
            u, v = a_old[k], a_new[k]
            if isinstance(u, np.ndarray) and isinstance(v, np.ndarray):
                if u.shape != v.shape:
                    check(f"2 {k} same shape", False, f"{u.shape} vs {v.shape}")
                    continue
                d = float(np.nanmax(np.abs(u.astype(float) - v.astype(float)))) \
                    if u.size else 0.0
                if d > worst:
                    worst, where = d, k
        check("2 every anchor array is numerically identical", worst == 0.0,
              f"max |diff| {worst:.3e}" + (f" at {where}" if worst else ""))

        # B1J6_base's flags as fpipe._apply_std passes them: --std-k 80,
        # --std-to-prior (to_career False -- shrink to the league prior, not the
        # career rate), --std-season-prior.
        kw = dict(k=80.0, priors=None, to_career=False, multi_k=(),
                  season_prior=True, excess=False, ratio=False, k_by=None)
        try:
            o = old["season_std"].add_std(df.copy(), a_old, **kw)
            n_ = new["season_std"].add_std(df.copy(), a_new, **kw)
        except Exception as e:
            # `season_prior` is a table built upstream in fpipe, not a flag, so
            # add_std cannot be driven standalone from these arguments. Say so
            # rather than passing: an inconclusive parity check must not read
            # as a clean one, because the caller's next decision (reuse an old
            # control, or fit a fresh one) turns on it.
            print(f"  INCONCLUSIVE  add_std could not be driven standalone: "
                  f"{type(e).__name__}: {e}")
            print("                build_anchors -- the function whose "
                  "shrinkage denominators changed -- IS identical above.")
            print("                Treat end-to-end parity as UNPROVEN and fit "
                  "a same-session fresh control.")
            FAIL.append("add_std parity unproven")
            return 1
        od = o[0] if isinstance(o, tuple) else o
        nd = n_[0] if isinstance(n_, tuple) else n_
        added = [c for c in nd.columns if c not in df.columns]
        check("3 add_std adds the same columns",
              added == [c for c in od.columns if c not in df.columns],
              f"{len(added)} columns")
        w2, wc = 0.0, ""
        for c in added:
            d = float(np.nanmax(np.abs(od[c].to_numpy(float)
                                       - nd[c].to_numpy(float))))
            if d > w2:
                w2, wc = d, c
        check("4 and the same values", w2 == 0.0,
              f"max |diff| {w2:.3e}" + (f" at {wc}" if w2 else ""))

        print("\nskill.build + skill.add at the control's flags (--feat-skill-pc)")
        s_old = old["skill"].build(df.copy())
        s_new = new["skill"].build(df.copy())
        check("5 skill.build returns the same structure",
              type(s_old) is type(s_new))
        so = old["skill"].add(df.copy(), s_old)
        sn = new["skill"].add(df.copy(), s_new)
        sod = so[0] if isinstance(so, tuple) else so
        snd = sn[0] if isinstance(sn, tuple) else sn
        sadd = [c for c in snd.columns if c not in df.columns]
        check("6 skill.add adds the same columns",
              sadd == [c for c in sod.columns if c not in df.columns],
              f"{len(sadd)} columns")
        w3, wc3 = 0.0, ""
        for c in sadd:
            d = float(np.nanmax(np.abs(sod[c].to_numpy(float)
                                       - snd[c].to_numpy(float))))
            if d > w3:
                w3, wc3 = d, c
        check("7 and the same values", w3 == 0.0,
              f"max |diff| {w3:.3e}" + (f" at {wc3}" if w3 else ""))

        print("\n" + ("all passed -- the control's feature path is unchanged, "
                      "so B1J6_base may be reused"
                      if not FAIL else
                      f"{len(FAIL)} FAILED: {FAIL}\n"
                      "The control's features moved. Do NOT reuse it; fit a "
                      "same-session fresh control."))
        return 1 if FAIL else 0
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
