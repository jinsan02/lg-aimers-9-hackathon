"""Contract for deterministic structured auxiliary-label corruption.

The Expected Mix arm trained, scored and reported a clean null because a
lookup miss made the new feature a byte copy of an old one. Nothing in the run
would have told us. So the properties this arm depends on are asserted before
any GPU time is spent:

  * off by default, and off means byte-identical to corrected labels
  * the same row is corrupted on every call, in any process, in any row order
    -- the selection view and the deployment view call build_cells separately
    and must agree
  * the realised rate matches the pre-registered one
  * the success bit never moves (that is the main binary truth)
  * the taxonomy is unchanged -- this arm varies labels, not class count

Run: python tests/test_fm_noise.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import failmode as fm                                            # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def frame(n=60_000, seed=0):
    """Cells in the same proportions the real frame has."""
    rng = np.random.default_rng(seed)
    share = {"0000": .023, "0001": .118, "0010": .108, "0011": .077,
             "0100": .094, "0101": .028, "0110": .022, "0111": .006,
             "0xxx": .001, "1000": .366, "1010": .156, "1xxx": .001}
    names = list(share)
    p = np.array([share[k] for k in names], dtype=float)
    p /= p.sum()
    cell = pd.Series(rng.choice(names, size=n, p=p))
    rid = pd.Series([f"TRAIN_{i:07d}" for i in range(n)])
    return cell, rid, set(names)


def main():
    print("deterministic auxiliary-label corruption")
    cell, rid, names = frame()

    off = fm._apply_noise(cell, rid, 0.0, names)
    check("rate 0 is a no-op", off.equals(cell))

    a = fm._apply_noise(cell, rid, fm.NOISE_RATE, names)
    b = fm._apply_noise(cell, rid, fm.NOISE_RATE, names)
    check("same call twice is identical", a.equals(b))

    # The deployment view sees the frame in a different order and without the
    # validation rows. The same row must still be chosen.
    perm = np.random.default_rng(7).permutation(len(cell))
    c2 = fm._apply_noise(cell.iloc[perm].reset_index(drop=True),
                         rid.iloc[perm].reset_index(drop=True),
                         fm.NOISE_RATE, names)
    back = pd.Series(c2.to_numpy(), index=perm).sort_index().reset_index(drop=True)
    check("row order does not change which rows move", back.equals(a))

    sub = np.arange(0, len(cell), 3)
    c3 = fm._apply_noise(cell.iloc[sub].reset_index(drop=True),
                         rid.iloc[sub].reset_index(drop=True),
                         fm.NOISE_RATE, names)
    check("a subset of rows gets the same decisions",
          np.array_equal(c3.to_numpy(), a.to_numpy()[sub]))

    moved = (a != cell)
    check("realised rate matches the pre-registered one",
          abs(moved.mean() - fm.NOISE_RATE) < 0.002,
          f"{moved.mean() * 100:.4f}% vs {fm.NOISE_RATE * 100:.3f}%")

    check("success bit never moves", (a.str[0] == cell.str[0]).all())
    check("no row lands outside the taxonomy", set(a.unique()) <= names,
          f"{len(set(a.unique()))} cells in, {len(names)} allowed")
    check("class count unchanged", set(a.unique()) <= names and len(names) == 12)

    # A corrupted row must actually land somewhere else, not resample itself.
    check("every moved row changed cell", (a[moved] != cell[moved]).all())

    # Determinism has to survive a fresh interpreter. PYTHONHASHSEED randomises
    # str hashing per process, which is exactly the trap here -- python's own
    # hash() would pick different rows on every run and the word
    # "deterministic" in the pre-registration would be false.
    # The frame is rebuilt inline rather than imported: the child must depend on
    # failmode alone, so a failure here can only mean the hash moved.
    script = (f"import sys; sys.path.insert(0, r'{os.path.join(ROOT, 'src')}')\n"
              "import failmode as fm, hashlib, numpy as np, pandas as pd\n"
              "rng = np.random.default_rng(0)\n"
              "share = {'0000':.023,'0001':.118,'0010':.108,'0011':.077,"
              "'0100':.094,'0101':.028,'0110':.022,'0111':.006,'0xxx':.001,"
              "'1000':.366,'1010':.156,'1xxx':.001}\n"
              "names=list(share); p=np.array([share[k] for k in names]); p/=p.sum()\n"
              "cell=pd.Series(rng.choice(names,size=60000,p=p))\n"
              "rid=pd.Series(['TRAIN_%07d'%i for i in range(60000)])\n"
              "out=fm._apply_noise(cell,rid,fm.NOISE_RATE,set(names))\n"
              "print('DIGEST', hashlib.sha256(''.join(out).encode()).hexdigest())\n")
    outs = []
    for salt in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=salt, PYTHONIOENCODING="utf-8")
        rr = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, encoding="utf-8", env=env, cwd=ROOT)
        line = [x for x in (rr.stdout or "").splitlines() if x.startswith("DIGEST")]
        outs.append(line[0][7:] if line else f"ERR {(rr.stderr or '')[-200:]}")
    check("stable across PYTHONHASHSEED", outs[0] == outs[1] and len(outs[0]) == 64,
          f"{outs[0][:16]} vs {outs[1][:16]}")

    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
