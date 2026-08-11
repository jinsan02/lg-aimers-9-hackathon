"""Paired seed report for either val or test prediction surfaces."""

import argparse
import glob
import os
import re

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(tag, kind):
    fs = glob.glob(os.path.join(ROOT, "out", f"cat_{tag}_s*_{kind}_preds.npz"))
    out, y = {}, None
    for f in fs:
        m = re.search(r"_s(\d+)_", os.path.basename(f))
        if not m:
            continue
        z = np.load(f)
        yy = z["y"].astype(float)
        if y is not None and not np.array_equal(y, yy):
            raise ValueError(f"target mismatch in {tag}")
        y = yy
        out[int(m.group(1))] = z["pred"].astype(float)
    if not out:
        raise FileNotFoundError(f"{tag}/{kind}")
    return out, y


def bss(y, p, center=True):
    p = np.asarray(p, float).copy()
    if center:
        p += y.mean()-p.mean()
    return 1e5*(1-np.mean((np.clip(p,0,1)-y)**2)/(y.mean()*(1-y.mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("alt")
    ap.add_argument("--kind", choices=["val", "test"], default="val")
    args = ap.parse_args()
    b, y = load(args.base, args.kind)
    a, ya = load(args.alt, args.kind)
    if not np.array_equal(y, ya):
        raise ValueError("base/alt target mismatch")
    seeds = sorted(set(b)&set(a))
    delta = np.asarray([bss(y,a[s])-bss(y,b[s]) for s in seeds])
    se = delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta)>1 else np.nan
    t = delta.mean()/se if np.isfinite(se) and se else np.nan
    be = bss(y,np.mean([b[s] for s in seeds],0))
    ae = bss(y,np.mean([a[s] for s in seeds],0))
    print(f"{args.kind} seeds={seeds} n={len(y):,}")
    print("paired " + " ".join(f"s{s}:{d:+.3f}" for s,d in zip(seeds,delta)))
    print(f"mean={delta.mean():+.3f} SE={se:.3f} t={t:+.3f} "
          f"ensemble={be:.3f}->{ae:.3f} delta={ae-be:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
