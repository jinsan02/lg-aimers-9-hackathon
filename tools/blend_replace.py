"""같은 계열의 기준 멤버를 후보로 교체할 가치와 반분 전이를 계산한다.

실행: python tools/blend_replace.py VB2_base SK2_k40 ZD5
현행 구성은 base 0.45 + cell 0.55로 고정한다. 후보는 base 자리를 전부 또는
절반 대체하며, 전체 검증 최적 가중을 제출 근거로 사용하지 않는다.
"""

import glob
import sys

import numpy as np
import pandas as pd


CELL_SEEDS = {"42", "7", "13", "3", "4", "5"}


def load(tag, seeds=None):
    fs = sorted(glob.glob(f"./out/cat_{tag}_s*_val_preds.npz"))
    if seeds is not None:
        fs = [f for f in fs if f.rsplit("_s", 1)[1].split("_", 1)[0] in seeds]
    if not fs:
        raise FileNotFoundError(tag)
    out = {}
    y = None
    for f in fs:
        z = np.load(f)
        yy = z["y"].astype(np.float64)
        if y is None:
            y = yy
        elif not np.array_equal(y, yy):
            raise ValueError(f"target mismatch: {f}")
        seed = f.rsplit("_s", 1)[1].split("_", 1)[0]
        out[seed] = z["pred"].astype(np.float64)
    return np.mean(list(out.values()), axis=0), y, out


def score(y, p):
    r = float(y.mean())
    p = p - (p.mean() - r)
    return float(1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean()
                            / (r * (1 - r))))


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    bt, ct, zt = sys.argv[1:]
    b, y, bs = load(bt)
    c, yc, cs = load(ct)
    z, yz, zs = load(zt, CELL_SEEDS)
    if not np.array_equal(y, yc) or not np.array_equal(y, yz):
        raise ValueError("target mismatch across groups")

    common = sorted(set(bs) & set(cs), key=int)
    dif = np.array([score(y, cs[s]) - score(y, bs[s]) for s in common])
    se = dif.std(ddof=1) / np.sqrt(len(dif))
    print(f"paired {ct}-{bt}: mean {dif.mean():+.3f} SE {se:.3f} "
          f"t={dif.mean()/se:+.3f} n={len(dif)}")

    current = .45 * b + .55 * z
    variants = {
        "current": current,
        "replace_base": .45 * c + .55 * z,
        "split_base_50": .225 * b + .225 * c + .55 * z,
        "base_only": b,
        "candidate_only": c,
    }
    cur = score(y, current)
    for name, p in variants.items():
        print(f"{name:<18} {score(y,p):9.3f} delta {score(y,p)-cur:+8.3f}")

    raw = pd.read_csv("./data/train.csv", usecols=["season", "game_month"])
    month = raw.loc[raw.season == 2024, "game_month"].to_numpy()
    if len(month) != len(y):
        raise ValueError(f"2024 metadata mismatch {len(month)} != {len(y)}")
    for label, mask in (("Mar-Jun", month <= 6), ("Jul-Oct", month > 6)):
        base_score = score(y[mask], current[mask])
        print(label)
        for name in ("replace_base", "split_base_50"):
            print(f"  {name:<16} delta "
                  f"{score(y[mask], variants[name][mask])-base_score:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
