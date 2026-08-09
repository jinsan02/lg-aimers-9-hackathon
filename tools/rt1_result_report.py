"""RT1 단일시드 결과를 동일 머신 기준선과 세그먼트별로 분해한다."""

import argparse
import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import roster_transition as rt


def pred(path):
    paths = sorted(glob.glob(path))
    if not paths:
        raise FileNotFoundError(path)
    vals = [np.asarray(np.load(p)["pred"], dtype=float) for p in paths]
    if len({len(v) for v in vals}) != 1:
        raise ValueError(f"예측 길이 불일치: {paths}")
    print(f"prediction ensemble: {path} -> {len(paths)} files")
    return np.mean(vals, axis=0)


def score(p, y):
    den = float(y.mean() * (1.0 - y.mean()))
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    d = d[d.season == args.season].reset_index(drop=True)
    y = d.control_success.to_numpy(dtype=float)
    pb, pa = pred(args.base), pred(args.alt)
    if not (len(d) == len(pb) == len(pa)):
        raise ValueError(f"길이 불일치 data/base/alt={len(d)}/{len(pb)}/{len(pa)}")

    pack = joblib.load(args.model)
    table = pack["fpipe"]["roster"]
    d, _ = rt.add_features(d, table)
    status = np.full(len(d), "other", dtype=object)
    status[d.rt_seen_before.to_numpy() == 0] = "unseen"
    status[d.rt_same_cont.to_numpy() == 1] = "same_cont"
    status[d.rt_same_return.to_numpy() == 1] = "same_return"
    status[d.rt_f_to_r.to_numpy() == 1] = "F_to_R"
    status[d.rt_r_to_f.to_numpy() == 1] = "R_to_F"
    d["rt_status"] = status
    d["base"] = pb
    d["alt"] = pa
    d["gain_sq"] = (pb - y) ** 2 - (pa - y) ** 2

    den = float(y.mean() * (1.0 - y.mean()))
    sb, sa = score(pb, y), score(pa, y)
    pbc = pb - (pb.mean() - y.mean())
    pac = pa - (pa.mean() - y.mean())
    print(f"overall n={len(y):,} base={sb:.3f} alt={sa:.3f} delta={sa-sb:+.3f}")
    print(f"mean actual={y.mean():.6f} base={pb.mean():.6f} alt={pa.mean():.6f} "
          f"rms={np.sqrt(np.mean((pa-pb)**2)):.6f}")
    print(f"centered base={score(pbc,y):.3f} alt={score(pac,y):.3f} "
          f"delta={score(pac,y)-score(pbc,y):+.3f}")

    def report(keys, title):
        rows = []
        for key, g in d.groupby(keys, observed=True, sort=False):
            n = len(g)
            contribution = 1e5 * g.gain_sq.sum() / (len(d) * den)
            rows.append((key, n, g.control_success.mean(), g.base.mean(),
                         g.alt.mean(), contribution))
        out = pd.DataFrame(rows, columns=["segment", "n", "actual", "base",
                                         "alt", "delta_contrib"])
        out = out.sort_values("delta_contrib", ascending=False)
        print(f"\n=== {title} (delta_contrib 합=전체 delta) ===")
        print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    report(["game_type", "rt_status"], "league x roster status")
    d["half"] = np.where(d.game_month <= 6, "month<=6", "month>=7")
    report(["half"], "season halves")


if __name__ == "__main__":
    main()
