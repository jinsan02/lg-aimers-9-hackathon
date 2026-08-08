"""같은 검증 표면의 val_preds 태그들을 시드 페어로 비교한다."""

import glob
import re
import sys

import numpy as np


def load(tag):
    preds = {}
    y = None
    for path in glob.glob(f"./out/*_{tag}_s*_val_preds.npz"):
        seed = int(re.search(r"_s(\d+)_", path).group(1))
        z = np.load(path, allow_pickle=True)
        preds[seed] = z["pred"].astype(np.float64)
        y = z["y"].astype(np.float64)
    return preds, y


def score(pred, y):
    rate = y.mean()
    pred = pred - (pred.mean() - rate)
    return 1e5 * (1 - np.mean((np.clip(pred, 0, 1) - y) ** 2) / (rate * (1-rate)))


def main(tags):
    if len(tags) < 2:
        raise SystemExit("사용: python tools/val_surf_report.py BASE CAND...")
    base, y = load(tags[0])
    if not base:
        raise FileNotFoundError(tags[0])
    base_ens = score(np.mean(list(base.values()), axis=0), y)
    print(f"기준 {tags[0]} n={len(base)} 앙상블={base_ens:.3f}")
    for tag in tags[1:]:
        cand, cy = load(tag)
        common = sorted(set(base) & set(cand))
        if not common:
            print(f"{tag}: 공통 시드 없음")
            continue
        delta = np.array([score(cand[s], cy) - score(base[s], y) for s in common])
        se = delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else float("nan")
        ens = score(np.mean([cand[s] for s in common], axis=0), cy)
        print(f"{tag}: n={len(common)} 앙상블={ens:.3f} Δ={ens-base_ens:+.3f} "
              f"페어={delta.mean():+.3f} SE={se:.3f} t={delta.mean()/se:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
