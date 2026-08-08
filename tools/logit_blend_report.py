"""확률 평균과 로짓 평균을 같은 멤버·가중에서 비교한다.

검증 예측만 읽는 분석 도구다. 제출 스크립트는 바꾸지 않는다.

실행 예:
  python tools/logit_blend_report.py \
    --base VB2_base --cell ZD5 --weight 0.55 \
    --base-seeds 42,7,13,3,4,5,6,8 --cell-seeds 42,7,13,3,4,5
"""

import argparse
import glob
import re

import numpy as np


def load(tag, seeds):
    wanted = set(seeds)
    preds = []
    y = None
    used = []
    for path in glob.glob(f"./out/*_{tag}_s*_val_preds.npz"):
        seed = int(re.search(r"_s(\d+)_", path).group(1))
        if seed not in wanted:
            continue
        z = np.load(path, allow_pickle=True)
        if y is None:
            y = z["y"].astype(np.float64)
        elif not np.array_equal(y, z["y"]):
            raise ValueError(f"검증 라벨 불일치: {path}")
        preds.append(z["pred"].astype(np.float64))
        used.append(seed)
    if set(used) != wanted:
        raise FileNotFoundError(f"{tag}: 요청 {sorted(wanted)}, 발견 {sorted(used)}")
    return np.mean(preds, axis=0), y


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bss(p, y, center=False):
    if center:
        p = p - (p.mean() - y.mean())
    base = y.mean() * (1 - y.mean())
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2) / base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--weight", type=float, required=True)
    ap.add_argument("--base-seeds", required=True)
    ap.add_argument("--cell-seeds", required=True)
    args = ap.parse_args()
    parse = lambda x: [int(v) for v in x.split(",") if v.strip()]  # noqa: E731
    pb, y = load(args.base, parse(args.base_seeds))
    pc, yc = load(args.cell, parse(args.cell_seeds))
    if not np.array_equal(y, yc):
        raise ValueError("두 멤버의 검증 라벨 불일치")

    w = args.weight
    prob = (1 - w) * pb + w * pc
    odds = sigmoid((1 - w) * logit(pb) + w * logit(pc))
    print(f"base={args.base} cell={args.cell} w={w:.3f} n={len(y):,}")
    print(f"{'연산자':<12}{'raw BSS':>12}{'편향제거':>12}{'평균':>10}{'sd':>10}{'범위':>21}")
    for name, pred in (("probability", prob), ("logit", odds)):
        print(f"{name:<12}{bss(pred, y):>12.3f}{bss(pred, y, True):>12.3f}"
              f"{pred.mean():>10.5f}{pred.std():>10.5f}"
              f"  {pred.min():.5f}~{pred.max():.5f}")
    print(f"logit - probability: raw {bss(odds, y)-bss(prob, y):+.3f} | "
          f"편향제거 {bss(odds, y, True)-bss(prob, y, True):+.3f}")


if __name__ == "__main__":
    main()
