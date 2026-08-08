"""다중 모델 블렌드 최적화 — Caruana greedy ensemble selection (복원추출).

사용: python src/eval_blend_multi.py out/cat_e12_val_preds.npz out/lgbm_v3_val_preds.npz ...
각 npz: y + (pred 또는 ensemble) 키. 2024 검증셋 동일 행 순서 전제.
"""

import sys

import numpy as np


def bss(y, p):
    r = y.mean()
    return float(max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))))


def main():
    paths = sys.argv[1:]
    assert len(paths) >= 2
    y = None
    names, preds = [], []
    for p in paths:
        d = np.load(p)
        pr = d["pred"] if "pred" in d else d["ensemble"]
        if y is None:
            y = d["y"]
        else:
            assert np.allclose(y, d["y"]), f"검증셋 불일치: {p}"
        names.append(p.split("/")[-1].replace("_val_preds.npz", ""))
        preds.append(pr.astype(np.float64))
        print(f"{names[-1]:24s} BSS {bss(y, pr):8.2f}")

    # Caruana greedy (복원추출 50스텝)
    chosen = []
    cur_sum = np.zeros_like(y, dtype=np.float64)
    best_hist = []
    for step in range(50):
        best_i, best_s = None, -np.inf
        for i, pr in enumerate(preds):
            cand = (cur_sum + pr) / (len(chosen) + 1)
            s = bss(y, cand)
            if s > best_s:
                best_i, best_s = i, s
        chosen.append(best_i)
        cur_sum += preds[best_i]
        best_hist.append(best_s)
        if step > 5 and best_hist[-1] <= best_hist[-2] - 1e-9:
            chosen.pop()
            cur_sum -= preds[best_i]
            break

    weights = np.bincount(chosen, minlength=len(preds)) / len(chosen)
    final = cur_sum / len(chosen)
    print(f"\nGREEDY ENSEMBLE: BSS {bss(y, final):.2f} ({len(chosen)}스텝)")
    for n, w in sorted(zip(names, weights), key=lambda t: -t[1]):
        if w > 0:
            print(f"  {n:24s} w={w:.3f}")
    np.savez_compressed("out/blend_weights.npz", names=np.array(names),
                        weights=weights)


if __name__ == "__main__":
    main()
