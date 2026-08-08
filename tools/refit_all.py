"""refit 배수 1.5 vs 2.0 — **세 배치를 합산**해 판정한다.

배치마다 시드가 다르다: RN(3,4,5,6,8,13) · RQ(21~26) · RS(31~36).
전부 4070, 전부 같은 설정이므로 페어 차이를 그냥 모아 t 를 낸다.
6시드씩 세 번 보류였는데, 18시드면 SE 가 절반 이하로 준다.

실행: python tools/refit_all.py
"""

import glob

import numpy as np

PAIRS = [("RN1.5", "RN2.0"), ("RQ1.5", "RQ2.0"), ("RS1.5", "RS2.0")]


def load(tag):
    out = {}
    y = None
    for f in sorted(glob.glob(f"./out/*_{tag}_s*_test_preds.npz")):
        z = np.load(f, allow_pickle=True)
        out[f.split("_s")[-1].split("_")[0]] = z["pred"].astype(np.float64)
        y = z["y"].astype(np.float64)
    return out, y


def main():
    d_all, rows = [], []
    for lo, hi in PAIRS:
        A, y = load(lo)
        B, _ = load(hi)
        if not A or not B:
            print(f"{lo}/{hi}: 없음 (A {len(A)} / B {len(B)})")
            continue
        r = float(y.mean())
        base = r * (1 - r)

        def sc(p):
            p = p - (p.mean() - r)
            return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)
        common = sorted(set(A) & set(B))
        d = np.array([sc(B[s]) - sc(A[s]) for s in common])
        d_all.append(d)
        rows.append((f"{lo}→{hi}", len(d), d.mean(),
                     d.std(ddof=1) / np.sqrt(len(d))))
    print(f"{'배치':<16}{'시드':>5}{'페어평균':>10}{'SE':>8}{'t':>7}")
    for nm, n, m, se in rows:
        print(f"{nm:<16}{n:>5}{m:>+10.2f}{se:>8.2f}{m / se:>+7.2f}")
    if not d_all:
        return 1
    d = np.concatenate(d_all)
    se = d.std(ddof=1) / np.sqrt(len(d))
    hi = d.mean() + 1.96 * se
    print(f"\n{'합산':<16}{len(d):>5}{d.mean():>+10.2f}{se:>8.2f}"
          f"{d.mean() / se:>+7.2f}")
    print(f"95%CI [{d.mean() - 1.96 * se:+.2f}, {hi:+.2f}]  →  "
          + ("채택" if d.mean() / se >= 2.4 else
             ("기각" if hi < 3 else "보류 — 시드 더 필요")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
