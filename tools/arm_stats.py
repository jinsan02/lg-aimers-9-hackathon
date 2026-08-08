"""스윕 결과를 시드 페어드로 판정한다 (운영 규칙 자동화).

LEVERS.md 규칙: 같은 머신 3시드 이상 페어드, t >= 2.0 채택 / t < 1.0 기각 /
그 사이는 시드 추가. 그리고 **원 BSS 비교 전에 수준 편향을 본다** -
공통 편향이 설정 간 차이를 가린 사례가 있었다(레버 E/F/H).

사용:
  python tools/arm_stats.py Y_g100 Y_g090 Y_g080 Y_g070 Y_w3
    -> 첫 인자를 기준으로 나머지를 페어드 비교. 각 arm 의 편향도 같이 낸다.

출력의 '편향제거 후'는 각 arm 에서 자기 편향을 뺀 뒤의 BSS 다.
설정 비교는 이쪽을 봐야 '수준을 잘 맞춘 덕'과 '변별을 잘한 덕'이 안 섞인다.
"""

import glob
import re
import sys

import numpy as np


def load(arm):
    """arm 이름으로 시드별 검증예측을 모은다. {seed: (y, pred)}"""
    out = {}
    for f in glob.glob(f"./out/*_{arm}_s*_val_preds.npz"):
        m = re.search(rf"_{re.escape(arm)}_s(\d+)_val_preds\.npz$",
                      f.replace("\\", "/"))
        if not m:
            continue
        z = np.load(f)
        out[int(m.group(1))] = (z["y"], z["pred"])
    return out


def bss(y, p, base):
    return 100000 * (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2).mean() / base)


def main():
    arms = sys.argv[1:]
    if len(arms) < 2:
        print("사용: python tools/arm_stats.py <기준arm> <arm2> [arm3 ...]")
        return 1
    data = {a: load(a) for a in arms}
    miss = [a for a in arms if not data[a]]
    if miss:
        print(f"예측 파일 없음: {miss}")
        arms = [a for a in arms if data[a]]
        if len(arms) < 2:
            return 1
    ref = arms[0]
    seeds = sorted(set.intersection(*[set(data[a]) for a in arms]))
    print(f"공통 시드 {seeds} | 기준 {ref}\n")

    stats = {}
    for a in arms:
        raws, adjs, bias = [], [], []
        for s in seeds:
            y, p = data[a][s]
            r = y.mean()
            base = r * (1 - r)
            b = p.mean() - r
            raws.append(bss(y, p, base))
            adjs.append(bss(y, p - b, base))
            bias.append(b)
        stats[a] = (np.array(raws), np.array(adjs), np.array(bias))

    hdr = f"{'arm':<14}" + "".join(f"{f's{s}':>9}" for s in seeds) \
        + f"{'평균':>9}{'편향':>10}{'편향제거후':>11}"
    print(hdr)
    for a in arms:
        raws, adjs, bias = stats[a]
        print(f"{a:<14}" + "".join(f"{v:>9.1f}" for v in raws)
              + f"{raws.mean():>9.1f}{bias.mean():>+10.5f}{adjs.mean():>11.1f}")

    print(f"\n=== {ref} 대비 페어드 (n={len(seeds)}) ===")
    print(f"{'arm':<14}{'원BSS차':>10}{'t':>7}   "
          f"{'편향제거차':>11}{'t':>7}   판정")
    for a in arms[1:]:
        line = f"{a:<14}"
        verdicts = []
        for j in (0, 1):
            d = stats[a][j] - stats[ref][j]
            m = d.mean()
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            t = m / se if se and se > 0 else 0.0
            line += f"{m:>10.2f}{t:>7.2f}   " if j == 0 else \
                    f"{m:>11.2f}{t:>7.2f}   "
            verdicts.append(t)
        t = verdicts[1]        # 판정은 편향제거 기준
        v = "채택" if t >= 2.0 else ("기각" if t < 1.0 else "시드추가")
        print(line + v)
    print("\n※ 판정은 편향제거 기준 t. 원BSS 차이는 수준 운(SHIFT로 따로 해결)이 섞인다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
