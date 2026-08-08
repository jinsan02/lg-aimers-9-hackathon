"""LOO 절제 결과를 편향제거 페어드 t 로 판정한다.

지금까지의 판정은 "현재 설정 위에 하나 얹는 한계 효과"였다. 이건 반대로
**최종 스택에서 하나씩 빼는** 절제라, 채택된 레버끼리의 중첩을 본다.

두 가지를 고쳐 넣었다.
  ① **수준 편향 제거 후** 비교 (LEVERS 측정 원칙. greedy 블렌드 도구는 이걸 어긴다)
  ② **행 페어드 부트스트랩 SE 를 시드 SE 와 합산.** 지금까지 시드 잡음만 세서
     t 가 일률적으로 부풀려져 있었다. n=3 에서 t=2.0 은 양측 p≈0.18 이다.

실행: python tools/loo_report.py A24        (또는 A23)
"""

import glob
import re
import sys

import numpy as np

BOOT = 200


def collect(prefix):
    out = {}
    for f in sorted(glob.glob(f"./out/cat_{prefix}_*_val_preds.npz")):
        m = re.search(rf"cat_({prefix}_.+?)_s(\d+)_val_preds\.npz",
                      f.replace("\\", "/"))
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = f
    return out


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "A24"
    arms = collect(prefix)
    base_name = f"{prefix}_full"
    if base_name not in arms:
        print(f"기준 팔 {base_name} 없음. 있는 팔: {list(arms)}")
        return 1

    z = np.load(next(iter(arms[base_name].values())))
    y = z["y"].astype(np.float64)
    r = y.mean()
    var = r * (1 - r)

    def score(p, idx=None):
        p = p - (p.mean() - r)                 # 수준 편향 제거
        d = (p - y) ** 2
        return 1e5 * (1 - (d if idx is None else d[idx]).mean() / var)

    preds = {a: {s: np.load(f)["pred"].astype(np.float64)
                 for s, f in d.items()} for a, d in arms.items()}
    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(y), len(y)) for _ in range(BOOT)]

    print(f"{prefix} | {len(y):,}행 | 성공률 {r:.4f} | 편향제거 기준\n")
    b = preds[base_name]
    print(f"{'뺀 레버':<16}{'시드':>4}{'평균':>9}{'차이':>9}"
          f"{'시드SE':>8}{'행SE':>7}{'합산t':>8}   판정")
    base_mean = np.mean([score(p) for p in b.values()])
    print(f"{'(기준=전체)':<16}{len(b):>4}{base_mean:>9.2f}\n")

    rows = []
    for a in sorted(arms):
        if a == base_name:
            continue
        sh = sorted(set(preds[a]) & set(b))
        if len(sh) < 2:
            continue
        # 뺐을 때의 변화. 부호를 뒤집어 **레버의 기여**로 보고한다.
        d = np.array([score(preds[a][s]) - score(b[s]) for s in sh])
        contrib = -d.mean()
        se_seed = d.std(ddof=1) / np.sqrt(len(d))
        # 행 잡음: 시드평균 예측끼리 부트스트랩
        pa = np.mean([preds[a][s] for s in sh], 0)
        pb = np.mean([b[s] for s in sh], 0)
        se_row = np.std([score(pb, i) - score(pa, i) for i in boot], ddof=1)
        se = float(np.hypot(se_seed, se_row))
        t = contrib / se if se > 0 else 0.0
        lo, hi = contrib - 2 * se, contrib + 2 * se
        v = ("채택유지" if t >= 2.4 else
             "제거후보" if hi < 3 else "판정불가")
        rows.append((a, len(sh), np.mean([score(preds[a][s]) for s in sh]),
                     contrib, se_seed, se_row, t, v, lo, hi))

    for a, n, m, c, ss, sr, t, v, lo, hi in sorted(rows, key=lambda x: -x[3]):
        print(f"{a.replace(prefix + '_', ''):<16}{n:>4}{m:>9.2f}{c:>+9.2f}"
              f"{ss:>8.2f}{sr:>7.2f}{t:>+8.2f}   {v}  [{lo:+.1f}, {hi:+.1f}]")
    print("\n※ '차이' = 그 레버의 기여(빼면 얼마나 나빠지는가). 양수면 레버가 일하고 있다.")
    print("※ 판정선: 채택유지 t≥2.4 / 제거후보 = 95%CI 상한 < +3 / 그 외 시드추가")
    return 0


if __name__ == "__main__":
    sys.exit(main())
