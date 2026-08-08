"""E142 검정 — Optuna 최적 설정 vs 현행 v14f를 **탐색에 안 쓴 시드**로 비교.

tune.py 는 시드 42,7 만 보고 40 trial 중 최고를 골랐다. 그 최고값에는 승자의
저주가 섞여 있다(2시드 SE 3.2). 여기서는 3,4,5,6,8,13 으로만 비교하므로
선택 편의가 없다. 같은 시드를 짝지어 빼므로(paired) 시드 잡음도 상쇄된다.
"""

import numpy as np

SEEDS = ["3", "4", "5", "6", "8", "13"]

y = np.load("./out/cat_v14f_s3_val_preds.npz")["y"].astype(np.float64)
r = float(y.mean())
base = r * (1 - r)


def raw(p):
    return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)


def sc(p):
    return raw(p - (p.mean() - r))          # 수준 편향은 SHIFT 가 따로 잡는다


B = {s: np.load(f"./out/cat_v14f_s{s}_val_preds.npz")["pred"].astype(np.float64)
     for s in SEEDS}
O = {s: np.load(f"./out/cat_O1_s{s}_val_preds.npz")["pred"].astype(np.float64)
     for s in SEEDS}

print(f"{'시드':>5}{'v14f':>10}{'O1':>10}{'차이':>9}")
d = []
for s in SEEDS:
    b, o = sc(B[s]), sc(O[s])
    d.append(o - b)
    print(f"{s:>5}{b:>10.2f}{o:>10.2f}{o - b:>+9.2f}")
d = np.array(d)
se = d.std(ddof=1) / np.sqrt(len(d))
print(f"\n페어 평균 차이 {d.mean():+.2f}  SE {se:.2f}  t={d.mean() / se:+.2f}")

pb = np.mean([B[s] for s in SEEDS], 0)
po = np.mean([O[s] for s in SEEDS], 0)
print(f"\n6시드 앙상블  v14f {sc(pb):8.2f}  O1 {sc(po):8.2f}  "
      f"차이 {sc(po) - sc(pb):+.2f}")
print(f"  원점수      v14f {raw(pb):8.2f}  O1 {raw(po):8.2f}")
print(f"  예측평균    v14f {pb.mean():.4f}   O1 {po.mean():.4f}   실제 {r:.4f}")

# 둘을 블렌드하면? margin = Dmax - D 가 양수여야 기여가 있다
rms = float(np.sqrt(((pb - po) ** 2).mean()))
dmax = 400320 * rms ** 2
dd = sc(pb) - sc(po)
print(f"\n블렌드 여지: rms {rms:.4f} | Dmax {dmax:.1f} | D {dd:+.1f} | "
      f"margin {dmax - dd:+.1f} | 이득 {max(0.0, dmax - dd) ** 2 / (4 * dmax):.2f}")
