"""out/*_val_preds.npz 에서 태그별 시드 BSS 를 다시 계산한다 (로그 유실 대비)."""
import glob, re, sys
import numpy as np

pat = sys.argv[1] if len(sys.argv) > 1 else "*"
rows = {}
for f in sorted(glob.glob(f"out/cat_{pat}_val_preds.npz")):
    z = np.load(f)
    y = z["y"].astype(np.float64); p = z["pred"].astype(np.float64)
    r = y.mean(); base = r * (1 - r)
    bss = 1e5 * (1 - ((p - y) ** 2).mean() / base)
    m = re.match(r"out/cat_(.+)_s(\d+)_val_preds\.npz", f.replace("\\", "/"))
    tag, sd = (m.group(1), m.group(2)) if m else (f, "?")
    rows.setdefault(tag, {})[sd] = bss

tags = list(rows)
for t in tags:
    v = rows[t]
    print(f"{t:<16} n={len(v):>2}  mean {np.mean(list(v.values())):8.2f}  "
          + " ".join(f"s{k}={x:.1f}" for k, x in sorted(v.items())))
if len(tags) == 2:
    a, b = tags
    sh = sorted(set(rows[a]) & set(rows[b]))
    d = np.array([rows[b][s] - rows[a][s] for s in sh])
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    print(f"\n짝지은 차이 {b} - {a}: {d.mean():+.2f}  t={t:+.2f}  (n={len(d)})")
