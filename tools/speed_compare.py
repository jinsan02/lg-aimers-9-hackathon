"""노트북 네이티브 vs WSL 속도 비교 — 동일 작업 3종."""
import time

import numpy as np
import pandas as pd

t0 = time.perf_counter()
df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
t_read = time.perf_counter() - t0

t0 = time.perf_counter()
g = df.groupby(["pitcher_id", "season"])["control_success"].agg(["mean", "size"])
_ = df.pivot_table(index="balls_before", columns="strikes_before",
                   values="control_success", aggfunc="mean")
t_agg = time.perf_counter() - t0

t0 = time.perf_counter()
num = df.select_dtypes(include=[np.number])
_ = num.corr()
t_corr = time.perf_counter() - t0

print(f"csv 로드   {t_read:6.2f}s")
print(f"groupby    {t_agg:6.2f}s")
print(f"상관행렬   {t_corr:6.2f}s")
print(f"합계       {t_read + t_agg + t_corr:6.2f}s  (행 {len(df):,} × 열 {df.shape[1]})")
