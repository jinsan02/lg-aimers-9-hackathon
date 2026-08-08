"""제출 경로를 **평가 서버 조건 그대로** 재서 예산을 확정한다.

평가 서버: 6 vCPU / 28GB RAM, 추론 10분(600초) 한도, 245,789행.
지금까지 "38초" 같은 숫자는 모델 예측만 잰 것이라 **피처 생성 비용이 빠져 있었다.**
여기서는 script.py 가 실제로 하는 일을 전부 잰다: CSV 읽기 -> fpipe.transform ->
멤버별 예측 -> 블렌드.

용량(10GB)은 사실상 제약이 아니다. 진짜 제약은 이 600초다.

실행: OMP_NUM_THREADS=6 taskset -c 0-5 python tools/bench_submit.py
"""

import glob
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
N = 245_789
LIMIT = 600.0


def main():
    import fpipe

    pats = sys.argv[1:] or ["model/cat_v14f_s*.pkl", "model/cat_v14c_s*.pkl"]
    paths = sorted(p for pat in pats for p in glob.glob(pat))
    if not paths:
        print(f"모델 없음: {pats}")
        return 1
    mb = sum(os.path.getsize(p) for p in paths) / 1e6
    print(f"멤버 {len(paths)}개 | {mb:.0f}MB | 목표 {N:,}행 / {LIMIT:.0f}초\n")

    t0 = time.perf_counter()
    cols = pd.read_csv("./data/test.csv", encoding="utf-8-sig", nrows=0).columns
    src = pd.read_csv("./data/train.csv", encoding="utf-8-sig",
                      usecols=[c for c in cols])
    src = src[src.season == 2024].reset_index(drop=True)
    if len(src) < N:                      # 부족하면 반복해 채운다
        src = pd.concat([src] * (N // len(src) + 1), ignore_index=True)
    test = src.iloc[:N].reset_index(drop=True)
    t_read = time.perf_counter() - t0
    print(f"  CSV 읽기·구성   {t_read:6.1f}s")

    total = t_read
    packs = []
    t0 = time.perf_counter()
    for p in paths:
        packs.append(joblib.load(p))
    t_load = time.perf_counter() - t0
    total += t_load
    print(f"  모델 로드       {t_load:6.1f}s")

    # 피처 생성은 멤버마다 아티팩트가 달라 **멤버 수만큼** 반복된다.
    # (지금 script.py 가 그렇게 돼 있다 — 여기서 그 비용이 처음 드러난다)
    preds = np.zeros(len(test))
    for i, (path, pack) in enumerate(zip(paths, packs)):
        t0 = time.perf_counter()
        p = fpipe.predict(pack, test)
        el = time.perf_counter() - t0
        total += el
        preds += p / len(paths)
        print(f"  [{i + 1}/{len(paths)}] {os.path.basename(path):<22} "
              f"{el:6.1f}s  (누적 {total:6.1f}s)")
        if total > LIMIT:
            print(f"  !! {LIMIT:.0f}초 초과 — 여기서 중단")
            break

    print(f"\n합계 {total:.1f}s / {LIMIT:.0f}s  (여유 {LIMIT - total:.0f}s, "
          f"{100 * total / LIMIT:.0f}% 사용)")
    if total < LIMIT:
        room = (LIMIT - total) / max(total / len(paths), 1e-9)
        print(f"같은 비용의 멤버를 {room:.0f}개 더 넣을 여유가 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
