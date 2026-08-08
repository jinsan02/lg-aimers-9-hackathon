"""trackman 기반 **릴리스 일관성** 피처 (E101).

착안: 기존 tm 피처(data/processed/tm_pitcher_feats.csv)는 전부 **평균**이다
  - tm_rel_speed, tm_spin_rate, tm_rel_height, tm_rel_side, tm_extension ...
그런데 이 대회의 타깃은 "공이 의도한 곳으로 갔는가"이고, 그 물리적 실체는
**같은 동작을 반복할 수 있는가** = 릴리스 포인트/구속/무브먼트의 **분산**이다.
평균 구속이 빠른 것과 제구가 좋은 것은 별개다. 지금까지 분산은 한 번도 안 썼다.

핵심 설계 - **구종 내 분산**:
  4가지 구종을 던지는 투수는 구종 배합만으로도 구속 분산이 크다. 그건 제구력이
  아니라 레퍼토리다. 그래서 구종별로 분산을 재고 표본수 가중 평균한다.
  단, 릴리스 포인트(rel_height/rel_side)는 구종에 무관하게 같아야 하는 것이므로
  전체 분산도 같이 낸다 - 오히려 이쪽이 순수 커맨드 지표다.

누수 차단: (투수, 시즌) 단위로 만들되 **그 시즌 이전까지만** 누적해서 쓴다
  (train_gbdt2 의 --tm-feats 병합 방식이 season 키를 쓰므로 그대로 따른다).

실행: python src/build_tm_consistency.py
산출: data/processed/tm_consistency.csv  (pitcher_id, season, tmc_* )
"""

import os
import sys

import numpy as np
import pandas as pd

DATA = "./data"
OUT = f"{DATA}/processed/tm_consistency.csv"

# 분산을 잴 물리량
VARY = ["rel_height", "rel_side", "rel_speed", "spin_rate",
        "induced_vert_break", "horz_break", "extension"]
# 구종에 무관하게 일정해야 하는 것 = 전체 분산이 곧 커맨드 지표
POSE = ["rel_height", "rel_side", "extension"]


def main():
    use = ["pitcher_trackman_id", "season", "pitch_type_group"] + VARY
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=use)
    print(f"trackman {len(tm):,}행")

    # 1) 구종 내 분산 -> 표본수 가중 평균 (레퍼토리 효과 제거)
    g = tm.groupby(["pitcher_trackman_id", "season", "pitch_type_group"])
    within = g[VARY].std()
    cnt = g.size().rename("n")
    within = within.join(cnt).reset_index()
    within = within[within["n"] >= 20]          # 표본 부족 구종 제외
    rows = []
    for keys, sub in within.groupby(["pitcher_trackman_id", "season"]):
        w = sub["n"].to_numpy(np.float64)
        rec = {"tm_id": keys[0], "season": keys[1], "tmc_n": w.sum()}
        for c in VARY:
            v = sub[c].to_numpy(np.float64)
            m = ~np.isnan(v)
            rec[f"tmc_w_{c}"] = float((v[m] * w[m]).sum() / w[m].sum()) \
                if m.any() else np.nan
        rows.append(rec)
    wf = pd.DataFrame(rows)
    print(f"구종내 분산 (투수x시즌) {len(wf):,}건")

    # 2) 릴리스 자세는 구종 무관하게 일정해야 함 -> 전체 분산
    g2 = tm.groupby(["pitcher_trackman_id", "season"])
    allv = g2[POSE].std().reset_index()
    allv.columns = ["tm_id", "season"] + [f"tmc_all_{c}" for c in POSE]
    # 릴리스 포인트 2차원 산포 = sqrt(var_h + var_s) : 한 숫자로 요약한 커맨드
    allv["tmc_rel_scatter"] = np.sqrt(allv["tmc_all_rel_height"] ** 2
                                      + allv["tmc_all_rel_side"] ** 2)
    f = wf.merge(allv, on=["tm_id", "season"], how="outer")

    # 3) pitcher_id 로 링키지 (기존 pitcher_map 재사용)
    pmap = pd.read_csv(f"{DATA}/processed/pitcher_map.csv")
    rep = (pmap.groupby(["pitcher_id", "tm_id"])["score"].agg(["count", "sum"])
           .reset_index()
           .sort_values(["pitcher_id", "count", "sum"], ascending=[True, False, False])
           .drop_duplicates("pitcher_id")[["pitcher_id", "tm_id"]])
    f = f.merge(rep, on="tm_id", how="inner")
    print(f"링키지 후 {len(f):,}건 / 투수 {f.pitcher_id.nunique()}명")

    # 4) **그 시즌 이전까지**의 누적 평균으로 바꿔 누수를 막는다
    f = f.sort_values(["pitcher_id", "season"])
    cols = [c for c in f.columns if c.startswith("tmc_")]
    out = []
    for pid, sub in f.groupby("pitcher_id"):
        sub = sub.sort_values("season")
        cum = sub[cols].expanding().mean().shift(1)   # 직전 시즌까지
        cum["pitcher_id"] = pid
        cum["season"] = sub["season"].to_numpy()
        out.append(cum)
    res = pd.concat(out)
    # 다음 시즌(2025) 행도 만들어 제출 추론에서 쓰게 한다
    last = res.sort_values("season").groupby("pitcher_id").tail(1).copy()
    last["season"] = last["season"] + 1
    res = pd.concat([res, last])
    res = res.dropna(subset=["tmc_rel_scatter"])
    res = res[["pitcher_id", "season"] + cols]
    os.makedirs(f"{DATA}/processed", exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"저장 {OUT}: {len(res):,}행, 피처 {len(cols)}개")
    print(f"  시즌 범위 {res.season.min():.0f}~{res.season.max():.0f}")
    print(res[["tmc_rel_scatter", "tmc_w_rel_speed"]].describe().round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
