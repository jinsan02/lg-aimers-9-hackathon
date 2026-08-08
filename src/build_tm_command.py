"""trackman 기반 **구종별 커맨드 + 스태미나** 피처 (E103).

두 가지를 만든다. 둘 다 train/test 에는 원천적으로 없는 물리 정보다.

① 구종별 릴리스 산포 (구종별 커맨드)
   볼카운트가 구종을 예측한다(0-2는 변화구 유인, 3-0은 직구 스트라이크).
   따라서 "이 투수는 변화구 커맨드가 유독 나쁘다"는 정보는 볼카운트와 결합해
   **물리적 근거가 있는 투수x카운트 상호작용**이 된다.
   TE `_dev`(+27.2)가 같은 축을 '과거 성적 통계'로 잡았다면 이건 '메커니즘'이다.
   실측: 투수별 fastball 릴리스 산포가 0.022~0.323 으로 15배 차이난다.

② 자기 투구수에 따른 구속 감소 기울기 (스태미나)
   trackman 의 pitch_no 는 경기 전체 순번(양 팀 합산, 1~468)이라 그대로는 못 쓴다.
   (trackman_game_id, pitcher) 로 묶어 순위를 매기면 **그 투수의 자기 투구수**가 나오고,
   거기에 rel_speed 를 회귀한 기울기가 곧 스태미나다.
   `--feat-fatigue`(이닝/경험 대리지표)는 실패했지만 이건 실측 물리량이다.
   train/test 의 inning 과 결합하면 "이 투수는 7회부터 무너진다"가 된다.

2025 평가 행에는 **투수별 정적 속성**으로 붙는다 (trackman 은 2019~2024 뿐).
누수 차단: (투수, 시즌) 단위로 만들되 **그 시즌 이전까지만** 누적한다.

실행: python src/build_tm_command.py
산출: data/processed/tm_command.csv
"""

import os
import sys

import numpy as np
import pandas as pd

DATA = "./data"
OUT = f"{DATA}/processed/tm_command.csv"


def main():
    use = ["pitcher_trackman_id", "season", "trackman_game_id", "pitch_no",
           "pitch_type_group", "rel_speed", "rel_height", "rel_side",
           "spin_rate", "horz_break"]
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=use)
    print(f"trackman {len(tm):,}행")

    # ---- ① 구종별 릴리스 산포
    g = tm.groupby(["pitcher_trackman_id", "season", "pitch_type_group"])
    sc = g.agg(n=("rel_height", "size"), h=("rel_height", "std"),
               s=("rel_side", "std"), sp=("rel_speed", "std"),
               spin=("spin_rate", "std"), hb=("horz_break", "std")).reset_index()
    sc = sc[sc["n"] >= 60]
    sc["scatter"] = np.sqrt(sc["h"] ** 2 + sc["s"] ** 2)
    piv = sc.pivot_table(index=["pitcher_trackman_id", "season"],
                         columns="pitch_type_group",
                         values=["scatter", "sp", "hb"])
    piv.columns = [f"tm_{a}_{b[:2]}" for a, b in piv.columns]
    piv = piv.reset_index()
    keep = [c for c in piv.columns if c.endswith(("_fa", "_br", "_of"))]
    piv = piv[["pitcher_trackman_id", "season"] + keep]
    # 구종별 커맨드 격차 = 직구는 되는데 변화구가 안 되는 정도
    if "tm_scatter_br" in piv.columns and "tm_scatter_fa" in piv.columns:
        piv["tm_scatter_gap_br"] = piv["tm_scatter_br"] - piv["tm_scatter_fa"]
    if "tm_scatter_of" in piv.columns and "tm_scatter_fa" in piv.columns:
        piv["tm_scatter_gap_of"] = piv["tm_scatter_of"] - piv["tm_scatter_fa"]
    print(f"① 구종별 산포: {len(piv):,}건, 피처 {len(piv.columns) - 2}개")

    # ---- ② 자기 투구수 대비 구속 기울기
    tm = tm.sort_values(["trackman_game_id", "pitcher_trackman_id", "pitch_no"])
    tm["own_no"] = tm.groupby(["trackman_game_id", "pitcher_trackman_id"]) \
        .cumcount() + 1
    sub = tm[tm.rel_speed.notna()]
    rows = []
    for (pid, se), d2 in sub.groupby(["pitcher_trackman_id", "season"]):
        if len(d2) < 200:
            continue
        x = d2["own_no"].to_numpy(np.float64)
        v = d2["rel_speed"].to_numpy(np.float64)
        xm, vm = x.mean(), v.mean()
        den = ((x - xm) ** 2).sum()
        slope = ((x - xm) * (v - vm)).sum() / den if den > 0 else np.nan
        rows.append({"pitcher_trackman_id": pid, "season": se,
                     "tm_velo_slope": slope,          # 음수 = 던질수록 구속 하락
                     "tm_own_max": x.max()})          # 그 시즌 최대 연투수
    st = pd.DataFrame(rows)
    print(f"② 구속 기울기: {len(st):,}건 | 중앙값 {st.tm_velo_slope.median():+.5f}"
          f" (음수면 피로로 구속 하락)")

    f = piv.merge(st, on=["pitcher_trackman_id", "season"], how="outer")
    f = f.rename(columns={"pitcher_trackman_id": "tm_id"})

    # ---- 링키지 + 직전 시즌까지 누적 (누수 차단)
    pmap = pd.read_csv(f"{DATA}/processed/pitcher_map.csv")
    rep = (pmap.groupby(["pitcher_id", "tm_id"])["score"].agg(["count", "sum"])
           .reset_index()
           .sort_values(["pitcher_id", "count", "sum"], ascending=[True, False, False])
           .drop_duplicates("pitcher_id")[["pitcher_id", "tm_id"]])
    f = f.merge(rep, on="tm_id", how="inner").sort_values(["pitcher_id", "season"])
    cols = [c for c in f.columns if c.startswith("tm_") and c != "tm_id"]
    out = []
    for pid, sub2 in f.groupby("pitcher_id"):
        sub2 = sub2.sort_values("season")
        cum = sub2[cols].expanding().mean().shift(1)
        cum["pitcher_id"] = pid
        cum["season"] = sub2["season"].to_numpy()
        out.append(cum)
    res = pd.concat(out)
    last = res.sort_values("season").groupby("pitcher_id").tail(1).copy()
    last["season"] = last["season"] + 1        # 2025 행
    res = pd.concat([res, last]).dropna(subset=cols, how="all")
    res = res[["pitcher_id", "season"] + cols]
    os.makedirs(f"{DATA}/processed", exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"\n저장 {OUT}: {len(res):,}행, 피처 {len(cols)}개")
    print(f"  투수 {res.pitcher_id.nunique()}명 | 시즌 {res.season.min():.0f}~{res.season.max():.0f}")
    print(res[cols].describe().loc[["mean", "std", "min", "max"]].round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
