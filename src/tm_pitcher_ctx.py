"""투수 × 상황 결합 trackman — train.csv에 **정말로 없는** 유일한 정보원.

"이 투수가 3-0 카운트에서 던지는 속구 비율이 리그 평균과 얼마나 다른가"
= 개인의 상황별 성향 편차. asof_*는 전체 평균만 담고 있어 이건 못 담는다.

⚠️ 링키지 정확도 86.4%(E19)가 상한 — 13.6%는 잘못된 투수 통계가 섞인다.
   그래서 편차를 shrinkage로 눌러 노이즈 유입을 제한한다.

실행: ~/.venvs/aimers/bin/python src/tm_pitcher_ctx.py
산출: data/processed/tm_pitcher_ctx.csv  (pitcher_id, 카운트상태 → 편차 피처)
"""

import numpy as np
import pandas as pd

DATA = "./data"
K_SHRINK = 50   # 셀 표본이 이보다 작으면 리그 평균 쪽으로 수축


def main():
    pmap = pd.read_csv(f"{DATA}/processed/pitcher_map.csv")
    rep = (pmap.groupby(["pitcher_id", "tm_id"])["score"].agg(["count", "sum"])
           .reset_index()
           .sort_values(["count", "sum"], ascending=False)
           .drop_duplicates("pitcher_id")[["pitcher_id", "tm_id"]])

    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["pitcher_trackman_id", "balls_before",
                              "strikes_before", "pitch_type_group", "rel_speed"])
    tm = tm.rename(columns={"pitcher_trackman_id": "tm_id"})
    tm["cnt"] = tm.balls_before.astype(str) + "-" + tm.strikes_before.astype(str)
    tm["fb"] = (tm.pitch_type_group == "fastball").astype(np.float32)

    league = tm.groupby("cnt").agg(fb_lg=("fb", "mean"),
                                   sp_lg=("rel_speed", "mean"))
    cell = tm.groupby(["tm_id", "cnt"]).agg(fb=("fb", "mean"),
                                            sp=("rel_speed", "mean"),
                                            n=("fb", "size")).reset_index()
    cell = cell.merge(league, on="cnt")

    # 편차를 표본수로 수축: n이 작으면 0(=리그 평균과 같음)으로
    w = cell.n / (cell.n + K_SHRINK)
    cell["d_fb"] = w * (cell.fb - cell.fb_lg)
    cell["d_sp"] = w * (cell.sp - cell.sp_lg)

    out = cell.merge(rep, on="tm_id")[["pitcher_id", "cnt", "d_fb", "d_sp", "n"]]
    out = out.rename(columns={"n": "tmp_n"})
    out.to_csv(f"{DATA}/processed/tm_pitcher_ctx.csv", index=False)
    print(f"저장: tm_pitcher_ctx.csv {out.shape} "
          f"(투수 {out.pitcher_id.nunique()}명 × 카운트 {out.cnt.nunique()})")
    print(f"편차 분포 d_fb: 표준편차 {out.d_fb.std():.4f} "
          f"| d_sp: {out.d_sp.std():.3f} km/h")
    print("\n샘플 (3-0 카운트에서 리그 대비 속구 편차 상위):")
    s = out[out.cnt == "3-0"].nlargest(5, "d_fb")
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
