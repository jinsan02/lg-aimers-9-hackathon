"""trackman 상황별 집계 — **선수 링키지 불필요** 경로.

놓쳤던 사실: trackman과 train은 상황 키(season/월/요일/이닝/초말/카운트/좌우)를
**공유**한다. 선수 ID를 못 맞춰도 '이 상황에서 리그 전체가 어떻게 던지는가'는 집계 가능.

train.csv에 없는 정보 3종을 상황 키로 붙인다:
  1) pitch_of_pa   — 이 상황이 타석 몇 번째 공인지 (파울 포함 실제 깊이)
  2) pitch_no      — 경기 내 몇 번째 투구인지 (투수 피로도 대리)
  3) 구종 분포·구속 — 이 상황에서 무엇을 던지는가 (= 투수 의도)

⚠️ 규칙 준수: trackman은 2019~2024 공식 보조 데이터. 평가 데이터(test) 행은 전혀 사용하지
않고, 집계 결과는 상수 lookup으로 저장되어 각 행에 독립 적용된다.

실행: ~/.venvs/aimers/bin/python src/tm_context.py
산출: data/processed/tm_context.csv
"""

import numpy as np
import pandas as pd

DATA = "./data"
# train과 trackman이 공유하는 상황 키 (선수 ID 불필요)
KEY = ["balls_before", "strikes_before", "outs_before", "inning", "top_bottom",
       "pitcher_hand", "batter_hand"]
MIN_N = 200      # 이 미만 셀은 상위 키로 폴백


def main():
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=KEY + ["pitch_of_pa", "pitch_no", "pitch_type_group",
                                    "rel_speed", "spin_rate", "extension"])
    tm["top_bottom"] = tm["top_bottom"].map({"Top": "T", "Bottom": "B"})
    for c in ["pitcher_hand", "batter_hand"]:
        tm[c] = tm[c].map({"Left": 1, "Right": 2})
    tm = tm.dropna(subset=["top_bottom", "pitcher_hand", "batter_hand"])
    tm["inning"] = tm["inning"].clip(upper=12)
    print(f"trackman {len(tm)}행")

    # 구종 원핫 → 상황별 비율
    for g in ["fastball", "breaking", "offspeed"]:
        tm[f"pt_{g}"] = (tm.pitch_type_group == g).astype(np.float32)

    agg = {"pitch_of_pa": "mean", "pitch_no": "mean", "rel_speed": "mean",
           "spin_rate": "mean", "extension": "mean",
           "pt_fastball": "mean", "pt_breaking": "mean", "pt_offspeed": "mean"}
    full = tm.groupby(KEY).agg(agg)
    full["n"] = tm.groupby(KEY).size()

    # 표본 부족 셀 → 카운트만으로 만든 상위 집계로 폴백
    coarse_key = ["balls_before", "strikes_before", "outs_before"]
    coarse = tm.groupby(coarse_key).agg(agg)
    global_mean = tm[list(agg)].mean()

    full = full.reset_index()
    weak = full["n"] < MIN_N
    print(f"셀 {len(full)}개 | 표본<{MIN_N} 폴백 대상 {weak.sum()}개")
    cidx = pd.MultiIndex.from_frame(full.loc[weak, coarse_key])
    for c in agg:
        vals = coarse[c].reindex(cidx).to_numpy()
        full.loc[weak, c] = np.where(np.isnan(vals), global_mean[c], vals)

    full = full.rename(columns={c: f"tmc_{c}" for c in agg})
    full = full.rename(columns={"n": "tmc_n"})
    full.to_csv(f"{DATA}/processed/tm_context.csv", index=False)
    print(f"저장: tm_context.csv {full.shape}")
    print("\n샘플 — 볼카운트별 속구 비율/투구깊이 (0아웃·5회·우투우타):")
    s = full[(full.outs_before == 0) & (full.inning == 5)
             & (full.pitcher_hand == 2) & (full.batter_hand == 2)]
    print(s[["balls_before", "strikes_before", "tmc_pt_fastball",
             "tmc_pitch_of_pa", "tmc_pitch_no", "tmc_rel_speed"]]
          .sort_values(["balls_before", "strikes_before"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
