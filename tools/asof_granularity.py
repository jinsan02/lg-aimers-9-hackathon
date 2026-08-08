"""asof 갱신 단위 — 투구마다인가, 경기마다인가 (문서에 없는 성질 추적).

E99 는 "asof 가 시즌마다 리셋되는가?"를 확인해서 나왔다(+72.6).
E118 은 "prev1/3/5 가 중첩 창인가?"를 확인해서 나왔다.
같은 질문을 갱신 **단위**에 던진다.

  투구마다 갱신 -> asof_pitcher_n 이 행마다 +1 (지금 우리가 가정하는 것)
  경기마다 갱신 -> 한 경기 안에서 **상수**, 경기 경계에서 도약
                  => **도약 폭 = 그 경기의 투구 수**  <- 완전히 새로운 정보
                  => 그리고 같은 값을 가진 행끼리가 같은 경기 (경기 식별!)

경기 단위라면 2025 행에서도 '이 투수가 직전 경기에 몇 구 던졌나'를 알 수 있고,
주최측 발표자료 p14 CASE 03("투구 수가 늘어나도 제구가 유지되는가")에 직접 답한다.

같이 확인하는 것:
  - asof_batter_n 도 통산 누적인가 (E99 는 투수만 확인했다)
  - li / win_expectancy 가 상황에서 결정론적으로 계산되는가 (아니면 잔차가 정보)
  - row_id 가 시간 순인가

실행: python tools/asof_granularity.py
"""

import sys

import numpy as np
import pandas as pd

DATA = "./data"


def main():
    use = ["row_id", "season", "game_month", "game_dayofweek", "pitcher_id",
           "batter_id", "asof_pitcher_n", "asof_batter_n", "inning",
           "top_bottom", "balls_before", "strikes_before", "outs_before",
           "base_state", "li", "home_win_expectancy", "score_diff_home",
           "pitcher_team_id", "batter_team_id", "control_success"]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig", usecols=use)
    d = df[df.season == 2024].sort_values("row_id").reset_index(drop=True)
    print(f"2024 {len(d):,}행\n")

    print("=== ① asof_pitcher_n 은 행마다 +1 인가, 경기마다 도약하는가 ===")
    g = d.groupby("pitcher_id", sort=False)
    dn = g["asof_pitcher_n"].diff()
    v = dn.dropna().to_numpy()
    print(f"  차분 분포: =1 {float((v == 1).mean()):.1%} | =0 {float((v == 0).mean()):.1%} "
          f"| >1 {float((v > 1).mean()):.1%} | <0 {float((v < 0).mean()):.1%}")
    print(f"  >1 인 값들의 중앙 {np.median(v[v > 1]) if (v > 1).any() else 0:.0f} "
          f"| 90분위 {np.percentile(v[v > 1], 90) if (v > 1).any() else 0:.0f}")
    if float((v == 1).mean()) > 0.95:
        print("  -> **투구마다 갱신**. 도약 정보 없음 (우리 가정이 맞다)")
    elif float((v == 0).mean()) > 0.5:
        print("  -> **경기(또는 등판)마다 갱신**! 도약 폭 = 그 단위의 투구 수 = 신규 정보")
    else:
        print("  -> 혼재. 아래 세부 확인 필요")

    # 같은 asof 값을 갖는 연속 행이 정말 같은 경기인가 (이닝이 단조인지로 검증)
    d["_blk"] = (d.groupby("pitcher_id")["asof_pitcher_n"]
                 .transform(lambda s: (s.diff() != 0).cumsum()))
    sz = d.groupby(["pitcher_id", "_blk"]).size()
    print(f"  동일 asof 값 연속 블록 크기: 중앙 {sz.median():.0f} | "
          f"평균 {sz.mean():.1f} | 최대 {sz.max()}")

    print("\n=== ② asof_batter_n 도 통산 누적인가 (E99 는 투수만 확인했다) ===")
    last = (df.sort_values("asof_batter_n")
            .groupby(["batter_id", "season"], sort=False)
            .agg(mx=("asof_batter_n", "max"), mn=("asof_batter_n", "min"))
            .reset_index().sort_values(["batter_id", "season"]))
    prev_mx = last.groupby("batter_id")["mx"].shift(1)
    ok = prev_mx.notna()
    cont = (np.abs(last.loc[ok, "mn"] - prev_mx[ok]) <= 2).mean()
    reset = (last.loc[ok, "mn"] <= 2).mean()
    print(f"  직전 시즌 말과 이어짐 {cont:.3f} | 시즌 시작이 0 근처(리셋) {reset:.3f}")
    print("  -> 이어짐이 1.0 이면 타자도 통산 누적 (E99 차분이 타자에도 유효)")

    print("\n=== ③ li / 기대승률이 상황에서 결정론적으로 계산되는가 ===")
    key = ["inning", "top_bottom", "outs_before", "base_state",
           "score_diff_home"]
    for col in ["li", "home_win_expectancy"]:
        gg = d.groupby(key)[col]
        nuq = gg.nunique()
        rng = (gg.max() - gg.min())
        print(f"  {col:<22} 상황당 고유값 중앙 {nuq.median():.0f} | "
              f"폭 중앙 {rng.median():.4f} | 폭 90분위 {rng.quantile(0.9):.4f}")
    print("  -> 고유값이 1이면 상황만으로 완전 결정 = 중복. 아니면 잔차가 추가 정보")

    print("\n=== ④ row_id 가 시간 순인가 ===")
    rid = pd.to_numeric(d["row_id"], errors="coerce")
    if rid.isna().all():
        rid = pd.Series(np.arange(len(d)), index=d.index)   # 파일 순서로 대체
        print("  row_id 가 비수치 - 파일 순서로 대체해 확인")
    print(f"  row_id 예시: {list(d['row_id'].head(3))}")
    for c in ["game_month", "asof_pitcher_n"]:
        print(f"  row_id(순서) vs {c:<18} 상관 "
              f"{np.corrcoef(rid, d[c].fillna(0))[0, 1]:+.4f}")
    # 같은 (월,요일,팀) 이면 같은 경기일 가능성 - row_id 가 연속인지
    d["_g"] = (d.game_month.astype(str) + "_" + d.game_dayofweek.astype(str)
               + "_" + d.pitcher_team_id.astype(str) + "_"
               + d.batter_team_id.astype(str))
    d["_rid"] = rid
    sp = d.groupby("_g")["_rid"].agg(["min", "max", "size"])
    dens = (sp["size"] / (sp["max"] - sp["min"] + 1)).median()
    print(f"  (월,요일,두팀) 그룹의 row_id 밀도 중앙 {dens:.3f}")
    print("  -> 1에 가까우면 row_id 가 경기별로 연속 배치돼 있다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
