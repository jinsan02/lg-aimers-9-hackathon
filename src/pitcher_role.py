"""투수 역할·교체 타이밍 — train 이력에서 유도, test에 lookup 적용.

착안: train.csv에는 '선발/중간/마무리' 라벨이 없지만, **투수별 등판 이닝 분포**로
      역할을 복원할 수 있다. 그러면 같은 6회라도 의미가 달라진다.
        · 선발(1회 시작)에게 6회 = 투구수 90개쯤, 체력 소진 구간
        · 마무리(9회 등판)에게 6회 = 이례적 조기 투입
      → `inning - 그 투수의 통상 등판 이닝` = **개인 기준 상대 피로도**

산출: data/processed/pitcher_role.csv
  (pitcher_id → 등판 이닝 중앙값/최빈, 1회 비중=선발성, 9회+ 비중=마무리성)
평가(2025)는 train(2019~24) lookup — 상수 조회라 평가 행 간 정보 사용 없음.
"""

import numpy as np
import pandas as pd

DATA, T = "./data", "control_success"


def build_role_table(df):
    g = df.groupby("pitcher_id")["inning"]
    tab = pd.DataFrame({
        "role_inn_med": g.median(),
        "role_inn_min": g.quantile(0.05),
        "role_start_share": df.assign(_s=(df.inning == 1).astype(float))
                              .groupby("pitcher_id")["_s"].mean(),
        "role_close_share": df.assign(_c=(df.inning >= 9).astype(float))
                              .groupby("pitcher_id")["_c"].mean(),
        "role_n": g.size(),
    })
    return tab.reset_index()


def add_role(df, tab):
    out = df.merge(tab, on="pitcher_id", how="left")
    # 개인 기준 상대 피로도: 통상 등판 이닝보다 얼마나 깊이 왔는가
    out["role_depth"] = out["inning"] - out["role_inn_min"]
    cols = ["role_inn_med", "role_start_share", "role_close_share", "role_depth"]
    return out, cols


def main():
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=["pitcher_id", "pitcher_team_id", "season",
                              "inning", "top_bottom", "asof_pitcher_n", T])
    tab = build_role_table(df[df.season < 2024])
    out, cols = add_role(df, tab)

    print("=== 1. 역할 분류 (1회 등판 비중 기준) ===")
    out["role"] = pd.cut(out.role_start_share, [-0.01, 0.02, 0.15, 1.0],
                         labels=["불펜", "스윙맨", "선발"])
    print(out.groupby("role", observed=True).agg(
        n=(T, "size"), rate=(T, "mean"),
        inn_med=("role_inn_med", "median"),
        close=("role_close_share", "mean"),
        exp=("asof_pitcher_n", "median")).round(4).to_string())

    print("\n=== 2. 게임 구간별 교체 = 어느 이닝에 누가 던지나 ===")
    out["phase"] = pd.cut(out.inning, [0, 3, 6, 8, 13],
                          labels=["초반1-3", "중반4-6", "후반7-8", "마무리9+"])
    print(out.pivot_table(index="role", columns="phase", values=T,
                          aggfunc=["size", "mean"], observed=True).round(4).to_string())

    print("\n=== 3. 개인 상대 피로도(role_depth) → 제구 ===")
    out["depth_bin"] = pd.cut(out.role_depth, [-9, -1, 0, 2, 4, 12],
                              labels=["이른등판", "통상", "+1~2", "+3~4", "+5이상"])
    print(out.groupby("depth_bin", observed=True)[T]
          .agg(["size", "mean"]).round(4).to_string())

    print("\n=== 4. 9회+ 등판(마무리) 전용 ===")
    nine = out[out.inning >= 9]
    print(f"9회+ 행수 {len(nine):,} | 성공률 {nine[T].mean():.4f} "
          f"(전체 {out[T].mean():.4f})")
    print(nine.groupby("role", observed=True)[T].agg(["size", "mean"]).round(4).to_string())

    print("\n=== 5. 팀별 차이 (교체 운영·투수 육성) ===")
    tm = out.groupby("pitcher_team_id").agg(
        n=(T, "size"), rate=(T, "mean"),
        bullpen_share=("role_start_share", lambda s: (s < 0.02).mean()))
    print(tm.sort_values("rate").round(4).to_string())
    print(f"  팀 간 성공률 폭: {tm.rate.max() - tm.rate.min():.4f}")

    tab.to_csv(f"{DATA}/processed/pitcher_role.csv", index=False)
    print(f"\n저장: pitcher_role.csv {tab.shape}")


if __name__ == "__main__":
    main()
