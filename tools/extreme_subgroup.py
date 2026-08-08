"""결정론적 하위집단이 존재하는가 — 우리 예측이 0.39~0.52 에 갇힌 이유를 캔다.

동기:
  1위(LB 1371)와 우리(1061)의 격차 310 은 "전체의 0.31%(약 770행)를 완벽히 아는 것"과
  산술이 정확히 같다.  1371.40 = f x 100000 + (1-f) x 1061.21  ->  f = 0.00314
  (물론 이건 여러 분해 중 하나일 뿐이다 — 전 구간 균일 개선도 같은 총합을 낸다.)

  그런데 우리 모델의 예측은 전부 0.39~0.52 안에 있다. 즉 **확신하는 행이 하나도 없다.**
  이게 "그런 하위집단이 없어서"인지 "우리가 못 찾아서"인지는 다른 문제다.

방법 (교차적합 + 시즌 이전 검사 — E123 에서 이걸 안 해서 -4.07 을 태웠다):
  ① 2024 를 반으로 갈라 A 에서 그룹 평균, B 에서 평가 -> 시즌 내 극단성
  ② 2021~23 에서 만든 그룹 평균을 2024 에 적용 -> **시즌 간 이전성**
  ②가 살아남는 그룹만 진짜다.

실행: python tools/extreme_subgroup.py
"""

import sys
from itertools import combinations

import numpy as np
import pandas as pd

DATA = "./data"
TARGET = "control_success"
MIN_N = 300          # 이보다 작은 그룹은 잡음
K = 50.0             # 수축


def main():
    use = ["season", "balls_before", "strikes_before", "outs_before", "inning",
           "base_state", "top_bottom", "game_type", "pitcher_hand",
           "batter_hand", "num_runners_on", "score_diff_pitcher_team", TARGET]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig", usecols=use)
    df["cnt"] = df.balls_before.astype(str) + "-" + df.strikes_before.astype(str)
    df["sd3"] = df.score_diff_pitcher_team.clip(-5, 5)
    df["inn"] = df.inning.clip(upper=10)
    axes = ["cnt", "base_state", "outs_before", "inn", "top_bottom",
            "game_type", "pitcher_hand", "batter_hand", "sd3"]

    past = df[df.season.between(2021, 2023)]
    cur = df[df.season == 2024].reset_index(drop=True)
    r_past, r_cur = past[TARGET].mean(), cur[TARGET].mean()
    print(f"2021~23 {len(past):,}행 (성공률 {r_past:.4f}) | "
          f"2024 {len(cur):,}행 ({r_cur:.4f})\n")

    rng = np.random.default_rng(0)
    half = rng.random(len(cur)) < 0.5

    rows = []
    # 1~3개 축 조합 전부
    for k in (1, 2, 3):
        for combo in combinations(axes, k):
            keys = list(combo)
            gp = past.groupby(keys, observed=True)[TARGET].agg(["sum", "size"])
            gp = gp[gp["size"] >= MIN_N]
            if gp.empty:
                continue
            m_past = (gp["sum"] + K * r_past) / (gp["size"] + K)
            # 시즌 간 이전: 2021~23 값을 2024 행에 붙여 실제와 비교
            idx = pd.MultiIndex.from_frame(cur[keys]) if k > 1 else cur[keys[0]]
            p = pd.Series(m_past.reindex(idx).to_numpy(), index=cur.index)
            ok = p.notna()
            if ok.sum() < 1000:
                continue
            dev = (p - r_past).abs()
            for thr in (0.15, 0.25, 0.35):
                sel = ok & (dev > thr)
                if sel.sum() < MIN_N:
                    continue
                y = cur.loc[sel, TARGET]
                rows.append({
                    "축": "+".join(keys), "임계": thr, "행수": int(sel.sum()),
                    "비율%": 100 * sel.sum() / len(cur),
                    "과거예측": float(p[sel].mean()),
                    "2024실제": float(y.mean()),
                    "|편차|": float(abs(y.mean() - r_cur)),
                })

    if not rows:
        print("=== 어떤 축 조합에서도 |과거 - 리그평균| > 0.15 인 그룹이 없다 ===")
        print("-> 상황 축만으로는 결정론적 하위집단이 존재하지 않는다.")
    else:
        t = pd.DataFrame(rows).sort_values("|편차|", ascending=False)
        print("=== 시즌 간 이전되는 극단 그룹 (2021~23 로 만들고 2024 로 확인) ===")
        print(t.head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 참고: 축 하나짜리 최대 편차 — 애초에 상황이 타깃을 얼마나 흔드는가
    print("\n=== 참고: 단일 축의 2024 실제 성공률 범위 ===")
    for a in axes:
        g = cur.groupby(a, observed=True)[TARGET].agg(["mean", "size"])
        g = g[g["size"] >= MIN_N]
        if g.empty:
            continue
        print(f"  {a:<12} {g['mean'].min():.4f} ~ {g['mean'].max():.4f} "
              f"(폭 {g['mean'].max() - g['mean'].min():.4f}, 그룹 {len(g)})")
    print("\n※ 폭이 0.3 이상인 축이 없으면 '확신할 수 있는 상황'이 애초에 없다는 뜻이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
