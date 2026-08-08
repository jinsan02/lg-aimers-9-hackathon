"""남은 여지를 **정직하게** 재는 도구 — 교차적합 오라클 + 잔차 구조 분석.

왜 필요한가:
  tools/signal_audit.py 의 오라클(투수 990.8 / 투수x카운트 2740.9)은
  **2024 실제 그룹 평균을 알 때**의 값이다. 그런데 투수x카운트는 400x12=4,800 그룹에
  245K 행이라 그룹당 ~50행이고, 50행 평균은 대부분 잡음이다. 그 잡음을 예측에 쓰면
  같은 데이터에서 평가할 때만 좋아 보인다. **도달 불가능한 허수**일 수 있다.

정직한 측정:
  ① 2024 를 A/B 로 나눈다.
  ② A 에서 그룹 평균(수축 포함)을 만들고 **B 에서 평가**한다.
  ③ 우리 모델 예측과 비교한다.
  이러면 '그 축에 실제로 얼마나 남아 있는가'가 나온다.

더 날카로운 버전 - **잔차 구조**:
  우리 모델의 잔차(y - pred)를 각 축으로 그룹지어, A 에서 잔차 평균을 구하고
  B 에서 그만큼 보정했을 때 BSS 가 오르는지 본다. 오르면 그 축에 **우리가 놓친
  구조**가 남아 있는 것이고, 안 오르면 이미 다 뽑아낸 것이다.

메모리: 2024 행만 필요한 컬럼으로 읽는다 (노트북에서 돌려도 안전).
실행: python tools/honest_ceiling.py
"""

import glob
import sys

import numpy as np
import pandas as pd

DATA = "./data"
TARGET = "control_success"
KS = (0.0, 10.0, 30.0, 100.0)      # 수축 강도 후보


def main():
    P = [np.load(f) for f in glob.glob("./out/*_v13f_s*_val_preds.npz")]
    if not P:
        P = [np.load(f) for f in glob.glob("./out/*_v11f_s*_val_preds.npz")]
    y = P[0]["y"].astype(np.float64)
    pred = np.mean([z["pred"] for z in P], 0).astype(np.float64)
    r = y.mean()
    base = r * (1 - r)

    use = ["season", "pitcher_id", "batter_id", "balls_before",
           "strikes_before", "pitcher_hand", "batter_hand", "inning",
           "base_state", "outs_before", TARGET]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig", usecols=use)
    df = df[df.season == 2024].reset_index(drop=True)
    assert len(df) == len(y), f"행 수 불일치 {len(df)} vs {len(y)}"
    print(f"2024 {len(df):,}행 | 모델 BSS {1e5 * (1 - ((pred - y) ** 2).mean() / base):.2f}\n")

    rng = np.random.default_rng(0)
    half = rng.random(len(df)) < 0.5          # A: True, B: False
    yb, pb = y[~half], pred[~half]

    def bss(q):
        return 1e5 * (1 - ((np.clip(q, 1e-6, 1 - 1e-6) - yb) ** 2).mean() / base)

    m0 = bss(pb)
    df["_cnt"] = df.balls_before.astype(str) + "-" + df.strikes_before.astype(str)
    AXES = {
        "투수": ["pitcher_id"],
        "투수 x 카운트": ["pitcher_id", "_cnt"],
        "투수 x 타자손": ["pitcher_id", "batter_hand"],
        "투수 x 이닝": ["pitcher_id", "inning"],
        "타자": ["batter_id"],
        "카운트": ["_cnt"],
        "투수 x 주자상태": ["pitcher_id", "base_state"],
    }

    print("=== ① 교차적합 오라클: 그 축의 그룹 평균만으로 예측 (A로 만들고 B에서 평가) ===")
    print(f"{'축':<18}{'그룹수':>9}{'그룹당행':>9}"
          + "".join(f"{f'k={k:.0f}':>10}" for k in KS))
    for name, keys in AXES.items():
        g = df.groupby(keys, observed=True)
        ng = g.ngroup().to_numpy()
        nG = int(ng.max()) + 1
        sa = np.bincount(ng[half], weights=y[half], minlength=nG)
        na = np.bincount(ng[half], minlength=nG)
        row = f"{name:<18}{nG:>9,}{len(df) / nG:>9.0f}"
        for k in KS:
            gm = (sa + k * r) / (na + k)
            row += f"{bss(gm[ng[~half]]):>10.1f}"
        print(row)
    print(f"{'(참고) 우리 모델':<18}{'':>9}{'':>9}{m0:>10.1f}")

    print("\n=== ② 잔차 구조: 우리 모델이 그 축에서 놓친 게 남아 있는가 ===")
    res = y - pred
    print(f"{'축':<18}" + "".join(f"{f'k={k:.0f}':>10}" for k in KS)
          + "   (모델 대비 이득)")
    for name, keys in AXES.items():
        g = df.groupby(keys, observed=True)
        ng = g.ngroup().to_numpy()
        nG = int(ng.max()) + 1
        sa = np.bincount(ng[half], weights=res[half], minlength=nG)
        na = np.bincount(ng[half], minlength=nG)
        row = f"{name:<18}"
        for k in KS:
            adj = sa / (na + k)               # 그룹 잔차 평균(수축)
            row += f"{bss(pb + adj[ng[~half]]) - m0:>10.2f}"
        print(row)
    # ---- (3) 시즌 이전성: 보정이 다음 시즌으로 넘어가는가
    print("")
    print("=== (3) 카운트 효과를 2021~23 에서 만들어 2024 에 적용 ===")
    d2 = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig", usecols=use)
    d2 = d2[d2.season.between(2021, 2023)].copy()
    d2["_cnt"] = d2.balls_before.astype(str) + "-" + d2.strikes_before.astype(str)
    tab = {}
    for se, sub in d2.groupby("season"):
        tab[int(se)] = sub.groupby("_cnt")[TARGET].mean() - sub[TARGET].mean()
    cur = df.groupby("_cnt")[TARGET].mean() - df[TARGET].mean()
    T = pd.DataFrame(tab)
    T["y2024"] = cur
    print((T * 1000).round(1).to_string())
    print("  (x1000. 시즌 간 값이 비슷하면 구조적 효과라 이전된다)")
    cc = T[[2021, 2022, 2023]].mean(1)
    print(f"  2021~23 평균 vs 2024 상관 {np.corrcoef(cc, cur)[0, 1]:.4f}")
    off = df["_cnt"].map(cc).to_numpy(np.float64)
    ra, oa = res[half], off[half]
    beta = float((ra * oa).sum() / max((oa * oa).sum(), 1e-12))
    print(f"  잔차 회귀계수 beta = {beta:+.4f} (1.0 이면 모델이 전혀 반영 못 함)")
    print(f"  B 반쪽 적용 이득 {bss(pb + beta * off[~half]) - m0:+.2f} BSS")
    print(f"  beta=1 강제 시     {bss(pb + off[~half]) - m0:+.2f} BSS")
    print("\n※ ②가 양수면 그 축에 **아직 뽑을 게 남아 있다**. 0 근처면 포화다.")
    print("※ k=0 은 수축 없음 - 그룹이 작으면 잡음 과적합으로 크게 음수가 난다.")
    print("   그게 바로 signal_audit 오라클이 부풀려진 이유다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
