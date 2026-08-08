"""GPBoost (E150) — 트리(상황)와 그룹 랜덤효과(투수·타자)를 **동시에** 적합한다.

## 왜

우리 파이프라인은 선수 능력을 세 번 우회해서 다룬다: ① asof_* (주최측 누적),
② TE k=50 (손으로 정한 수축), ③ skill.py (별도 선형회귀 → 예측을 피처로).
전부 **본 모델 바깥에서 정해진 값**을 트리에 먹이는 구조다.

GPBoost 는 랜덤효과의 분산성분을 부스팅과 함께 최대가능도로 추정한다.
  y ~ Bernoulli(logit^-1( F(x) + b_pitcher + b_batter ))
  b ~ N(0, sigma^2)   ← sigma 를 데이터가 정한다 (우리 k=50/80 은 손으로 정했다)

우리에게 맞을 것으로 보는 근거:
  - cold-start 가 원리적으로 처리된다. 신규 투수 → b=0 → 고정효과만.
    2024 신규 투수가 행 기준 20% 다.
  - E116 에서 **학습된 결합이 손 수축을 21.7%p 이겼다**(59.0% vs 37.3%).
    같은 방향의 증거를 이미 갖고 있다.
  - Sigrist(JMLR 2022) 고카디널리티 벤치마크에서 CatBoost 가 특히 나빴다.
    단, 그건 저자 자신의 벤치마크이고 데이터도 랜덤효과 구조를 전제로 만든
    것들이다. **가설이지 사실이 아니다.**

유보: asof_pitcher_success_rate 가 이미 랜덤효과의 상당 부분을 담고 있다.
그래서 --drop-asof 로 '랜덤효과가 그 일을 대신하게' 두는 변형도 같이 잰다.

실행: python src/train_gpb.py --tag G1 --sub 0.2 --iters 500
"""

import argparse
import time

import numpy as np

import fpipe
from train_gbdt2 import CAT_COLS, TARGET, load


class A:                       # fpipe.fit 이 보는 플래그 묶음 (v14 base 와 동일)
    feat_v2 = feat_std = te_dev = feat_domain = feat_skill_pc = True
    feat_prof = feat_form = feat_window = feat_count = False
    feat_cross = feat_skill = te_flat = False
    te = "p,pc,ph,b,pi"
    te_k = 50.0
    te_halflife = 0.0
    feat_k = 50.0
    std_k = 80.0
    std_to_prior = std_season_prior = True
    std_multi_k = ""
    std_excess = std_ratio = False
    std_k_mix = std_k_bat = 0.0
    prof_k = 60.0
    prof_lags = 2
    skill_axes = ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="G1")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--val-season", type=int, default=2024)
    ap.add_argument("--sub", type=float, default=0.0,
                    help="학습 서브샘플 비율. 0=전체. 속도 탐침용")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--nthreads", type=int, default=56)
    ap.add_argument("--groups", default="pitcher_id,batter_id")
    ap.add_argument("--drop-asof", action="store_true",
                    help="asof_pitcher_* 를 빼고 랜덤효과가 그 역할을 하게 둔다")
    # --drop-asof 만으로는 검정이 안 된다: te_pitcher_*, std_*, skill_* 이
    # 남아서 투수 정체성을 계속 나른다. 랜덤효과가 '할 일이 남았는지' 보려면
    # 투수 유래 피처를 **전부** 빼야 한다.
    ap.add_argument("--drop-pre", default="",
                    help="쉼표로 구분한 접두사. 해당 피처를 전부 제외")
    args = ap.parse_args()

    train, features, tm = load()
    is_val = (train["season"] == args.val_season).to_numpy()
    train, new_cols, _new_cats, _art = fpipe.fit(train, A, ~is_val, tm)
    features = features + [c for c in new_cols if c not in features]

    gcols = [c for c in args.groups.split(",") if c.strip()]
    feats = [c for c in features if c not in gcols]
    if args.drop_asof:
        feats = [c for c in feats if not c.startswith("asof_pitcher_")]
    pre = tuple(p for p in args.drop_pre.split(",") if p.strip())
    if pre:
        drop = [c for c in feats if c.startswith(pre)]
        feats = [c for c in feats if c not in drop]
        print(f"제외 {len(drop)}개: {', '.join(drop[:6])}...")
    cats = [c for c in CAT_COLS if c in feats]

    X = train[feats].copy()
    for c in cats:                      # LightGBM 계열은 정수 코드를 받는다
        X[c] = X[c].astype("category").cat.codes.astype(np.int32)
    X = X.astype(np.float32).to_numpy()
    y = train[TARGET].to_numpy(np.float64)
    G = train[gcols].astype(str).to_numpy()

    tr = ~is_val
    if args.sub > 0:
        rng = np.random.default_rng(0)
        keep = rng.random(len(y)) < args.sub
        tr = tr & keep
    y_va = y[is_val]
    r = float(y_va.mean())
    base = r * (1 - r)
    n_new = len(set(map(tuple, G[is_val])) - set(map(tuple, G[tr])))
    print(f"피처 {len(feats)}개(범주 {len(cats)}) | 학습 {tr.sum():,} / "
          f"검증 {is_val.sum():,} | 그룹 {gcols} | 검증 신규조합 {n_new:,}",
          flush=True)

    import gpboost as gpb

    def bss(p):
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y_va) ** 2).mean() / base)

    for sd in [int(s) for s in args.seeds.split(",") if s.strip()]:
        t0 = time.time()
        gp = gpb.GPModel(group_data=G[tr], likelihood="bernoulli_logit")
        ds = gpb.Dataset(X[tr], label=y[tr], categorical_feature=cats and
                         [feats.index(c) for c in cats])
        params = {"objective": "binary", "learning_rate": args.lr,
                  "num_leaves": args.leaves, "min_data_in_leaf": args.min_leaf,
                  "verbose": 0, "num_threads": args.nthreads, "seed": sd}
        bst = gpb.train(params=params, train_set=ds,
                        num_boost_round=args.iters, gp_model=gp)
        t_fit = time.time() - t0
        gp.summary()
        pr = bst.predict(data=X[is_val], group_data_pred=G[is_val],
                         predict_var=False, pred_latent=False)
        p = np.asarray(pr["response_mean"], dtype=np.float64)
        pc = p - (p.mean() - r)          # 수준 편향은 SHIFT 가 따로 잡는다
        print(f"[gpb {args.tag}_s{sd}] val{args.val_season} BSS {bss(p):.2f} "
              f"| 편향제거 {bss(pc):.2f} | 예측평균 {p.mean():.4f} vs {r:.4f} "
              f"| {t_fit:.0f}s", flush=True)
        np.savez_compressed(f"./out/gpb_{args.tag}_s{sd}_val_preds.npz",
                            y=y_va, pred=p)


if __name__ == "__main__":
    main()
