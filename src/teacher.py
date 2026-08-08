"""증류 교사 — 학생이 못 보는 정보까지 써서 **부드러운 타깃**을 만든다 (E165).

## 왜

라벨 분산은 p(1-p)≈0.25 인데 우리가 설명하는 분산은 0.0025 다. **잡음이 신호의
100배**다. 0/1 대신 E[y|x] 추정치로 학습하면 분할 결정의 분산이 크게 준다.

## 순차 정보

asof 한 투구 차분으로 **직전 투구의 결과**를 99.9% 복원할 수 있다(train 전용).
실측: 직전 실패 → 현재 0.5235 / 직전 성공 → 0.5591 (차이 3.6%p, BSS 상당 +127).
그리고 **볼카운트가 이걸 전혀 흡수하지 못한다**(조건부로 오히려 +134).
완전히 독립적인 신호다 — 투수의 그 순간 상태가 연속 투구에 이어진다.

⚠️ 평가 시점엔 못 쓴다. 직전 투구는 test.csv 의 **다른 행**이고, 같은 대수를
test 에 적용하면 2025 타깃이 96.79% 복원된다(실격). 그래서 **교사만** 이 피처를
쓰고 제출되는 학생은 합법 피처만 쓴다. 교사는 zip 에 안 들어간다.

## 누수 차단

교사가 자기 라벨을 본 채로 예측하면 부드러운 타깃이 y 에 붙어 증류 효과가 사라진다.
K-fold 로 **out-of-fold** 예측만 쓴다.

실행:
  python src/teacher.py --tag T_seq --prev            # 직전 투구 포함
  python src/teacher.py --tag T_self                  # 자기증류(합법 피처만)
  python src/teacher.py --tag T2_self --max-season 2023
"""

import argparse
import time

import numpy as np
import pandas as pd

import failmode as fm
import fpipe
from train_gbdt2 import CAT_COLS, TARGET, load


class A:                       # v14 base 와 같은 피처 설정
    feat_v2 = feat_std = te_dev = feat_domain = feat_skill_pc = True
    feat_prof = feat_form = feat_window = feat_count = False
    feat_cross = feat_skill = te_flat = False
    te = "p,pc,ph,b,pi"
    te_k = "50"
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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prev", action="store_true",
                    help="직전 투구 결과를 교사 피처로 추가 (train 전용)")
    ap.add_argument("--max-season", type=int, default=0,
                    help="이 시즌 이하 행만 OOF 교사 학습·예측에 사용")
    ap.add_argument("--drop-f-pre", type=int, default=0,
                    help="이 시즌 이전 F리그 행을 load 단계에서 제외")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, features, tm = load(drop_f_pre=args.drop_f_pre)
    if args.max_season:
        n0 = len(train)
        train = train.loc[train["season"] <= args.max_season].sort_index().copy()
        print(f"교사 상한 시즌 {args.max_season}: {n0} -> {len(train)}행")

    is_val = pd.Series(False, index=train.index)      # 교사는 전체를 쓴다
    train, new_cols, new_cats, _ = fpipe.fit(train, A, ~is_val, tm)
    features = features + [c for c in new_cols if c not in features]
    CAT_COLS[:] = [c for c in CAT_COLS + new_cats if c in features]

    if args.prev:
        # ⚠️ train 전용. 같은 투수의 직전 투구 결과 — 학생은 절대 못 본다.
        lab = fm._pitch_labels(train, modes=("middle", "ball", "reverse"))
        d = pd.concat([train[["pitcher_id", TARGET]], lab], axis=1)
        d = d.sort_index()
        add = []
        for c in (TARGET, "middle", "ball", "reverse"):
            nm = f"prev_{c}"
            train[nm] = d.groupby("pitcher_id")[c].shift(1)
            add.append(nm)
        features = features + add
        print(f"교사 전용 피처 {len(add)}개 추가 | 복원률 "
              f"{train['prev_' + TARGET].notna().mean() * 100:.1f}%")

    y = train[TARGET].to_numpy(np.float64)
    r = float(y.mean())
    base = r * (1 - r)
    rng = np.random.default_rng(args.seed)
    fold = rng.integers(0, args.folds, len(train))
    oof = np.full(len(train), np.nan)

    from catboost import CatBoostClassifier, Pool
    for c in CAT_COLS:
        train[c] = train[c].astype(str)
    for k in range(args.folds):
        t0 = time.time()
        tr, va = fold != k, fold == k
        m = CatBoostClassifier(
            iterations=args.iters, learning_rate=args.lr, depth=args.depth,
            l2_leaf_reg=10, border_count=254, task_type="GPU", devices="0",
            loss_function="Logloss", random_seed=args.seed, verbose=0)
        m.fit(Pool(train.loc[tr, features], y[tr], cat_features=CAT_COLS))
        oof[va] = m.predict_proba(train.loc[va, features])[:, 1]
        b = 1e5 * (1 - ((oof[va] - y[va]) ** 2).mean() / base)
        print(f"  fold {k}: OOF BSS {b:.1f} | {time.time() - t0:.0f}s",
              flush=True)

    tot = 1e5 * (1 - ((oof - y) ** 2).mean() / base)
    print(f"\n교사 OOF 전체 BSS {tot:.1f} | 평균 {oof.mean():.4f} "
          f"(실제 {r:.4f}) | sd {oof.std():.4f}")
    np.savez_compressed(f"./out/teacher_{args.tag}.npz",
                        row_id=train["row_id"].to_numpy(), prob=oof)
    print(f"저장: ./out/teacher_{args.tag}.npz")
    print("※ 이 파일은 **학습 타깃**으로만 쓴다. 제출 zip 에 안 들어간다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
