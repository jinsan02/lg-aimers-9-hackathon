"""공동 하이퍼파라미터 탐색 (E142) — 우리가 건너뛴 유일한 로드맵 항목.

## 왜 지금

지금까지 "튜닝은 무의미하다"고 결론냈는데, 근거는 **1차원 스윕 10번**이었다.
축 간 교호작용은 그렇게 안 보인다. 오늘만 두 번 반증됐다:

  - depth 7 이 std-k 30 에서 +6.5 였다가 k=100 에서 -0.32 로 사라짐
  - 셀 멤버가 depth 8 에서 D=25.5 였다가 **depth 5 에서 D=2.8** (1차원 스윕으로는
    못 찾는다 — depth 는 이미 8 로 '확정'돼 있었으니까)

## 설계

  - **데이터를 한 번만 읽는다.** 매 trial 마다 CSV+피처를 다시 만들면 32초씩
    낭비된다(200 trial 이면 1.8시간). 여기서는 프로세스 하나 안에서 돈다.
  - **trial 당 2시드.** 시드 잡음 sd 가 4.5 라 1시드로는 +4 짜리 차이를 못 가른다.
    2시드 평균이면 SE 3.2 — 여전히 크지만 TPE 는 잡음에 어느 정도 강하다.
  - 목적함수는 **편향제거 BSS**. 수준 편향은 SHIFT 가 따로 잡으므로 튜닝이
    편향을 줄이는 쪽으로 새면 안 된다 (LEVERS 측정 원칙).

실행: python src/tune.py --trials 120 --seeds 42,7 --tag T1
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import fpipe
from train_gbdt2 import CAT_COLS, TARGET, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--tag", default="T1")
    ap.add_argument("--val-season", type=int, default=2024)
    ap.add_argument("--timeout", type=float, default=0, help="초. 0=무제한")
    args = ap.parse_args()

    # ── 피처는 한 번만 만든다 ────────────────────────────────────────────
    class A:                       # fpipe.fit 이 보는 플래그 묶음
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

    train, features, tm = load()
    is_val = (train["season"] == args.val_season)
    train, new_cols, new_cats, _ = fpipe.fit(train, A, ~is_val, tm)
    features = features + [c for c in new_cols if c not in features]
    CAT_COLS[:] = [c for c in CAT_COLS + new_cats if c in features]
    y_va = train.loc[is_val, TARGET].to_numpy(np.float64)
    r = float(y_va.mean())
    base = r * (1 - r)
    Xtr = train.loc[~is_val, features]
    ytr = train.loc[~is_val, TARGET]
    Xva = train.loc[is_val, features]
    print(f"피처 {len(features)}개 | 학습 {len(Xtr):,} / 검증 {len(Xva):,}",
          flush=True)

    from catboost import CatBoostClassifier, Pool
    tr_pool = Pool(Xtr, ytr, cat_features=CAT_COLS)
    va_pool = Pool(Xva, train.loc[is_val, TARGET], cat_features=CAT_COLS)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    def score(p):
        # **편향제거 후** 평가 — 수준은 SHIFT 가 따로 잡으므로 튜닝이 그쪽으로
        # 새면 안 된다 (LEVERS 측정 원칙).
        p = p - (p.mean() - r)
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y_va) ** 2).mean() / base)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t_start = time.time()
    best = {"v": -1e9}

    def objective(t):
        pr = dict(
            learning_rate=t.suggest_float("lr", 0.005, 0.04, log=True),
            depth=t.suggest_int("depth", 6, 9),
            l2_leaf_reg=t.suggest_float("l2", 1.0, 60.0, log=True),
            border_count=t.suggest_categorical("border", [128, 254]),
            random_strength=t.suggest_float("rs", 0.0, 2.0),
            bagging_temperature=t.suggest_float("bt", 0.0, 1.0),
            min_data_in_leaf=t.suggest_int("minleaf", 1, 200, log=True),
        )
        vs = []
        for sd in seeds:
            m = CatBoostClassifier(iterations=4000, early_stopping_rounds=400,
                                   task_type="GPU", devices="0",
                                   loss_function="Logloss", verbose=0,
                                   random_seed=sd, **pr)
            m.fit(tr_pool, eval_set=va_pool)
            vs.append(score(m.predict_proba(Xva)[:, 1]))
        v = float(np.mean(vs))
        if v > best["v"]:
            best.update(v=v, params=pr, iters=int(m.get_best_iteration()))
            print(f"  ★ {v:.2f}  {json.dumps({k: round(x, 4) if isinstance(x, float) else x for k, x in pr.items()})}",
                  flush=True)
        return v

    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(objective, n_trials=args.trials,
                timeout=args.timeout or None)
    os.makedirs("./out", exist_ok=True)
    with open(f"./out/tune_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump({"best": best, "n": len(st.trials),
                   "elapsed_s": time.time() - t_start}, f,
                  ensure_ascii=False, indent=2, default=str)
    print(f"\n최고 {best['v']:.2f} | trial {len(st.trials)}개 | "
          f"{(time.time() - t_start) / 60:.0f}분")
    print(json.dumps(best.get("params", {}), indent=2, default=str))
    df = st.trials_dataframe()
    print(df.nlargest(8, "value")[["value"] + [c for c in df.columns
                                               if c.startswith("params_")]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
