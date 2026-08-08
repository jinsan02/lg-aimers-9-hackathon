"""CatBoost / XGBoost 학습 (GPU) — E06 스타일 설정 이식.

실행(4070 Windows): python -m uv run python src\train_gbdt2.py --model cat
                    python -m uv run python src\train_gbdt2.py --model xgb
"""

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

import fpipe

DATA = "./data"
ID, TARGET = "row_id", "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand",
            "batter_hand", "pitcher_team_id", "batter_team_id"]
DROP = {"pitcher_id", "batter_id"}  # E06 발견 적용

# tools/column_relations.py 실측: 시즌마다 타깃 상관 부호가 뒤집히는 = 이전되지 않는 신호
UNSTABLE = {"asof_batter_n", "asof_batter_middle_rate", "asof_pitcher_ball_rate",
            "asof_pitcher_offspeed_rate", "game_month", "pitcher_team_id",
            "run_top_before", "pitcher_hand", "asof_pitcher_n",
            "asof_pitcher_pitchmix_n", "asof_pitcher_strike_rate",
            "strikes_before", "batter_hand", "outs_before",
            "asof_pitcher_fastball_rate", "batter_team_id", "runner_on_3b",
            "runner_on_2b", "away_win_expectancy", "home_win_expectancy",
            "game_dayofweek", "score_diff_home", "runner_on_1b",
            "num_runners_on"}

# |r| ≈ 1.0 완전 중복 쌍에서 제거할 쪽
REDUNDANT = {"away_win_expectancy",        # home_win_expectancy와 r=1.000
             "asof_pitcher_pitchmix_n",    # asof_pitcher_n과 r=1.000
             "run_top_before"}             # run_total_before와 r=0.804


def bss(y, p):
    r = y.mean()
    return float(max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))))


def load(tm_feats_path="", drop_f_pre=0, drop_unstable=False,
         drop_redundant=False, keep_ids=False, league=""):
    test_cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
    drop = set() if keep_ids else set(DROP)
    if keep_ids:
        # E06(ID 제외 +205)은 **LightGBM** 결과다. LightGBM은 고카디널리티 정수를
        # 수치로 쪼개 과적합하지만, CatBoost는 ordered target statistics로
        # 누수 없이 처리한다 → CatBoost에서 재검증 필요 (E88).
        for c in ("pitcher_id", "batter_id"):
            if c not in CAT_COLS:
                CAT_COLS.append(c)
    if drop_unstable:
        drop |= UNSTABLE
    if drop_redundant:
        drop |= REDUNDANT
    features = [c for c in test_cols if c != ID and c not in drop]
    if len(drop) > len(DROP):
        print(f"피처 제거 {len(drop) - len(DROP)}개 → 사용 {len(features)}개")
    # row_id 는 피처가 아니지만 실패모드 라벨 복원이 시간 순서를 필요로 한다
    use = list(dict.fromkeys(features + [TARGET, ID, "pitcher_id", "batter_id"]))
    train = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig", usecols=use)
    if league:
        # 한 리그만 남긴다. F리그는 ABS 를 2023 에 도입했으므로 F 2024 는 ABS
        # 2년차이고 학습 마지막 시즌(F 2023)은 1년차다 - **R리그 2024->2025 와
        # 정확히 같은 구조**라 최신시즌 가중을 판정할 유일한 대조군이다.
        before = len(train)
        train = train[train.game_type == league].reset_index(drop=True)
        print(f"{league}리그만: {before:,} -> {len(train):,}행")
    if drop_f_pre:
        # 퓨처스(F) 리그는 2022→2023 사이 라벨 체계 변경 의심(성공률 0.71→0.47)
        # → 옛 체계 구간 제외
        before = len(train)
        train = train[~((train.game_type == "F") & (train.season <= drop_f_pre))]
        train = train.reset_index(drop=True)
        print(f"F리그 {drop_f_pre} 이전 제외: {before} → {len(train)}행")
    # 병합은 fpipe.fit 이 한다 (추론과 같은 함수를 쓰기 위해). 여기선 표만 읽는다.
    # 구종x볼카운트 표(E115)는 (pitcher_id, season, balls, strikes) 4키다.
    tm_table = pd.read_csv(tm_feats_path) if tm_feats_path else None
    return train, features, tm_table


# 도메인상 방향이 확실한 피처들 (+1 = 클수록 성공률↑)
MONO = {"asof_pitcher_success_rate": 1, "asof_pitcher_success_rate_shr": 1,
        "asof_pitcher_prev5_game_success_rate": 1,
        "asof_pitcher_prev3_game_success_rate": 1,
        "asof_pitcher_prev1_game_success_rate": 1,
        "asof_batter_success_rate": 1, "asof_batter_success_rate_shr": 1,
        "asof_pitcher_reverse_rate": -1, "asof_pitcher_reverse_rate_shr": -1,
        "asof_pitcher_middle_rate": -1, "asof_pitcher_middle_rate_shr": -1}


def _extra(args, features):
    """grow_policy / monotone 등 공통 추가 파라미터."""
    ex = {}
    if args.grow:
        ex["grow_policy"] = args.grow
        if args.grow in ("Depthwise", "Lossguide"):
            ex["task_type"] = "CPU"      # GPU는 비대칭 트리 미지원 조합 있음
    if args.monotone:
        mc = [MONO.get(c, 0) for c in features]
        ex["monotone_constraints"] = mc
        ex["task_type"] = "CPU"          # 단조 제약은 CPU 경로
        print(f"단조 제약 적용 {sum(1 for v in mc if v)}개 피처")
    return ex


def _decay_w(gamma, seasons):
    """시즌 지수감쇠 가중 (E109). gamma^(마지막시즌 - 그 행 시즌).

    근거(E105): 제구 성공률 드리프트가 시즌당 -0.008~-0.012 로 **꾸준하다**.
    ABS 도입조차 순효과 -0.004 뿐이었다. 즉 체제 변화는 계단이 아니라 기울기이므로
    '마지막 시즌만 x3' 이라는 계단 가중보다 매끄러운 감쇠가 형태상 맞다.
    """
    if gamma == 1.0:
        return np.ones(len(seasons), np.float64)
    last = float(np.max(seasons))
    return np.power(float(gamma), last - seasons.astype(np.float64))


def _refit_weights(args, train):
    """제출용 전체 재학습 시 가중치 — 최신 시즌(2024=ABS 1년차)을 강조.

    근거(E85): F리그 예행연습(2019~23 학습·2023 가중 → 2024 검증)에서
    W=3이 W=1 대비 +54.1 (3σ). F는 2023이 ABS 1년차, 2024가 2년차로
    R리그의 2024→2025 관계와 정확히 같은 구조.
    """
    if args.last_season_weight == 1.0 and args.season_decay == 1.0:
        return None
    se = train["season"].to_numpy()
    last = int(se.max())
    w = _decay_w(args.season_decay, se)
    if args.last_season_weight != 1.0:
        w = w * np.where(se == last, args.last_season_weight, 1.0)
    print(f"  refit 가중: {last} 시즌 ×{args.last_season_weight}"
          f" | 시즌 감쇠 {args.season_decay} (최저 {w.min():.3f})")
    return w


def _sample_weights(args, train, is_val, features):
    """표본 가중치 — 피처를 늘리지 않고 손실만 재배분한다.

    exp: 베테랑 가중. 근거(E72) — 경험 7k+ 투수의 시즌 간 성공률 표준편차가 0.0102로
         신인(0.0527)의 1/5. 즉 베테랑 행이 다음 시즌으로 더 잘 이전된다.
    adv: 적대적 검증 중요도 가중. 학습 행이 '검증 시즌처럼 보이는' 정도를 확률로 재고
         w = p/(1-p) 로 covariate shift를 보정하는 표준 기법.
    """
    tr = train.loc[~is_val]
    if args.val_last_weight != 1.0 or args.season_decay != 1.0:
        # 검증 단계에도 가중 (F리그 예행연습용 — 체제 구조가 맞을 때만 의미)
        se = tr["season"].to_numpy()
        last = int(se.max())
        w = _decay_w(args.season_decay, se)
        if args.val_last_weight != 1.0:
            w = w * np.where(se == last, args.val_last_weight, 1.0)
        print(f"검증학습 마지막 시즌({last}) 가중 ×{args.val_last_weight} "
              f"| 시즌 감쇠 {args.season_decay} (최저 {w.min():.3f})")
        return w
    if not args.weight_mode:
        return None
    if args.weight_mode == "exp":
        n = tr["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
        w = 1.0 + args.weight_alpha * (n / (n + 2000.0))
        print(f"경험 가중: 신인 {w.min():.2f} ~ 베테랑 {w.max():.2f} "
              f"(평균 {w.mean():.2f})")
        return w
    # adv — 학습 vs 검증 시즌 판별기
    # ⚠️ season/월 같은 시간 표지를 넣으면 판별이 자명해져 가중치가 퇴화한다(전부 동일).
    #    비시간 피처만으로 '검증 시즌스러움'을 재야 의미가 있다.
    from catboost import CatBoostClassifier, Pool
    TIME_MARK = {"season", "game_month", "month_cat", "season_progress",
                 "abs_year", "is_abs"}
    adv_f = [c for c in features if c not in TIME_MARK]
    adv_cat = [c for c in CAT_COLS if c in adv_f]
    y_dom = is_val.astype(int).to_numpy()
    clf = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
                             task_type="GPU", devices="0", verbose=0,
                             random_seed=args.seed)
    clf.fit(Pool(train[adv_f], y_dom, cat_features=adv_cat))
    p = clf.predict_proba(train.loc[~is_val, adv_f])[:, 1]
    p = np.clip(p, 0.02, 0.5)          # 극단 가중 방지
    w = p / (1 - p)
    w = w / w.mean()
    print(f"적대적 가중: AUC 대용 p 평균 {p.mean():.3f} | "
          f"w {w.min():.2f}~{w.max():.2f}")
    return w


def run_cat(args, train, features, is_val):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    for c in CAT_COLS:
        train[c] = train[c].astype(str)

    w_tr = _sample_weights(args, train, is_val, features)
    y_tr = train.loc[~is_val, TARGET].astype(float)
    # E165 증류: 0/1 대신 **교사의 부드러운 확률**로 학습한다.
    #   라벨 분산은 p(1-p)~0.25 인데 우리가 설명하는 분산은 0.0025 다 — 잡음이
    #   신호의 100배다. 타깃을 E[y|x] 추정치로 바꾸면 분할 결정의 분산이 준다.
    #   검증 라벨은 절대 안 바꾼다(아래 va 는 TARGET 그대로).
    if "_soft" in train.columns:
        y_tr = train.loc[~is_val, "_soft"].astype(float)
        if args.loss == "Logloss":
            args.loss = "CrossEntropy"
    if args.label_smooth > 0:
        e = args.label_smooth
        y_tr = y_tr * (1 - 2 * e) + e
        # 평활된 타깃은 [0,1] 실수 → Logloss가 아닌 CrossEntropy가 받는다
        if args.loss == "Logloss":
            args.loss = "CrossEntropy"
    tr = Pool(train.loc[~is_val, features], y_tr, cat_features=CAT_COLS,
              weight=w_tr)
    va = Pool(train.loc[is_val, features], train.loc[is_val, TARGET],
              cat_features=CAT_COLS)

    # E124: 실패모드 셀 다중분류. 셀에 **타깃 비트를 포함**하므로
    # P(성공) = sum(성공 셀) 이 정확히 성립한다. 실패모드만으로는 타깃이
    # 재현되지 않는다(최선의 합집합 84.1%). 라벨은 train 행끼리만 만든다.
    if args.failmode_cells:
        import failmode as fm
        code, names, succ = fm.build_cells(
            train, modes=tuple(m.strip() for m in args.fm_modes.split(",")
                               if m.strip()),
            context=args.fm_context, min_share=args.fm_min_share)
        tr = Pool(train.loc[~is_val, features], code[~is_val],
                  cat_features=CAT_COLS, weight=w_tr)
        va = Pool(train.loc[is_val, features], code[is_val],
                  cat_features=CAT_COLS)
        clf = CatBoostClassifier(
            iterations=args.iters, learning_rate=args.lr, depth=args.depth,
            l2_leaf_reg=args.l2, border_count=args.border_count,
            task_type=args.device, devices="0", loss_function="MultiClass",
            classes_count=len(names), early_stopping_rounds=args.es,
            random_seed=args.seed, verbose=200)
        clf.fit(tr, eval_set=va)
        cell_proba = clf.predict_proba(train.loc[is_val, features])
        p = fm.success_prob(cell_proba, succ)
        best_iter = clf.get_best_iteration()
        clf._fm_success = sorted(succ)      # 추론에서 성공 셀을 알아야 한다
        if args.dump_cell_proba:
            # 성공 셀을 합친 스칼라만 저장하면 14개 실패 구성의 정보가 사라진다.
            # 검증 모델의 전체 분포를 별도 산출물로 남겨 시간 전이 메타모델을
            # 검문한다. 모델 학습·점수·pkl에는 아무 영향이 없다.
            np.savez_compressed(
                f"./out/{args.model}_{args.tag}_cell_val.npz",
                y=train.loc[is_val, TARGET].to_numpy(np.float64),
                row_id=train.loc[is_val, "row_id"].to_numpy(),
                proba=cell_proba.astype(np.float32),
                success=np.array(sorted(succ), dtype=np.int16),
                names=np.array(names),
            )
        if args.no_refit:
            return clf, np.clip(p, 0.0, 1.0), best_iter
        full = Pool(train[features], code, cat_features=CAT_COLS,
                    weight=_refit_weights(args, train))
        final = CatBoostClassifier(
            iterations=max(int(best_iter * args.refit_mult), 1),
            learning_rate=args.lr, depth=args.depth, l2_leaf_reg=args.l2,
            border_count=args.border_count, task_type=args.device, devices="0",
            loss_function="MultiClass", classes_count=len(names),
            random_seed=args.seed, verbose=0)
        final.fit(full)
        final._fm_success = sorted(succ)
        return final, np.clip(p, 0.0, 1.0), best_iter

    # E141b: **다중라벨** (MultiLogloss). 셀 다중분류가 심플렉스(합=1)라면
    # 이건 독립 시그모이드 4개 + **공유 트리**다. 기하가 또 다르다 —
    # 한 트리가 성공/실투/볼/반대 4개를 동시에 설명해야 하므로 분할 기준이
    # 이진 모델과 계통적으로 달라진다. P(성공) 은 0번 헤드를 그대로 쓴다.
    if args.fm_multilabel:
        import failmode as fm
        lab = fm._pitch_labels(train)
        Y = np.column_stack([train[TARGET].to_numpy(np.float32)]
                            + [np.nan_to_num(lab[m].to_numpy(np.float32),
                                             nan=0.0) for m in fm.MODES])
        print(f"다중라벨 {Y.shape} (성공 + 실패모드 {len(fm.MODES)}개)")
        tr = Pool(train.loc[~is_val, features], Y[(~is_val).to_numpy()],
                  cat_features=CAT_COLS)
        va = Pool(train.loc[is_val, features], Y[is_val.to_numpy()],
                  cat_features=CAT_COLS)
        clf = CatBoostClassifier(
            iterations=args.iters, learning_rate=args.lr, depth=args.depth,
            l2_leaf_reg=args.l2, border_count=args.border_count,
            task_type=args.device, devices="0", loss_function="MultiLogloss",
            early_stopping_rounds=args.es, random_seed=args.seed, verbose=200)
        clf.fit(tr, eval_set=va)
        p = clf.predict_proba(train.loc[is_val, features])[:, 0]
        best_iter = clf.get_best_iteration()
        clf._fm_multilabel = True
        if args.no_refit:
            return clf, np.clip(p, 0.0, 1.0), best_iter
        full = Pool(train[features], Y, cat_features=CAT_COLS)
        final = CatBoostClassifier(
            iterations=max(int(best_iter * args.refit_mult), 1),
            learning_rate=args.lr, depth=args.depth, l2_leaf_reg=args.l2,
            border_count=args.border_count, task_type=args.device, devices="0",
            loss_function="MultiLogloss", random_seed=args.seed, verbose=0)
        final.fit(full)
        final._fm_multilabel = True
        return final, np.clip(p, 0.0, 1.0), best_iter

    # E120 재검정: skill 추정치를 **baseline(로짓 오프셋)** 으로 준다.
    # 앞선 --resid-col 은 타깃을 잔차로 바꾸면서 손실(RMSE)·링크(identity)까지
    # 같이 바꿔 6가지를 한 번에 건드렸고 -17.8 이 나왔다. baseline 은
    # Logloss·로짓을 그대로 두고 출발점만 옮기므로 가설을 단독으로 검정한다.
    if args.baseline_col:
        if args.baseline_col not in train.columns:
            raise KeyError(f"--baseline-col 없음: {args.baseline_col}")
        q = np.clip(train[args.baseline_col].astype(float).fillna(0.5), 1e-4, 1 - 1e-4)
        lg = np.log(q / (1 - q)).to_numpy()
        tr.set_baseline(lg[(~is_val).to_numpy()])
        va.set_baseline(lg[is_val.to_numpy()])
        print(f"baseline: {args.baseline_col} (로짓 평균 {lg.mean():.4f}, "
              f"sd {lg.std():.4f})")

    # E120: 2단 잔차 구조. 1단(선형 회귀)이 매끄러운 실력 수준을 맡고,
    # 2단(트리)은 **잔차만** 맞춘다. 근거 - 최적 수축은 매끄러운 가중평균이라
    # 트리가 계단으로만 근사한다(실측: 선형 59.0% vs GBDT 46.5%).
    # baseline API 대신 잔차를 직접 타깃으로 두어 구현을 단순하게 유지한다.
    bl = None
    if args.resid_col:
        if args.resid_col not in train.columns:
            raise KeyError(f"--resid-col 없음: {args.resid_col}")
        bl = train[args.resid_col].astype(float).fillna(0.5)
        y_tr = y_tr - bl.loc[~is_val]
        tr = Pool(train.loc[~is_val, features], y_tr, cat_features=CAT_COLS,
                  weight=w_tr)
        va = Pool(train.loc[is_val, features],
                  train.loc[is_val, TARGET] - bl.loc[is_val],
                  cat_features=CAT_COLS)
        args.loss = "RMSE"      # 잔차는 [0,1] 밖이므로 회귀 경로만 유효
        print(f"2단 잔차: 기준선 {args.resid_col} "
              f"(평균 {bl.mean():.4f}) | 잔차 sd {y_tr.std():.4f}")

    if args.loss == "RMSE":
        # 0/1 타깃 회귀 = MSE = Brier 그 자체. 평가지표를 직접 최소화한다.
        reg = CatBoostRegressor(
            iterations=args.iters, learning_rate=args.lr, depth=args.depth,
            l2_leaf_reg=args.l2, border_count=args.border_count,
            bagging_temperature=args.bagging_temp,
            random_strength=args.random_strength,
            task_type=args.device, devices="0", loss_function="RMSE",
            early_stopping_rounds=args.es, random_seed=args.seed, verbose=200)
        reg.fit(tr, eval_set=va)
        p = reg.predict(train.loc[is_val, features])
        if bl is not None:
            p = p + bl.loc[is_val].to_numpy()       # 기준선을 되돌린다
        p = np.clip(p, 0.0, 1.0)
        best_iter = reg.get_best_iteration()
        if args.no_refit:
            return reg, p, best_iter
        y_all = train[TARGET].astype(float)
        if bl is not None:
            y_all = y_all - bl
        if args.label_smooth > 0:
            e = args.label_smooth
            y_all = y_all * (1 - 2 * e) + e
        full = Pool(train[features], y_all, cat_features=CAT_COLS,
                    weight=_refit_weights(args, train))
        final = CatBoostRegressor(
            iterations=max(int(best_iter * args.refit_mult), 1),
            learning_rate=args.lr,
            depth=args.depth, l2_leaf_reg=args.l2,
            border_count=args.border_count,
            bagging_temperature=args.bagging_temp,
            random_strength=args.random_strength,
            task_type=args.device, devices="0", loss_function="RMSE",
            random_seed=args.seed, verbose=0)
        final.fit(full)
        return final, p, best_iter

    if args.transfer_split:
        # 1단계: 구체제 데이터로 base — 일반적 패턴(투수 실력·상황 논리)을 학습
        base_m = train.loc[~is_val] .index[
            train.loc[~is_val, "season"] <= args.transfer_split]
        new_m = train.loc[~is_val].index[
            train.loc[~is_val, "season"] > args.transfer_split]
        print(f"전이학습: base {len(base_m)}행(~{args.transfer_split}) "
              f"→ shift {len(new_m)}행({args.transfer_split + 1}~)")
        # init_model 이어붙이기는 GPU 미지원 → CPU 경로
        common = dict(depth=args.depth, l2_leaf_reg=args.l2,
                      border_count=args.border_count, task_type="CPU",
                      loss_function=args.loss,
                      random_seed=args.seed, verbose=0)
        base = CatBoostClassifier(iterations=400, learning_rate=args.lr, **common)
        base.fit(Pool(train.loc[base_m, features], train.loc[base_m, TARGET],
                      cat_features=CAT_COLS))
        # 2단계: 신체제 데이터만으로 이어서 boosting — 잔차가 체제 차이를 흡수
        shifted = CatBoostClassifier(iterations=args.transfer_iters,
                                     learning_rate=args.transfer_lr, **common)
        shifted.fit(Pool(train.loc[new_m, features], train.loc[new_m, TARGET],
                         cat_features=CAT_COLS),
                    init_model=base)
        p = shifted.predict_proba(train.loc[is_val, features])[:, 1]
        return shifted, p, args.transfer_iters

    ex = _extra(args, features)
    if args.cat_min_leaf:
        ex["min_data_in_leaf"] = args.cat_min_leaf
    params = dict(
        iterations=args.iters, learning_rate=args.lr, depth=args.depth,
        l2_leaf_reg=args.l2, border_count=args.border_count,
        bagging_temperature=args.bagging_temp,
        random_strength=args.random_strength,
        task_type=args.device, devices="0",
        loss_function=args.loss, eval_metric=args.eval_metric,
        early_stopping_rounds=args.es, random_seed=args.seed, verbose=200)
    params.update(ex)
    if params.get("task_type") == "CPU":
        params.pop("devices", None)
        params.pop("bagging_temperature", None)
    model = CatBoostClassifier(**params)
    model.fit(tr, eval_set=va)
    if args.baseline_col:
        # baseline 은 예측에도 같이 줘야 한다 (안 주면 오프셋 없이 예측된다)
        p = model.predict_proba(va)[:, 1]
        model._baseline_col = args.baseline_col
    else:
        p = model.predict_proba(train.loc[is_val, features])[:, 1]
    best_iter = model.get_best_iteration()
    if args.no_refit:
        return model, p, best_iter

    # ★ 전체 데이터(2024 포함) 재학습 — 제출용. drift 데이터라 최신 시즌 포함이 필수
    #   (E18 교훈: 2023까지만 학습한 모델은 2025에서 2년 외삽이 되어 LB 폭락)
    full = Pool(train[features],
                train["_soft"] if "_soft" in train.columns else train[TARGET],
                cat_features=CAT_COLS, weight=_refit_weights(args, train))
    if args.baseline_col:
        qf = np.clip(train[args.baseline_col].astype(float).fillna(0.5),
                     1e-4, 1 - 1e-4)
        full.set_baseline(np.log(qf / (1 - qf)).to_numpy())
    # 재학습은 **검증 모델과 같은 설정**이어야 한다. 한쪽에만 규제를 걸면 best_iter 는
    # 규제된 모델 것인데 제출 모델은 규제가 없어 서로 다른 함수가 된다.
    # ex(grow_policy / monotone / min_data_in_leaf)는 task_type 까지 바꿀 수 있으므로
    # 위 params 와 **똑같은 순서**로 적용한 뒤 CPU 면 devices 를 뺀다.
    fp = dict(
        iterations=max(int(best_iter * args.refit_mult), 1),
        learning_rate=args.lr, depth=args.depth,
        l2_leaf_reg=args.l2, border_count=args.border_count,
        bagging_temperature=args.bagging_temp,
        random_strength=args.random_strength,
        task_type=args.device, devices="0",
        # 증류면 재학습 타깃도 [0,1] 실수다 — Logloss 는 2값만 받으므로 터진다.
        # (검증 모델은 args.loss 를 따라가서 안 걸렸고 재학습만 죽었다.)
        loss_function=("CrossEntropy" if "_soft" in train.columns
                       else "Logloss"),
        random_seed=args.seed, verbose=0)
    fp.update(ex)
    if fp.get("task_type") == "CPU":
        fp.pop("devices", None)
        fp.pop("bagging_temperature", None)
    final = CatBoostClassifier(**fp)
    final.fit(full)
    if args.baseline_col:
        final._baseline_col = args.baseline_col
    return final, p, best_iter



def _rank_pool(train, features, mask, group_size=64):
    """시간적으로 가까운 행끼리 고정 블록을 만들어 전역 수준 대신 로컬 순서를 학습."""
    from catboost import Pool

    d = train.loc[mask, [*features, TARGET, ID]].copy()
    d = d.sort_values(ID, kind="stable")
    group = np.arange(len(d), dtype=np.int64) // int(group_size)
    pool = Pool(d[features], d[TARGET].astype(float), cat_features=CAT_COLS,
                group_id=group)
    return pool, d.index.to_numpy()


def _fit_rank_sigmoid(raw, y):
    """ranking raw score를 확률로. source 검증 시즌에서만 적합해 미래 시즌에 고정."""
    raw = np.asarray(raw, np.float64)
    y = np.asarray(y, np.float64)
    mu, sd = float(raw.mean()), float(raw.std())
    sd = max(sd, 1e-8)
    z = (raw - mu) / sd
    X = np.column_stack([np.ones(len(z)), z])
    w = np.array([np.log(y.mean() / (1 - y.mean())), 1.0], np.float64)
    for _ in range(30):
        q = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        v = np.maximum(q * (1 - q), 1e-8)
        grad = X.T @ (y - q)
        hess = (X * v[:, None]).T @ X + 1e-6 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(w[0]), float(w[1]), mu, sd


def _rank_probability(raw, calib):
    a, b, mu, sd = calib
    z = (np.asarray(raw, np.float64) - mu) / sd
    return 1.0 / (1.0 + np.exp(-np.clip(a + b * z, -30, 30)))


def run_rank(args, train, features, is_val):
    """O2: Brier의 해상도 항을 직접 노리는 pairwise ranking CatBoost."""
    from catboost import CatBoostRanker

    for c in CAT_COLS:
        train[c] = train[c].astype(str)
    tr, _ = _rank_pool(train, features, ~is_val, args.rank_group_size)
    va, va_order = _rank_pool(train, features, is_val, args.rank_group_size)
    model = CatBoostRanker(
        iterations=args.iters, learning_rate=args.lr, depth=args.depth,
        l2_leaf_reg=args.l2, border_count=args.border_count,
        task_type=args.device, devices="0", loss_function="PairLogitPairwise",
        eval_metric="PairLogit", early_stopping_rounds=args.es,
        random_seed=args.seed, verbose=200,
    )
    model.fit(tr, eval_set=va)
    raw_sorted = model.predict(va)
    raw = pd.Series(raw_sorted, index=va_order).reindex(train.index[is_val]).to_numpy()
    y = train.loc[is_val, TARGET].to_numpy(np.float64)
    calib = _fit_rank_sigmoid(raw, y)
    p = _rank_probability(raw, calib)
    best_iter = model.get_best_iteration()
    if args.no_refit:
        model._rank_calib = calib
        return model, p, best_iter

    # CatBoost GPU ranker keeps its pair buffers alive with the fitted model.
    # Constructing the refit ranker while ``model`` and both Pools still exist
    # caused group16 to request another ~2.75 GB and OOM on the 4070; group64
    # native-crashed even on A100.  Only p/calib/best_iter are needed below.
    import gc
    del model, tr, va, raw_sorted
    gc.collect()
    full, _ = _rank_pool(train, features, np.ones(len(train), dtype=bool),
                         args.rank_group_size)
    final = CatBoostRanker(
        iterations=max(int(best_iter * args.refit_mult), 1),
        learning_rate=args.lr, depth=args.depth, l2_leaf_reg=args.l2,
        border_count=args.border_count, task_type=args.device, devices="0",
        loss_function="PairLogitPairwise", random_seed=args.seed, verbose=0,
    )
    final.fit(full)
    final._rank_calib = calib
    return final, p, best_iter


def run_lgb(args, train, features, is_val):
    """LightGBM 멤버 (E141).

    CatBoost 는 **oblivious tree**(한 깊이의 모든 노드가 같은 분할)이고 LightGBM 은
    **leaf-wise**(손실이 가장 큰 리프부터 쪼갬)다. 같은 데이터·같은 피처를 줘도
    만들어지는 결정면이 근본적으로 다르다 — 블렌드가 요구하는 '계통적 불일치'다.

    지금까지 rms 는 CatBoost 하이퍼파라미터 변주 0.003 / 셀 다중분류 0.011 이
    전부였고 이득 상한(100,080 x rms^2)에 이미 닿았다. 분할 알고리즘 자체를
    바꾸면 그 위로 올라갈 수 있다. LEVERS 2군에 적혀만 있고 한 번도 안 했다.
    """
    import lightgbm as lgb
    X = train[features].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    tr_i, va_i = ~is_val, is_val
    ds_tr = lgb.Dataset(X[tr_i], train.loc[tr_i, TARGET],
                        categorical_feature=CAT_COLS, free_raw_data=False)
    ds_va = lgb.Dataset(X[va_i], train.loc[va_i, TARGET], reference=ds_tr)
    params = dict(objective="binary", metric="binary_logloss",
                  learning_rate=args.lr, num_leaves=args.num_leaves,
                  min_data_in_leaf=args.min_leaf, lambda_l2=args.l2,
                  feature_fraction=args.ff, bagging_fraction=args.bf,
                  bagging_freq=1, max_bin=255,
                  num_threads=args.nthreads,
                  seed=args.seed, verbosity=-1)
    m = lgb.train(params, ds_tr, num_boost_round=args.iters,
                  valid_sets=[ds_va],
                  callbacks=[lgb.early_stopping(args.es, verbose=False),
                             lgb.log_evaluation(200)])
    p = m.predict(X[va_i], num_iteration=m.best_iteration)
    best_iter = m.best_iteration
    if args.no_refit:
        return m, p, best_iter
    ds_all = lgb.Dataset(X, train[TARGET], categorical_feature=CAT_COLS)
    final = lgb.train(params, ds_all,
                      num_boost_round=max(int(best_iter * args.refit_mult), 1))
    return final, p, best_iter


def run_xgb(args, train, features, is_val):

    import xgboost as xgb
    for c in CAT_COLS:
        train[c] = train[c].astype("category")
    dtr = xgb.DMatrix(train.loc[~is_val, features], train.loc[~is_val, TARGET],
                      enable_categorical=True)
    dva = xgb.DMatrix(train.loc[is_val, features], train.loc[is_val, TARGET],
                      enable_categorical=True)
    params = {"objective": "binary:logistic", "eval_metric": "logloss",
              "device": "cuda", "tree_method": "hist",
              "learning_rate": args.lr, "max_leaves": 63, "max_depth": 0,
              "grow_policy": "lossguide", "min_child_weight": 500,
              "subsample": 0.8, "colsample_bytree": 0.7,
              "reg_lambda": args.l2, "seed": args.seed}
    evals_result = {}
    booster = xgb.train(params, dtr, num_boost_round=3000,
                        evals=[(dva, "val")], early_stopping_rounds=args.es,
                        verbose_eval=200, evals_result=evals_result)
    p = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    best_iter = booster.best_iteration
    if args.no_refit:
        return booster, p, best_iter

    # 전체 데이터(2024 포함) 재학습 — cat과 동일 (E18 교훈)
    dall = xgb.DMatrix(train[features], train[TARGET], enable_categorical=True)
    final = xgb.train(params, dall, num_boost_round=max(best_iter + 1, 1))
    return final, p, best_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cat", "xgb", "lgb", "rank"], required=True)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=8)      # catboost
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--border-count", type=int, default=254)
    ap.add_argument("--bagging-temp", type=float, default=1.0)
    ap.add_argument("--random-strength", type=float, default=1.0)
    # E142(Optuna) 재현용. tune.py 가 탐색한 7축 중 유일하게 여기 없던 축이다.
    # 0 이면 CatBoost 기본값을 그대로 둔다(파라미터 자체를 넘기지 않는다).
    ap.add_argument("--cat-min-leaf", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tm-feats", default="",
                    help="data/processed/tm_pitcher_feats.csv 경로")
    ap.add_argument("--tm-context", default="",
                    help="data/processed/tm_context.csv — 상황별 trackman 집계(링키지 불필요)")
    ap.add_argument("--val-season", type=int, default=2024,
                    help="검증 시즌 (롤링 윈도우: 2023으로 두면 2019~22 학습)")
    ap.add_argument("--test-season", type=int, default=0,
                    help=">0이면 이 시즌으로 최종 평가 (미학습·미검증 = 진짜 1년 앞 성능)")
    ap.add_argument("--grow", default="",
                    choices=["", "Depthwise", "Lossguide", "SymmetricTree"],
                    help="트리 성장 방식 — 구조적 다양성")
    ap.add_argument("--monotone", action="store_true",
                    help="도메인 단조 제약 (피처 추가 없이 함수 형태만 제한)")
    ap.add_argument("--pca", type=int, default=0,
                    help="수치 피처를 N개 주성분으로 **대체** (추가 아님)")
    ap.add_argument("--pca-add", type=int, default=0,
                    help="주성분 N개를 기존 피처에 추가")
    ap.add_argument("--top-k", type=int, default=0,
                    help="CatBoost 중요도 상위 K개 피처만 사용")
    ap.add_argument("--ptype", type=int, default=0,
                    help="투수 유형 클러스터 K개를 범주형 피처로 추가 (0=미사용)")
    ap.add_argument("--tm-pctx", default="",
                    help="data/processed/tm_pitcher_ctx.csv — 투수×카운트 편차(링키지 필요)")
    ap.add_argument("--drop-f-pre", type=int, default=0,
                    help="이 시즌 이하의 game_type=F 행 제외 (예: 2022)")
    ap.add_argument("--feat-v2", action="store_true",
                    help="features.py 파생 피처 추가")
    ap.add_argument("--feat-graph-topology", action="store_true",
                    help="완료된 과거 시즌의 투수-타자 그래프 구조/이웃 피처")
    ap.add_argument("--feat-v4", action="store_true",
                    help="엔티티×상황 타깃 통계 lookup (feat-v2 필요)")
    ap.add_argument("--feat-v5", action="store_true",
                    help="시즌 상대화 (drift 흡수). feat-v2 필요")
    ap.add_argument("--loss", default="Logloss",
                    choices=["Logloss", "RMSE", "CrossEntropy"],
                    help="RMSE = 0/1 타깃에 대한 MSE = **Brier 직접 최적화**")
    ap.add_argument("--row-filter", default="",
                    help="피처 생성 후 학습·평가 행을 거른다 (예: strikes_before==2)")
    ap.add_argument("--soft-target", default="",
                    help="교사 OOF 확률 npz (row_id, prob). 학습 타깃을 이것으로 교체")
    ap.add_argument("--label-smooth", type=float, default=0.0,
                    help="타깃을 y*(1-2e)+e 로 평활 (라벨 노이즈 대응)")
    ap.add_argument("--gametype", default="",
                    help="R 또는 F만으로 학습·검증 (체제가 다른 리그 분리 모델)")
    ap.add_argument("--min-season", type=int, default=0,
                    help="이 시즌 이상만 학습에 사용 (계단형 drift 대응)")
    ap.add_argument("--transfer-split", type=int, default=0,
                    help="이 시즌 이하로 base 학습 → 이후 시즌으로 이어붙여 boosting "
                         "(구체제 지식을 유지한 채 신체제로 시프트)")
    ap.add_argument("--transfer-lr", type=float, default=0.03,
                    help="이어붙이는 단계의 learning rate")
    ap.add_argument("--transfer-iters", type=int, default=300)
    ap.add_argument("--last-season-weight", type=float, default=1.0,
                    help="**refit(제출용)** 에서 최신 시즌 가중 배수 — "
                         "2024=ABS 1년차를 강조해 2025(2년차) 예측. 검증엔 영향 없음")
    ap.add_argument("--season-decay", type=float, default=1.0,
                    help="E109: 시즌 지수감쇠 가중 gamma^(마지막시즌-시즌). "
                         "드리프트가 계단이 아니라 기울기라는 E105 결과에 맞춘 형태")
    ap.add_argument("--val-last-weight", type=float, default=1.0,
                    help="검증 단계 학습에도 마지막 시즌 가중 (F리그 예행연습 전용)")
    ap.add_argument("--weight-mode", default="",
                    choices=["", "exp", "adv"],
                    help="exp=경험(베테랑) 가중 / adv=적대적 검증 기반 중요도 가중")
    ap.add_argument("--weight-alpha", type=float, default=1.5,
                    help="exp 가중 강도")
    ap.add_argument("--feat-mgr", action="store_true",
                    help="감독 투수 운용 스타일 (팀×시즌 집계)")
    ap.add_argument("--feat-role", action="store_true",
                    help="투수 역할·개인 상대 피로도 (train 이력 lookup)")
    ap.add_argument("--feat-fatigue", action="store_true",
                    help="체력·심리 파생 (이닝×경험, 접전 압박)")
    ap.add_argument("--feat-gap", action="store_true",
                    help="경력 공백 피처 (군 복무 복귀 등) — train 이력 lookup")
    ap.add_argument("--feat-abs", action="store_true",
                    help="ABS(자동 볼판정) 측정 체제 플래그")
    ap.add_argument("--feat-rules", action="store_true",
                    help="야구 규칙 도메인 피처 (src/rules.py)")
    ap.add_argument("--drop-unstable", action="store_true",
                    help="시즌 간 상관 부호가 뒤집히는 불안정 피처 제거")
    ap.add_argument("--drop-redundant", action="store_true",
                    help="완전 중복 피처 제거 (|r|≈1.0 쌍의 한쪽)")
    ap.add_argument("--feat-std", action="store_true",
                    help="당해 시즌 성적 복원 피처 (E99). asof 통산 누적을 "
                         "직전 시즌 말 앵커로 차분한다")
    ap.add_argument("--feat-anchor", action="store_true",
                    help="시즌 시작 전에 확정된 순수 과거 n0/S0를 수축된 앵커 "
                         "피처로 노출한다. 신규선수 missing 플래그는 포함하지 않는다")
    ap.add_argument("--std-season-prior", action="store_true",
                    help="수축 목표를 직전 완료 시즌 리그평균 수준에 맞춘다 (E102). "
                         "학습 전체 평균으로 수축하면 예측 시즌보다 위로 끌어올려 "
                         "전 구간 상향 편향이 생긴다")
    ap.add_argument("--std-to-prior", action="store_true",
                    help="시즌내 지표를 리그 사전확률로 수축(구버전). 기본은 자기 통산 rate로 수축")
    ap.add_argument("--std-multi-k", default="",
                    help="추가 수축 강도 목록 (쉼표) - 예: 10,100")
    ap.add_argument("--std-k-mix", type=float, default=0.0,
                    help="E112: 구종배합 그룹의 수축 강도를 따로 준다 (0=--std-k 와 동일)")
    ap.add_argument("--std-k-bat", type=float, default=0.0,
                    help="E112: 타자 그룹의 수축 강도를 따로 준다 (0=--std-k 와 동일)")
    ap.add_argument("--std-excess", action="store_true",
                    help="수축된 rate 대신 충분통계량을 직접 준다 (E104): "
                         "초과 성공 수 _ex 와 이항 표준화 z. 수축 강도 k를 "
                         "고정하지 않고 트리가 문맥별로 고르게 한다")
    ap.add_argument("--std-ratio", action="store_true",
                    help="E110: std 를 그 시즌 리그평균으로 나눠 배수로 만든다. "
                         "TE 는 이미 배수인데 std 만 절대 성공률로 남아 있어 "
                         "시즌 간 이전이 안 된다")
    ap.add_argument("--league", default="", choices=["", "R", "F"],
                    help="한 리그만 사용. F 는 ABS 를 2023 에 도입해 F2024=2년차 - "
                         "R리그 2024->2025 와 같은 구조라 가중 판정용 대조군이다")
    # 학습 데이터는 그대로 두고 **검증셋에서만** 한 리그를 뺀다.
    # 이유: F리그 라벨 체제가 2022(0.71)->2023(0.47) 에서 바뀌어, val 2023 구조에서는
    # 학습(<=2022)에 신체제 F 가 없어 조기종료가 15 iter 에서 죽는다. 그렇다고
    # --drop-f-pre 로 학습에서 빼버리면 **제출 모델과 학습 데이터가 달라져서**
    # 거기서 잰 상수를 제출에 쓸 수 없다(v16 이 그래서 -6.15 였다).
    ap.add_argument("--val-league", default="", choices=["", "R", "F"],
                    help="검증셋만 이 리그로 제한 (학습·평가 데이터는 그대로)")
    ap.add_argument("--eval-metric", default="Logloss",
                    choices=["Logloss", "BrierScore", "CrossEntropy"],
                    help="E121: 조기종료 지표. 대회 지표는 Brier(MSE)인데 지금까지 "
                         "Logloss 최소 지점을 골라왔다 - 두 최적점은 다르다")
    ap.add_argument("--refit-mult", type=float, default=1.0,
                    help="E122: refit 반복수 배수. 검증은 <=2023(1.22M행)로 적합하고 "
                         "refit 은 전체(1.475M행 = x1.21)로 하는데 반복수를 그대로 "
                         "쓰면 제출 모델이 과소적합이다. 로컬로는 안 보인다")
    ap.add_argument("--resid-col", default="",
                    help="E120: 2단 잔차 구조. 이 컬럼을 기준선으로 두고 트리는 "
                         "잔차만 학습한다 (예: skill_pc_hat). 자동으로 RMSE 경로")
    ap.add_argument("--seeds", default="",
                    help="시드 목록(쉼표). 한 프로세스에서 루프를 돌아 피처 준비 "
                         "비용(런당 약 32초)을 한 번만 낸다. 태그는 <tag>_s<seed>")
    ap.add_argument("--drop-cols", default="",
                    help="제거할 열 이름 (쉼표). tools/feature_audit.py 결과 시험용")
    ap.add_argument("--fill-prev", action="store_true",
                    help="prev1/3/5 경기 결측을 그 시즌 리그평균으로 채운다. "
                         "결측률이 시즌마다 1.11~5.17%로 달라 트리가 결측을 "
                         "시즌 표지로 쓸 수 있다 (E115 가 그렇게 졌다)")
    ap.add_argument("--dump-npz", default="",
                    help="피처 행렬을 npz 로 내보내고 종료 (NN 학습용). "
                         "같은 파이프라인을 두 번 구현하지 않기 위한 이음매")
    ap.add_argument("--feat-skill", action="store_true",
                    help="E116: 학습된 투수 실력 추정치를 피처로. 손으로 정한 "
                         "k=80 수축(설명력 37.3%)보다 학습된 선형결합이 59.0%")
    ap.add_argument("--feat-skill-pc", action="store_true",
                    help="E117: 실력 추정을 **투수x볼카운트** 단위로. 신호감사 "
                         "오라클이 투수 990.8 -> 투수x카운트 2740.9 로 최대다")
    ap.add_argument("--nthreads", type=int, default=0,
                    help="LightGBM 스레드. 0=전부. GPU 작업과 같은 "
                         "머신에서 돌면 반드시 제한할 것 — 전부 쓰면 "
                         "CatBoost GPU 가 데이터 공급을 못 받아 3배 느려진다")
    ap.add_argument("--num-leaves", type=int, default=127,
                    help="LightGBM leaf-wise 성장 폭")
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--ff", type=float, default=0.8,
                    help="LightGBM feature_fraction")
    ap.add_argument("--bf", type=float, default=0.8,
                    help="LightGBM bagging_fraction")
    ap.add_argument("--fm-context", default="",
                    help="셀 코드에 붙일 상황 컬럼 (예: strikes_before)")
    ap.add_argument("--fm-min-share", type=float, default=0.005,
                    help="이보다 드문 셀은 성공 비트만 남기고 병합")
    ap.add_argument("--fm-modes", default="middle,ball,reverse",
                    help="셀에 넣을 실패모드. 조합을 바꾸면 **출력 기하가 달라져** "
                         "형제 멤버와 계통적으로 다른 오차가 난다 (rms 상승)")
    ap.add_argument("--feat-frac", type=float, default=1.0,
                    help="멤버마다 피처를 이 비율만 무작위로 쓴다(부분공간 배깅). "
                         "블렌드 이득 상한이 100,080 x rms^2 인데 CatBoost 변주로는 "
                         "rms 0.017 이 한계다 — 입력을 줄이면 rms 를 직접 키운다")
    ap.add_argument("--fm-multilabel", action="store_true",
                    help="독립 시그모이드 4개 + 공유 트리(MultiLogloss). "
                         "셀 다중분류(심플렉스)와 또 다른 출력 기하")
    ap.add_argument("--failmode-cells", action="store_true",
                    help="E124: (성공,실투,볼,반대) 셀 다중분류로 학습하고 "
                         "P(성공)=성공 셀 합으로 복원. 출력 기하가 심플렉스로 "
                         "바뀌어 형제 CatBoost 와 불일치(rms)가 커진다")
    ap.add_argument("--dump-cell-proba", action="store_true",
                    help="failmode-cells 검증/test의 전체 클래스 확률을 저장. "
                         "학습이나 저장 모델은 바꾸지 않는 분석 전용 출력")
    ap.add_argument("--rank-group-size", type=int, default=64,
                    help="PairLogitPairwise 학습 블록 크기. row_id 시간순 고정 블록")
    ap.add_argument("--baseline-col", default="",
                    help="E120 재검정: 그 열의 logit 을 CatBoost baseline 으로 "
                         "준다. 손실·링크는 Logloss 그대로 유지한다 "
                         "(잔차를 타깃으로 바꾸는 --resid-col 과 다르다)")
    ap.add_argument("--skill-axes", default="",
                    help="실력 추정기를 붙일 축을 쉼표로. 빈 항목='' = 투수 단위, "
                         "count = 투수x볼카운트(E117), hand = 투수x타자손(E123). "
                         "예: --skill-axes count,hand")
    ap.add_argument("--feat-window", action="store_true",
                    help="E118: 중첩된 prev1/3/5 를 분리된 창(경기1 / 2~3 / 4~5)으로 "
                         "분해. 역산값이 100%% [0,1] 안에 들어와 분해가 정확하다")
    ap.add_argument("--feat-count", action="store_true",
                    help="E113: 카운트가 여는 실패 모드 x 그 모드에서의 투수 취약도. "
                         "3볼이면 볼 성향의 영향이 약해지고 2스트라이크면 한가운데 "
                         "성향의 영향이 약해진다(count_intent.py 실측)")
    ap.add_argument("--feat-domain", action="store_true",
                    help="E111: 야구 기전 교차항 (동일손x투수스타일, 타자위협도x주자유무). "
                         "tools/domain_probe.py 로 실측 선별한 둘만 넣는다")
    ap.add_argument("--feat-form", action="store_true",
                    help="E108: 최근 1/3/5경기 폼을 **당해 시즌 기준선** 대비로. "
                         "기존 form_delta 는 통산 대비라 체제 차이가 섞였다")
    ap.add_argument("--feat-cross", action="store_true",
                    help="레버 H: std x TE dev 교차항")
    ap.add_argument("--feat-prof", action="store_true",
                    help="시즌 궤적 피처 (E100). 직전 완료 시즌 성적/추세")
    ap.add_argument("--prof-k", type=float, default=60.0)
    ap.add_argument("--prof-lags", type=int, default=2)
    ap.add_argument("--std-k", type=float, default=30.0,
                    help="시즌내 복원 피처의 수축 강도")
    ap.add_argument("--max-train-season", type=int, default=0,
                    help="이 시즌 이후 데이터를 통째로 제거 - 한 시즌 앞 "
                         "예측 상황 재현 (편향 측정용, E96)")
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU"],
                    help="CatBoost 연산 장치. 머신간 재현성 조사용(E94)")
    ap.add_argument("--feat-k", type=float, default=200.0,
                    help="피처 v2 shrinkage 강도. 신호감사상 asof 투수 피처가 "
                         "지배 신호(240.8)이므로 이 평활 강도가 중요하다")
    ap.add_argument("--iters", type=int, default=3000,
                    help="최대 부스팅 반복 (보통 조기종료가 먼저 걸린다)")
    ap.add_argument("--es", type=int, default=100,
                    help="조기종료 라운드. 낮은 lr에는 크게 잡아야 한다")
    ap.add_argument("--te", default="",
                    help="시즌 expanding 타깃 인코딩 키 (쉼표 구분): "
                         "p, pc, ph, b, pi — target_enc.SPECS 참조")
    ap.add_argument("--te-k", default="50", help="TE 수축. 스칼라 또는 축별 b:500,*:50")
    ap.add_argument("--te-halflife", type=float, default=0.0,
                    help="TE 시즌 반감기 (0=가중 없음)")
    ap.add_argument("--te-flat", action="store_true",
                    help="TE 기대값을 시즌 전체평균으로 (구버전). 기본은 층화 — "
                         "투수 뺀 나머지 키x시즌 평균으로 나눠 순수 상호작용만 남긴다")
    ap.add_argument("--te-dev", action="store_true",
                    help="상호작용 TE를 투수 기준선(te_pitcher_ratio)으로 나눈 편차 추가")
    ap.add_argument("--keep-ids", action="store_true",
                    help="pitcher_id/batter_id를 **범주형**으로 유지 (E88). "
                         "E06의 ID 제외(+205)는 LightGBM 결과이므로 "
                         "CatBoost ordered TS에서는 결론이 다를 수 있다")
    ap.add_argument("--no-refit", action="store_true",
                    help="검증만 하고 전체 재학습 생략 (실험용)")
    args = ap.parse_args()

    train, features, tm_table = load(args.tm_feats, args.drop_f_pre,
                                     args.drop_unstable, args.drop_redundant,
                                     args.keep_ids, league=args.league)
    test_df = None
    if args.test_season:
        # 미래 시즌은 학습·검증 어디에도 안 들어가게 한다. 단 **행은 남겨 둔다** —
        # 여기서 떼어내면 std/TE/skill 등 뒤에서 만들어지는 피처가 test_df 에
        # 안 붙어 예측 시점에 KeyError 가 난다. 대신 플래그로 표시해 두고
        # 표(prior/TE/skill) 적합에서만 제외한 뒤, 학습 직전에 분리한다.
        # (이 피처들은 전부 시즌 expanding 이라 test 시즌 행의 변환값은
        #  자동으로 그 이전 시즌만 쓴다 = 추론과 같은 절차다)
        train["_is_test"] = (train["season"] == args.test_season)
    if args.max_train_season:
        # 한 시즌 앞 예측 상황을 재현한다 (E96 편향 측정용).
        # val_season 이후 시즌을 통째로 제거해야 '미래를 보고 학습'하지 않는다.
        n0 = len(train)
        train = train[train["season"] <= args.max_train_season].reset_index(drop=True)
        print(f"학습·검증 상한 시즌 {args.max_train_season}: {n0} -> {len(train)}행")
    is_val = train["season"] == args.val_season
    assert is_val.any(), f"검증 시즌 {args.val_season} 행 없음"
    CAT_COLS[:] = [c for c in CAT_COLS if c in features]
    if args.gametype:
        n0 = len(train)
        train = train[train.game_type == args.gametype].reset_index(drop=True)
        is_val = train["season"] == args.val_season
        print(f"game_type={args.gametype} 전용: {n0} → {len(train)}행 "
              f"(검증 {int(is_val.sum())})")

    if args.min_season:
        # 검증(2024)은 유지하고 학습 구간만 자른다
        keep = (train["season"] >= args.min_season) | is_val
        n0 = len(train)
        train = train[keep].reset_index(drop=True)
        is_val = train["season"] == args.val_season
        print(f"학습 시즌 {args.min_season}+ : {n0} → {len(train)}행 "
              f"(학습 {int((~is_val).sum())})")
    # 타깃을 쓰는 표 적합에서는 검증 시즌과 test 시즌을 **둘 다** 뺀다
    is_fit = ~is_val & ~train.get("_is_test",
                                  pd.Series(False, index=train.index))
    # 피처 생성은 fpipe 가 전담한다 — 추론(script_blend_v6)이 쓰는
    # fpipe.transform 과 **같은 파일에 나란히** 있어서 순서가 어긋날 수 없다.
    ctx_tables = role_table = mgr_table = None
    train, new_cols, new_cats, art = fpipe.fit(train, args, is_fit, tm_table)
    features = features + [c for c in new_cols if c not in features]
    CAT_COLS.extend([c for c in new_cats if c not in CAT_COLS])
    print(f"피처 총 {len(features)}개 (범주형 {len(CAT_COLS)})")
    if args.tm_context:
        ctx = pd.read_csv(args.tm_context)
        ck = ["balls_before", "strikes_before", "outs_before", "inning",
              "top_bottom", "pitcher_hand", "batter_hand"]
        train["_inn"] = train["inning"].clip(upper=12)
        merged = train.merge(ctx.rename(columns={"inning": "_inn"}),
                             on=[c if c != "inning" else "_inn" for c in ck],
                             how="left")
        tmc = [c for c in ctx.columns if c.startswith("tmc_")]
        for c in tmc:
            train[c] = merged[c].to_numpy()
        train = train.drop(columns=["_inn"])
        features = features + tmc
        print(f"tm 상황 피처 +{len(tmc)}개 → 총 {len(features)}개 | "
              f"결측 {train[tmc[0]].isna().mean() * 100:.2f}%")

    if args.pca or args.pca_add:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import QuantileTransformer
        n_comp = args.pca or args.pca_add
        num_cols = [c for c in features if c not in CAT_COLS
                    and train[c].dtype != object]
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=500,
                                 random_state=args.seed)
        Xn = qt.fit_transform(train.loc[~is_val, num_cols].fillna(0))
        pca = PCA(n_components=n_comp, random_state=args.seed).fit(Xn)
        Z = pca.transform(qt.transform(train[num_cols].fillna(0)))
        pcs = [f"pc{i}" for i in range(n_comp)]
        for i, c in enumerate(pcs):
            train[c] = Z[:, i].astype(np.float32)
        evr = pca.explained_variance_ratio_.sum()
        if args.pca:
            features = [c for c in features if c in CAT_COLS] + pcs
            print(f"PCA 대체: 수치 {len(num_cols)}개 → 주성분 {n_comp}개 "
                  f"(설명 분산 {evr:.3f}) | 총 {len(features)}개")
        else:
            features = features + pcs
            print(f"PCA 추가: +{n_comp}개 (설명 분산 {evr:.3f}) → 총 {len(features)}개")

    if args.top_k:
        import joblib as _jl
        base = _jl.load("model/cat_fv2.pkl")
        imp = pd.Series(base["model"].get_feature_importance(),
                        index=base["model"].feature_names_)
        keep = set(imp.nlargest(args.top_k).index)
        features = [c for c in features if c in keep]
        CAT_COLS[:] = [c for c in CAT_COLS if c in features]
        print(f"중요도 상위 {args.top_k}개만 사용 → {len(features)}개 "
              f"(범주형 {len(CAT_COLS)})")

    if args.ptype:
        from pitcher_cluster import assign, fit
        pk = fit(train.loc[~is_val], k=args.ptype, seed=args.seed)
        train["ptype"] = assign(train, pk).astype(str)
        features = features + ["ptype"]
        CAT_COLS.append("ptype")
        print(f"투수 유형 클러스터 K={args.ptype} 추가 → 총 {len(features)}개")

    if args.tm_pctx:
        pc = pd.read_csv(args.tm_pctx)
        train["cnt"] = (train.balls_before.astype(str) + "-"
                        + train.strikes_before.astype(str))
        m = train[["pitcher_id", "cnt"]].merge(pc, on=["pitcher_id", "cnt"],
                                               how="left")
        for c in ["d_fb", "d_sp", "tmp_n"]:
            train[f"pctx_{c}"] = m[c].to_numpy()
        train = train.drop(columns=["cnt"])
        pcols = [f"pctx_{c}" for c in ["d_fb", "d_sp", "tmp_n"]]
        features = features + pcols
        print(f"투수×상황 편차 +{len(pcols)}개 → 총 {len(features)}개 | "
              f"결측 {train[pcols[0]].isna().mean() * 100:.2f}%")

    if args.feat_gap:
        from career_gap import add_gap, build_gap_table
        # 공백 표는 학습 구간으로만 — 검증 시즌 이력은 쓰지 않는다
        tab = build_gap_table(train.loc[~is_val])
        # 검증 시즌 행은 '직전 마지막 등장' 기준으로 별도 산출 (평가와 동일 절차)
        vs = int(args.val_season)
        last = (train.loc[~is_val].groupby("pitcher_id")["season"].max()
                .rename("prev").reset_index())
        last["season"] = vs
        last["gap_years"] = vs - last["prev"]
        tab = pd.concat([tab[tab.season != vs],
                         last[["pitcher_id", "season", "gap_years"]]])
        train, gap_cols = add_gap(train, tab)
        # 2019은 이전 이력이 없어 gap 정의 불가 → 결측 처리
        train.loc[train.season == train.season.min(), gap_cols] = np.nan
        features = features + gap_cols
        print(f"경력 공백 피처 +{len(gap_cols)}개 → 총 {len(features)}개 | "
              f"복귀 {train['is_return'].sum():,}행")

    if args.feat_abs:
        from features import add_abs_regime
        train, abs_cols = add_abs_regime(train)
        features = features + abs_cols
        print(f"ABS 체제 피처 +{len(abs_cols)}개 → 총 {len(features)}개 | "
              f"ABS 비율 {train['is_abs'].mean() * 100:.1f}%")

    if args.feat_mgr:
        import manager_feat as mf
        raw = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                          usecols=["row_id", "season", "game_month", "pitcher_id",
                                   "pitcher_team_id", "batter_team_id"])
        mtab = mf.build_style_table(raw[raw.season < args.val_season])
        mgr_table = mf.build_style_table(raw)      # 제출 refit용
        train, mcols = mf.add_style(train, mtab)
        features = features + mcols
        print(f"감독 운용 피처 +{len(mcols)}개 → 총 {len(features)}개")

    if args.feat_role:
        from pitcher_role import add_role, build_role_table
        # 검증용 표는 학습 구간만, 제출(refit)용 표는 전체 — pkl에 후자를 저장
        rtab = build_role_table(train.loc[~is_val])
        role_table = build_role_table(train)
        train, rcols = add_role(train, rtab)
        features = features + rcols
        print(f"역할 피처 +{len(rcols)}개 → 총 {len(features)}개 | "
              f"결측 {train[rcols[0]].isna().mean() * 100:.2f}%")

    if args.feat_fatigue:
        from rules import add_fatigue_features
        train, fcols = add_fatigue_features(train)
        features = features + fcols
        print(f"체력·심리 피처 +{len(fcols)}개 → 총 {len(features)}개")

    if args.feat_rules:
        from rules import add_rule_features
        train, rule_cols = add_rule_features(train)
        features = features + rule_cols
        print(f"규칙 피처: +{len(rule_cols)}개 → 총 {len(features)}개")

    season_means = None
    if args.feat_v5:
        assert args.feat_v2, "--feat-v5는 --feat-v2와 함께 사용"
        from features import add_season_relative, compute_season_means
        # 시즌 평균은 학습 구간(2019~2023)으로만 산출 — 검증 누수 차단
        season_means = compute_season_means(train.loc[~is_val])
        train, rel_cols = add_season_relative(train, season_means)
        features = features + rel_cols
        print(f"피처 v5(시즌 상대화): +{len(rel_cols)}개 → 총 {len(features)}개")

    if args.feat_v4:
        assert args.feat_v2, "--feat-v4는 --feat-v2와 함께 사용"
        # 주의: 여기서 numpy를 재import하면 main() 스코프에서 np가 지역변수가 되어
        # feat-v4를 안 쓰는 경로가 UnboundLocalError로 죽는다 (모듈 상단 import 사용)
        from features import (add_ctx_features_expanding, build_ctx_stats,
                              build_ctx_stats_expanding)
        train["base_empty"] = (train["base_state"] == "___").astype(np.int8)
        # 시즌 확장: 시즌 S의 통계는 S 이전 데이터로만 (E31 누수 수정)
        per_season = build_ctx_stats_expanding(train)
        ctx_tables = build_ctx_stats(train)  # 제출 추론용 (전체 시즌)
        train, ctx_cols = add_ctx_features_expanding(train, per_season, ctx_tables)
        features = features + ctx_cols
        print(f"피처 v4(ctx expanding): +{len(ctx_cols)}개 → 총 {len(features)}개")
    if args.fill_prev:
        # "정보 없으면 그 시즌 리그 평균" - std 피처와 같은 원칙.
        pv = [c for c in train.columns if "_prev" in c and c.endswith("_rate")]
        nb = train[pv].isna().mean().max()
        for c in pv:
            m = train.groupby("season")[c].transform("mean")
            train[c] = train[c].fillna(m)
        print(f"prev 결측 채움: {nb * 100:.2f}% -> "
              f"{train[pv].isna().mean().max() * 100:.2f}% ({len(pv)}개 열)")
    if args.drop_cols:
        rm = {c.strip() for c in args.drop_cols.split(",") if c.strip()}
        before = len(features)
        features = [c for c in features if c not in rm]
        print(f"열 제거 {before - len(features)}개 -> 총 {len(features)}개 "
              f"({sorted(rm & set(train.columns))})")

    if args.dump_npz:
        # NN 학습용으로 **CatBoost가 쓰는 것과 동일한 행렬**을 그대로 내보낸다.
        # 피처 파이프라인을 NN 쪽에 복제하면 반드시 어긋나므로 이음매를 여기 둔다.
        num = [c for c in features if c not in CAT_COLS]
        Xn = train[num].to_numpy(np.float32)
        codes, vocab = [], {}
        for c in CAT_COLS:
            ss = train[c].astype(str)
            # 범주 사전도 적합 행만으로 만든다. 검증/test 행 전체를 훑어 새 범주를
            # 미리 등록하면 값 자체는 타깃이 아니어도 행 독립 감사의 경계가 흐려진다.
            # 미관측 값은 전부 하나의 unknown 코드로 보낸다.
            cats = sorted(ss.loc[is_fit].unique())
            vocab[c] = cats
            unk = len(cats)
            codes.append(ss.map({v: i for i, v in enumerate(cats)})
                         .fillna(unk).to_numpy(np.int32))
        Xc = (np.stack(codes, 1) if codes
              else np.zeros((len(train), 0), np.int32))
        os.makedirs(os.path.dirname(args.dump_npz) or ".", exist_ok=True)
        # 복원한 실패모드 라벨을 **보조 타깃**으로 함께 내보낸다 (E124).
        # 한 행에 이진 타깃 1개가 아니라 상관된 라벨 4개가 있다는 게 이 대회에서
        # 우리가 가진 유일한 '남은 감독 신호'다. NN 은 공유 트렁크 + 다중 헤드로
        # 그걸 직접 쓸 수 있다 (GBDT 는 셀 다중분류로만 간접 사용).
        # ⚠️ train 행끼리만 만든다. 추론 경로는 이 라벨을 절대 안 본다.
        import failmode as fm
        lab = fm._pitch_labels(train)
        aux = np.stack([lab[m].to_numpy(np.float32) for m in fm.MODES], 1)
        cell, cell_names, cell_success = fm.build_cells(train)
        is_test = train.get("_is_test", pd.Series(False, index=train.index))
        print(f"보조 라벨 {aux.shape} | 복원률 "
              f"{np.isfinite(aux).all(1).mean() * 100:.2f}%")
        np.savez(args.dump_npz, Xn=Xn, Xc=Xc,
                 y=train[TARGET].to_numpy(np.float32),
                 aux=aux, aux_names=np.array(fm.MODES),
                 cell=cell.to_numpy(np.int16),
                 cell_names=np.array(cell_names),
                 cell_success=np.array(sorted(cell_success), np.int16),
                 is_val=is_val.to_numpy(), is_test=is_test.to_numpy())
        joblib.dump({"num": num, "cat_cols": CAT_COLS, "vocab": vocab,
                     "features": features, "fpipe": art},
                    args.dump_npz.replace(".npz", "_meta.pkl"), compress=3)
        print(f"dumped num{Xn.shape} cat{Xc.shape} -> {args.dump_npz}")
        return

    if args.feat_frac < 1.0:
        # 시드마다 다른 부분집합이어야 멤버끼리 달라진다. 범주형은 남긴다
        rs = np.random.default_rng(args.seed)
        num = [c for c in features if c not in CAT_COLS]
        keep = set(rs.choice(num, max(int(len(num) * args.feat_frac), 5),
                             replace=False).tolist())
        features = [c for c in features if c in CAT_COLS or c in keep]
        print(f"피처 부분공간 {args.feat_frac}: 수치 {len(num)} -> "
              f"{len(keep)}개 | 총 {len(features)}개")

    if args.row_filter:
        # E166 세그먼트 모델. **피처를 다 만든 뒤에** 거른다 — 먼저 거르면 TE·std·
        # skill 표까지 그 세그먼트로만 만들어져 가설이 둘 섞인다. 여기서 보려는 건
        # "이 구간은 자기 함수가 필요한가" 하나다.
        _m = train.eval(args.row_filter).to_numpy()
        print(f"행 필터 [{args.row_filter}]: {len(train):,} → {int(_m.sum()):,}행")
        train = train[_m].reset_index(drop=True)
        if args.test_season:
            train["_is_test"] = (train["season"] == args.test_season)
        is_val = train["season"] == args.val_season

    if args.soft_target:
        # 교사 OOF 확률을 row_id 로 붙인다. **검증 라벨은 안 건드린다** —
        # run_cat 이 학습 타깃으로만 쓰고 평가는 TARGET 으로 한다.
        _z = np.load(args.soft_target, allow_pickle=True)
        _m = pd.Series(_z["prob"].astype(np.float64), index=_z["row_id"])
        train["_soft"] = train["row_id"].map(_m)
        _miss = train["_soft"].isna()
        if _miss.any():          # 교사가 못 만든 행은 원래 라벨로 되돌린다
            train.loc[_miss, "_soft"] = train.loc[_miss, TARGET].astype(float)
        print(f"증류 타깃: {args.soft_target} | 결측 {_miss.mean() * 100:.2f}% | "
              f"평균 {train['_soft'].mean():.4f} (실제 {train[TARGET].mean():.4f}) "
              f"| sd {train['_soft'].std():.4f}")

    if args.test_season:
        # 피처가 다 붙은 뒤에 분리한다 — 이제 test_df 도 같은 컬럼을 갖는다
        m = train["_is_test"].to_numpy()
        test_df = train[m].reset_index(drop=True)
        train = train[~m].reset_index(drop=True)
        is_val = train["season"] == args.val_season
        print(f"test 시즌 {args.test_season} 분리: 학습·검증 {len(train):,}행 / "
              f"미학습 {len(test_df):,}행")

    if args.val_league:
        # 검증셋에서만 리그를 제한한다. 빠진 행은 **학습으로 가지 않는다** —
        # is_fit 이 이미 확정돼 있으므로 여기서 is_val 만 좁히면 그 행들은
        # 어디에도 안 쓰인다. 학습 데이터는 제출과 동일하게 유지된다.
        keep = is_val & (train["game_type"] == args.val_league)
        print(f"검증셋 리그 제한 {args.val_league}: "
              f"{int(is_val.sum()):,} → {int(keep.sum()):,}행 "
              f"(학습 데이터는 그대로)")
        is_val = keep

    y_va = train.loc[is_val, TARGET].to_numpy(np.float64)
    os.makedirs("./out", exist_ok=True)
    os.makedirs("./model", exist_ok=True)   # 원격 머신에는 없을 수 있다

    # 시드 스윕을 **한 프로세스 안에서** 돈다. 피처는 시드와 무관하게 동일한데
    # 매 런마다 CSV 재읽기(368MB) + std/TE/skill 재구축에 약 32초를 쓰고 있었다.
    # (로그의 초 표기는 학습 시간만이라 이 비용이 안 보였다)
    seeds = ([int(x) for x in args.seeds.split(",") if x.strip()]
             if args.seeds else [args.seed])
    base_tag = args.tag
    for _si, _sd in enumerate(seeds):
        args.seed = _sd
        args.tag = base_tag if len(seeds) == 1 else f"{base_tag}_s{_sd}"
        t0 = time.time()
        runner = {"cat": run_cat, "lgb": run_lgb, "rank": run_rank}.get(
            args.model, run_xgb)
        model, p, best_iter = runner(args, train, features, is_val)
        score = bss(y_va, p)
        print(f"[{args.model} {args.tag}] val{args.val_season} BSS {score:.2f} "
              f"| best_iter={best_iter} | {time.time() - t0:.0f}s", flush=True)
        # 조기종료가 즉시 걸리면 학습·검증이 **체제 변화를 가로지른** 것이다.
        # (F리그 라벨이 2022 0.71 -> 2023 0.47 로 바뀌어 val 2023 에서 두 번 당했다.
        #  --drop-f-pre 2022 로 고친다.) 조용히 지나가면 쓰레기 수치를 믿게 된다.
        if best_iter is not None and best_iter < 50 and args.es >= 100:
            print(f"  !! best_iter={best_iter} — 조기종료가 즉시 걸렸다. "
                  f"학습/검증 구간에 체제 변화가 끼었는지 확인할 것 "
                  f"(--drop-f-pre). 이 실행의 수치는 믿지 말 것.", flush=True)

        if args.test_season:
            # 학습·검증 어디에도 안 쓴 시즌으로 진짜 1년 앞 성능 측정
            Xt = test_df[features].copy()
            # 문자열 캐스팅은 **CatBoost 전용**이다. LightGBM 은 학습 때 category
            # dtype 을 쓰므로 여기서 str 로 바꾸면 예측이 터진다(DV_lgb 가 시드 1개
            # 뒤에 조용히 죽은 원인).
            if args.model == "cat":
                for c in CAT_COLS:
                    Xt[c] = Xt[c].astype(str)
            elif args.model == "lgb":
                # 범주 목록을 **학습 프레임에서** 가져와야 한다. test 프레임만
                # 보고 astype("category") 하면 등장 값이 달라 코드가 밀리는데,
                # LightGBM 은 코드로 분기하므로 조용히 틀린 예측이 나온다.
                for c in CAT_COLS:
                    Xt[c] = pd.Categorical(
                        Xt[c], categories=train[c].astype("category").cat.categories)
            # 셀 다중분류는 열이 14개다. [:,1] 을 집으면 '성공 확률'이 아니라
            # 셀 하나의 확률이라 완전히 다른 값이 나온다 (DV_cell 이 BSS -1367,
            # rms 0.37 로 나온 원인). 검증 경로처럼 성공 셀을 합산해야 한다.
            _succ = getattr(model, "_fm_success", None)
            _cell_proba = None
            if _succ is not None:
                _cell_proba = model.predict_proba(Xt)
                pt = _cell_proba[:, _succ].sum(axis=1)
            elif getattr(model, "_rank_calib", None) is not None:
                pt = _rank_probability(model.predict(Xt), model._rank_calib)
            elif getattr(model, "_fm_multilabel", False):
                pt = model.predict_proba(Xt)[:, 0]
            elif getattr(model, "_baseline_col", None):
                from catboost import Pool
                _bc = model._baseline_col
                _qt = np.clip(test_df[_bc].astype(float).fillna(0.5),
                              1e-4, 1 - 1e-4)
                _tp = Pool(Xt, cat_features=CAT_COLS)
                _tp.set_baseline(np.log(_qt / (1 - _qt)).to_numpy())
                pt = model.predict_proba(_tp)[:, 1]
            elif hasattr(model, "predict_proba"):
                pt = model.predict_proba(Xt)[:, 1]
            else:
                pt = np.clip(model.predict(Xt), 0, 1)
            yt = test_df[TARGET].to_numpy(np.float64)
            test_bss = bss(yt, pt)
            print(f"  → 미학습 {args.test_season} 시즌 BSS {test_bss:.2f} "
                  f"(예측평균 {pt.mean():.4f} vs 실제 {yt.mean():.4f})")
            # 산출물이 망가졌는지 **여기서** 잡는다. DV_cell 이 BSS -1367 / 편향
            # 0.37 로 나왔는데(다중분류 열을 잘못 집었다) 조용히 지나가서 그 위에
            # "다양성 경로 종료" 판정을 쌓았다.
            if test_bss < 0 or abs(pt.mean() - yt.mean()) > 0.05:
                print(f"  !!! 산출물 이상 — BSS {test_bss:.1f}, 편향 "
                      f"{pt.mean() - yt.mean():+.4f}. 예측 추출 경로를 의심할 것 "
                      f"(다중분류/다중라벨/범주 정렬). 이 수치로 판정하지 말 것.",
                      flush=True)
            # row_id 를 같이 남긴다 — 세그먼트별 드리프트를 보려면 예측을 원본
            # 행에 정확히 되붙여야 한다 (순서 가정은 조용히 틀어진다)
            _test_payload = {
                "y": yt, "pred": pt,
                "row_id": test_df["row_id"].to_numpy(),
            }
            if args.dump_cell_proba and _cell_proba is not None:
                _test_payload["cell_proba"] = _cell_proba.astype(np.float32)
                _test_payload["cell_success"] = np.asarray(_succ, dtype=np.int16)
            np.savez_compressed(f"./out/{args.model}_{args.tag}_test_preds.npz",
                                **_test_payload)

        np.savez_compressed(f"./out/{args.model}_{args.tag}_val_preds.npz",
                            y=y_va, pred=p)
        # ── 원장 자동 기록 ────────────────────────────────────────────────
        # 08-07 에 30개 실험을 돌리고 docs/EXPERIMENTS_LOG.md 를 하나도 안 갱신했다.
        # 그래서 E08 에서 -580 으로 끝난 season 제거를 "안 해본 축"이라 부르며
        # 다시 큐에 걸었다. 사람이 적는 단계를 없앤다 — tools/precheck.py 가 읽는다.
        try:
            import socket
            with open("./LEDGER.tsv", "a", encoding="utf-8") as _lg:
                _lg.write("\t".join([
                    time.strftime("%Y-%m-%d %H:%M"), socket.gethostname(),
                    args.tag, args.model, str(args.val_season),
                    str(args.test_season or ""), str(_sd), str(best_iter),
                    f"{score:.2f}",
                    f"{locals().get('test_bss', float('nan')):.2f}",
                    " ".join(sys.argv[1:])]) + "\n")
        except Exception as _e:                       # 기록 실패로 학습을 죽이지 않는다
            print(f"  (LEDGER 기록 실패: {_e})")
        # 피처 재현에 필요한 건 전부 art 안에 있다 (fpipe.transform 이 읽는다).
        joblib.dump({"model": model, "features": features, "cat_cols": CAT_COLS,
                     "best_iteration": best_iter, "val_bss": score,
                     "fpipe": art, "resid_col": args.resid_col,
                     "baseline_col": args.baseline_col,
                     "fm_success": getattr(model, "_fm_success", None),
                     "season_means": season_means},
                    f"./model/{args.model}_{args.tag}.pkl", compress=3)
        print(f"saved: model/{args.model}_{args.tag}.pkl")


if __name__ == "__main__":
    main()
