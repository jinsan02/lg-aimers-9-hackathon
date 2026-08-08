"""피처 파이프라인 — 학습과 추론이 **이 파일 하나만** 본다.

지금까지 학습(train_gbdt2.main)과 추론(script_blend_v6.predict_one)이 각자
순서를 손으로 적고 있었다. 그래서 두 번 깨졌다.

  - skill 을 학습에선 TE 뒤, 추론에선 TE 앞에 둬서 설계행렬 폭이 41 vs 44
  - 설정을 `std_anchors` **16칸 위치 튜플**로 넘겨서 `st[13]` 이 무엇인지
    아무도 모르고, 레버가 하나 늘 때마다 추론 쪽에 `len(st) > N` 가드가 붙었다

그래서 `fit()` 과 `transform()` 을 같은 파일에 나란히 두고, 아티팩트는
**이름 있는 dict** 으로 저장한다. 순서가 어긋나면 눈으로 바로 보인다.

  학습:  train, cols, cats, art = fpipe.fit(train, args, is_fit)
  추론:  test = fpipe.transform(test, art)

옛 pkl(위치 튜플)도 `artifact_of()` 가 dict 으로 바꿔 주므로 그대로 로드된다.
"""

import numpy as np
import pandas as pd

# ── 파이프라인 순서 (fit/transform 이 공유하는 유일한 정의) ─────────────
#   0. tm 요약표 병합      (정적 표. 뒤 단계가 참조하지 않음)
#   1. 기본 파생 피처 v2/v3
#   2. 시즌내 복원(std) → 궤적(profile) → 폼 → 도메인 교차 → 창분해 → 카운트
#   3. 타깃 인코딩(TE) → 교차항(cross)
#   4. 실력 추정(skill)   ← TE 산출 컬럼을 입력으로 쓰므로 **반드시 TE 뒤**
STEP_ORDER = ("tm", "v2", "std", "te", "skill")

TM_PREFIX = ("tm_", "tmx_", "sct_")
TM_KEYS = ("pitcher_id", "season", "balls_before", "strikes_before")


# ────────────────────────────── tm 요약표 ──────────────────────────────

def merge_tm(df, tm_table, verbose=False):
    """구종/구속 요약표를 병합하고, 링키지 실패분을 리그 프로필로 채운다.

    결측을 그대로 두면 '표본 적은 투수' 표지로 오용된다. 학습에서 채웠으면
    추론에서도 **같은 방식으로** 채워야 한다 (예전엔 추론 쪽이 안 채웠다).
    """
    if tm_table is None:
        return df, []
    keys = [c for c in TM_KEYS if c in tm_table.columns]
    cols = [c for c in tm_table.columns if c.startswith(TM_PREFIX)]
    df = df.merge(tm_table, on=keys, how="left")
    gk = [c for c in ("balls_before", "strikes_before") if c in tm_table.columns]
    if gk:
        lg = tm_table.groupby(gk)[cols].mean()
        idx = (pd.MultiIndex.from_arrays([df[c] for c in gk]) if len(gk) > 1
               else df[gk[0]].to_numpy())
        fill = lg.reindex(idx)
        fill.index = df.index
    else:
        fill = pd.DataFrame({c: tm_table[c].mean() for c in cols}, index=df.index)
    before = df[cols[0]].isna().mean()
    df[cols] = df[cols].fillna(fill)
    if verbose:
        print(f"tm 피처 {len(cols)}개 병합 (키 {keys}) | 결측 "
              f"{before * 100:.1f}% -> {df[cols[0]].isna().mean() * 100:.2f}% "
              f"(리그 프로필로 채움)")
    return df, cols


# ──────────────────────────────── fit ──────────────────────────────────

def fit(train, args, is_fit, tm_table=None, verbose=True):
    """학습 데이터에 피처를 붙이고, 추론에 필요한 아티팩트를 만든다.

    is_fit  타깃을 쓰는 표를 적합할 때 쓸 행 마스크 (검증·test 시즌 제외).
    반환    (train, 추가된 컬럼, 새 범주형 컬럼, 아티팩트 dict)
    """
    say = print if verbose else (lambda *a, **k: None)
    new_cols, new_cats = [], []
    art = {"steps": STEP_ORDER, "tm_table": tm_table}

    # 0. tm — 정적 표라 가장 먼저. (뒤 단계가 tm_* 를 읽지 않으므로 위치 무관)
    if tm_table is not None:
        train, cols = merge_tm(train, tm_table, verbose=verbose)
        new_cols += cols

    # 1. 기본 파생 피처
    if args.feat_v2:
        from features import NEW_CAT, add_features, compute_priors
        art["priors"] = compute_priors(train.loc[is_fit])
        train, cols = add_features(train, art["priors"], k=args.feat_k)
        new_cols += cols
        new_cats += [c for c in NEW_CAT if c in cols]
        say(f"피처 v2/v3: +{len(cols)}개")

    # 2. 시즌내 복원 계열
    if args.feat_std:
        import season_std as ss
        # E99: asof_* 는 통산 누적이라 직전 시즌 말을 빼면 당해 시즌이 복원된다.
        pri = {c: float(train[c].mean()) for c in train.columns
               if c.startswith("asof_")}
        k_by = {k: v for k, v in (("pitchmix", args.std_k_mix),
                                  ("batter", args.std_k_bat)) if v > 0} or None
        std = {
            "anchors": ss.build_anchors(train),
            "priors": pri,
            "k": args.std_k,
            "to_career": not args.std_to_prior,
            "multi_k": (tuple(float(x) for x in args.std_multi_k.split(","))
                        if args.std_multi_k else ()),
            "season_prior": ss.season_priors(train) if args.std_season_prior else None,
            "excess": args.std_excess,
            "ratio": args.std_ratio,
            "k_by": k_by,
        }
        art["std"] = std
        train, cols = _apply_std(train, std)

        if args.feat_prof:
            # E100: 앵커를 연속 차분해 선수별 시즌 궤적을 만든다.
            art["profile"] = {"table": ss.build_profile(train), "k": args.prof_k,
                              "lags": tuple(range(1, args.prof_lags + 1))}
            train, c = _apply_profile(train, art["profile"], pri)
            cols += c
        for flag, name, fn in (("feat_form", "form", ss.add_form),
                               ("feat_domain", "domain", ss.add_domain),
                               ("feat_window", "window", ss.add_window),
                               ("feat_count", "count", ss.add_count_style)):
            art[name] = bool(getattr(args, flag))
            if art[name]:
                train, c = fn(train)
                cols += c
        new_cols += cols
        say(f"시즌내 복원 계열: +{len(cols)}개 | 결측률 "
            f"{train[cols[1]].isna().mean() * 100:.1f}%")

    # 3. 타깃 인코딩
    if args.te:
        import target_enc as te_mod
        train = te_mod.add_inning_bucket(train)
        tables = []
        # E161: 축마다 필요한 수축이 다르다. 신뢰도(적률법/반분법)로 재보면
        #   투수 91/381 · 투수x카운트 112/209 · 투수x타자손 75/206 · **타자 556/2414**
        # 인데 우리는 전 축에 50 을 쓴다. 타자 축은 6~11배 덜 수축시키고 있다 —
        # 제구는 투수의 일이라 타자 정체성이 실어주는 신호가 거의 없다는 뜻이다.
        # `--te-k 50` 처럼 스칼라도 되고 `--te-k b:500,pc:110,*:50` 처럼 축별도 된다.
        _kmap, _kdef = {}, args.te_k
        if isinstance(args.te_k, str) and ":" in args.te_k:
            for part in args.te_k.split(","):
                nm, _, v = part.partition(":")
                if nm.strip() == "*":
                    _kdef = float(v)
                else:
                    _kmap[nm.strip()] = float(v)
        else:
            _kdef = float(args.te_k)
        for spec in args.te.split(","):
            spec = spec.strip()
            keys = te_mod.SPECS[spec]
            tables.append((keys, te_mod.build_te(
                train, keys, k=_kmap.get(spec, _kdef),
                half_life=args.te_halflife,
                strat=not args.te_flat)))
        art["te"] = {"tables": tables, "dev": args.te_dev,
                     "cross": bool(args.feat_cross and args.feat_std)}
        train, cols = _apply_te(train, art["te"])
        new_cols += cols
        say(f"TE({args.te}) k={args.te_k} hl={args.te_halflife}: +{len(cols)}개 "
            f"| 결측률 {train[cols[0]].isna().mean() * 100:.1f}%")

    # 4. 실력 추정 — TE 뒤 (te_pitcher_* 컬럼을 회귀 입력으로 읽는다)
    axes = skill_axes(args)
    if axes:
        import skill as sk_mod
        art["skill_packs"] = [sk_mod.build(train, axis=a) for a in axes]
        cols = []
        for pk in art["skill_packs"]:
            train, c = sk_mod.add(train, pk)
            cols += c
        new_cols += cols
        say(f"실력 추정({','.join(a or 'base' for a in axes)}): +{len(cols)}개")

    return train, new_cols, new_cats, art


def skill_axes(args):
    """어떤 축의 실력 추정기를 붙일지. 옛 불리언 플래그도 받아 준다."""
    if getattr(args, "skill_axes", ""):
        return [a.strip() for a in args.skill_axes.split(",") if a.strip()]
    return ([""] if args.feat_skill else []) + \
           (["count"] if args.feat_skill_pc else [])


# ───────────────────────────── transform ───────────────────────────────

def transform(df, art):
    """추론용. `fit()` 과 **같은 순서**로 같은 변환을 건다."""
    art = artifact_of(art)
    if art.get("tm_table") is not None:
        df, _ = merge_tm(df, art["tm_table"])
    if art.get("priors") is not None:
        from features import add_features
        df, _ = add_features(df, art["priors"])
    if art.get("std") is not None:
        df, _ = _apply_std(df, art["std"])
        if art.get("profile") is not None:
            df, _ = _apply_profile(df, art["profile"], art["std"]["priors"])
        import season_std as ss
        for name, fn in (("form", ss.add_form), ("domain", ss.add_domain),
                         ("window", ss.add_window), ("count", ss.add_count_style)):
            if art.get(name):
                df, _ = fn(df)
    if art.get("te") is not None:
        df, _ = _apply_te(df, art["te"])
    if art.get("skill_packs"):
        import skill as sk_mod
        for pk in art["skill_packs"]:
            df, _ = sk_mod.add(df, pk)
    return df


# ── fit/transform 이 공유하는 실제 호출부 (인자 불일치를 구조적으로 막는다) ──

def _apply_std(df, std):
    import season_std as ss
    return ss.add_std(df, std["anchors"], k=std["k"], priors=std["priors"],
                      to_career=std["to_career"], multi_k=std["multi_k"],
                      season_prior=std["season_prior"], excess=std["excess"],
                      ratio=std["ratio"], k_by=std["k_by"])


def _apply_profile(df, prof, priors):
    import season_std as ss
    return ss.add_profile(df, prof["table"], k=prof["k"], priors=priors,
                          lags=prof["lags"])


def _apply_te(df, te):
    import season_std as ss
    import target_enc as te_mod
    df = te_mod.add_inning_bucket(df)
    df, cols = te_mod.add_te(df, te["tables"], dev=te["dev"])
    if te["cross"]:
        df, xcols = ss.add_cross(df)
        cols = cols + xcols
    return df, cols


# ─────────────────────── 옛 pkl 호환 (위치 튜플) ────────────────────────

# std_anchors 튜플에 값이 들어간 순서. 레버가 늘 때마다 뒤에 붙었다.
_LEGACY_STD = ("anchors", "priors", "k", "_prof_table", "_prof_k", "_prof_lags",
               "to_career", "multi_k", "season_prior", "excess", "_form",
               "ratio", "_domain", "k_by", "_count", "_window")


def artifact_of(pack):
    """dict 이면 그대로, 옛 pkl(위치 튜플)이면 dict 으로 바꿔 돌려준다."""
    if pack.get("fpipe") is not None:
        return pack["fpipe"]
    if "steps" in pack:
        return pack
    st = pack.get("std_anchors")
    art = {"tm_table": pack.get("tm_table"),
           "priors": pack.get("priors") if pack.get("feat_v2") else None}
    if st:
        g = dict(zip(_LEGACY_STD, st))          # 짧은 튜플은 뒤가 그냥 없다
        art["std"] = {"anchors": g["anchors"], "priors": g["priors"],
                      "k": g["k"], "to_career": g.get("to_career", False),
                      "multi_k": g.get("multi_k", ()),
                      "season_prior": g.get("season_prior"),
                      "excess": g.get("excess", False),
                      "ratio": g.get("ratio", False),
                      "k_by": g.get("k_by")}
        if g.get("_prof_table") is not None:
            art["profile"] = {"table": g["_prof_table"], "k": g["_prof_k"],
                              "lags": tuple(range(1, g["_prof_lags"] + 1))}
        for name in ("form", "domain", "window", "count"):
            art[name] = bool(g.get("_" + name))
    if pack.get("te_tables"):
        art["te"] = {"tables": pack["te_tables"], "dev": pack.get("te_dev"),
                     "cross": bool(st)}
    sp = pack.get("skill_pack")
    art["skill_packs"] = [sp] if sp is not None else []
    return art


def design(df, pack):
    """모델이 먹을 수 있는 형태로 만든다 (범주형은 문자열)."""
    X = df[pack["features"]].copy()
    for c in pack["cat_cols"]:
        X[c] = X[c].astype(str)
    return X


def predict(pack, test):
    """아티팩트 하나로 예측까지."""
    src = transform(test, pack)
    model = pack["model"]
    if type(model).__module__.startswith("xgboost"):
        import xgboost as xgb
        X = src[pack["features"]].copy()
        for c in pack["cat_cols"]:
            X[c] = X[c].astype("category")
        return np.asarray(model.predict(xgb.DMatrix(X, enable_categorical=True)))
    proba = model.predict_proba(design(src, pack))
    if pack.get("fm_success"):
        # E124: 실패모드 셀 다중분류. P(성공) = 성공 비트를 가진 셀들의 합.
        # 셀에 타깃 비트를 넣었으므로 이 합산은 근사가 아니라 정확하다.
        return proba[:, pack["fm_success"]].sum(axis=1)
    return proba[:, 1]
