"""Gate a legal pitch-type mixture before building a full multitask model.

The 2023 validation residual of the current local baseline is grouped by the
high-confidence current pitch type recovered from the provided Trackman data.
A row-local pitch classifier trained on Trackman <=2023 then supplies
``P(type | pre-pitch x)`` for 2024.  The only deployable correction is the
expectation over those probabilities; actual 2024 pitch type is used solely as
an oracle diagnostic.

This is deliberately a low-capacity transfer gate.  If even the expected
three-type residual cannot improve the held-out 2024 Brier score, a larger
shared-encoder multitask network has no demonstrated route to the target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


DATA = Path("data")
OUT = Path("out")
TYPES = ["fastball", "breaking", "offspeed"]
CONTEXT = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "balls_before", "strikes_before", "outs_before", "pitcher_hand",
    "batter_hand",
]
KEY = CONTEXT + ["pitcher_id", "batter_id"]
PITCH_FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "balls_before", "strikes_before", "outs_before", "pitcher_id",
    "batter_id", "pitcher_hand", "batter_hand",
]
PITCH_CATS = [
    "game_dayofweek", "top_bottom", "pitcher_id", "batter_id",
    "pitcher_hand", "batter_hand",
]


def bss(y: np.ndarray, p: np.ndarray) -> float:
    r = float(np.mean(y))
    return float(max(0.0, 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))))


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    # The second pitch head below deliberately receives every competition
    # input column, so load the full table once rather than re-reading it.
    main = pd.read_csv(DATA / "train.csv")
    tm = pd.read_csv(
        DATA / "trackman_history.csv", encoding="utf-8-sig",
        usecols=CONTEXT + ["pitcher_trackman_id", "batter_trackman_id",
                           "pitch_type_group"],
    )
    pm = pd.read_csv(DATA / "processed/pitcher_map2.csv")
    bm = pd.read_csv(DATA / "processed/batter_map2.csv")
    pmap = pm.drop_duplicates("tm_id").set_index("tm_id")["pitcher_id"]
    bmap = bm.drop_duplicates("tm_batter_id").set_index("tm_batter_id")["batter_id"]
    tm["pitcher_id"] = tm.pitcher_trackman_id.map(pmap)
    tm["batter_id"] = tm.batter_trackman_id.map(bmap)
    tm["top_bottom"] = tm.top_bottom.map({"Top": "T", "Bottom": "B"})
    for col in ["pitcher_hand", "batter_hand"]:
        tm[col] = tm[col].map({"Left": 1, "Right": 2})
    tm = tm[tm.pitcher_id.notna() & tm.batter_id.notna()].copy()
    tm["pitcher_id"] = tm.pitcher_id.astype(np.int64)
    tm["batter_id"] = tm.batter_id.astype(np.int64)
    return main, tm


def exact_labels(main: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    """Return only one-main-row/one-Trackman-row groups with a 3-class label."""
    tr_n = main.groupby(KEY, dropna=False).size().rename("tr_n").reset_index()
    tm_one = tm.groupby(KEY, dropna=False).agg(
        tm_n=("pitch_type_group", "size"),
        pitch_type_group=("pitch_type_group", "first"),
    ).reset_index()
    keys = tr_n[tr_n.tr_n == 1].drop(columns="tr_n").merge(
        tm_one[(tm_one.tm_n == 1) & tm_one.pitch_type_group.isin(TYPES)]
        .drop(columns="tm_n"),
        on=KEY, how="inner",
    )
    return main[["row_id"] + KEY].merge(keys, on=KEY, how="inner")[[
        "row_id", "pitch_type_group",
    ]]


def fit_pitch_head(tm: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, CatBoostClassifier]:
    tr = tm[(tm.season <= 2023) & tm.pitch_type_group.isin(TYPES)].copy()
    xtr = tr[PITCH_FEATURES].copy()
    xva = target[PITCH_FEATURES].copy()
    for col in PITCH_CATS:
        xtr[col] = xtr[col].astype(str)
        xva[col] = xva[col].astype(str)
    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=500, depth=7,
        learning_rate=0.08, l2_leaf_reg=10, random_seed=42,
        task_type="GPU", devices="0", max_ctr_complexity=1,
        allow_writing_files=False, verbose=100,
    )
    model.fit(Pool(xtr, tr.pitch_type_group.astype(str), cat_features=PITCH_CATS))
    raw = model.predict_proba(xva)
    order = [list(model.classes_).index(c) for c in TYPES]
    return raw[:, order], model


def fit_main_pitch_head(main: pd.DataFrame,
                        target: pd.DataFrame) -> tuple[np.ndarray, CatBoostClassifier]:
    """Strongest legal head: all contest inputs, trained on exact joined types."""
    features = [c for c in pd.read_csv(DATA / "test.csv", nrows=0).columns
                if c != "row_id"]
    cats = [c for c in [
        "game_dayofweek", "top_bottom", "game_type", "base_state",
        "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
        "pitcher_team_id", "batter_team_id",
    ] if c in features]
    tr = main[(main.season <= 2023) & main.pitch_type_group.isin(TYPES)].copy()
    xtr, xva = tr[features].copy(), target[features].copy()
    for col in cats:
        xtr[col] = xtr[col].astype(str)
        xva[col] = xva[col].astype(str)
    model = CatBoostClassifier(
        loss_function="MultiClass", iterations=500, depth=7,
        learning_rate=0.08, l2_leaf_reg=10, random_seed=42,
        task_type="GPU", devices="0", max_ctr_complexity=1,
        allow_writing_files=False, verbose=100,
    )
    model.fit(Pool(xtr, tr.pitch_type_group.astype(str), cat_features=cats))
    raw = model.predict_proba(xva)
    order = [list(model.classes_).index(c) for c in TYPES]
    return raw[:, order], model


def residual_table(src: pd.DataFrame, group: list[str], k: float = 500.0) -> pd.Series:
    src = src.copy()
    src["resid"] = src.y - src.pred
    g = src.groupby(group, dropna=False).resid.agg(["sum", "count"])
    delta = g["sum"] / (g["count"] + k)
    # A pitch-type arm must add resolution, not re-estimate the global shift.
    joined = src[group].merge(delta.rename("d").reset_index(), on=group,
                              how="left")["d"].to_numpy()
    return delta - float(np.mean(joined))


def expected_delta(target: pd.DataFrame, proba: np.ndarray,
                   table: pd.Series, by_count: bool) -> np.ndarray:
    out = np.zeros(len(target), dtype=np.float64)
    if by_count:
        for j, typ in enumerate(TYPES):
            idx = pd.MultiIndex.from_arrays([
                np.full(len(target), typ, dtype=object),
                target.balls_before.to_numpy(), target.strikes_before.to_numpy(),
            ], names=table.index.names)
            out += proba[:, j] * table.reindex(idx, fill_value=0.0).to_numpy()
    else:
        out = proba @ table.reindex(TYPES, fill_value=0.0).to_numpy()
    return out


def oracle_delta(target: pd.DataFrame, table: pd.Series,
                 by_count: bool) -> np.ndarray:
    if by_count:
        idx = pd.MultiIndex.from_frame(target[[
            "pitch_type_group", "balls_before", "strikes_before",
        ]])
        return table.reindex(idx, fill_value=0.0).to_numpy()
    return target.pitch_type_group.map(table).fillna(0.0).to_numpy()


def main() -> None:
    main_df, tm = load_tables()
    labels = exact_labels(main_df, tm)
    main_df = main_df.merge(labels, on="row_id", how="left")

    val = np.load(OUT / "cat_MVA_native_val_preds.npz", allow_pickle=True)
    tst = np.load(OUT / "cat_MVA_native_test_preds.npz", allow_pickle=True)
    src = main_df[main_df.season == 2023].copy().reset_index(drop=True)
    assert len(src) == len(val["y"])
    assert np.array_equal(src.control_success.to_numpy(), val["y"])
    src["y"], src["pred"] = val["y"], val["pred"]

    tgt = pd.DataFrame({"row_id": tst["row_id"], "y": tst["y"],
                        "pred": tst["pred"]})
    tgt = tgt.merge(main_df[main_df.season == 2024], on="row_id", how="left",
                    validate="one_to_one")
    assert tgt.season.notna().all()

    proba, pitch_model = fit_pitch_head(tm, tgt)
    main_proba, main_pitch_model = fit_main_pitch_head(main_df, tgt)
    known = tgt.pitch_type_group.notna().to_numpy()
    true_idx = tgt.loc[known, "pitch_type_group"].map(
        {t: i for i, t in enumerate(TYPES)}).to_numpy()
    pitch_acc = float(np.mean(np.argmax(proba[known], axis=1) == true_idx))
    pitch_ll = float(-np.log(np.clip(proba[known, true_idx], 1e-15, 1)).mean())
    main_pitch_acc = float(np.mean(
        np.argmax(main_proba[known], axis=1) == true_idx))
    main_pitch_ll = float(-np.log(np.clip(
        main_proba[known, true_idx], 1e-15, 1)).mean())

    report = {
        "baseline_bss_2024": bss(tgt.y.to_numpy(), tgt.pred.to_numpy()),
        "exact_label_coverage_2023": float(src.pitch_type_group.notna().mean()),
        "exact_label_coverage_2024": float(known.mean()),
        "pitch_head_exact_match_accuracy_2024": pitch_acc,
        "pitch_head_exact_match_logloss_2024": pitch_ll,
        "main_pitch_head_exact_match_accuracy_2024": main_pitch_acc,
        "main_pitch_head_exact_match_logloss_2024": main_pitch_ll,
        "arms": {},
    }
    src_known = src[src.pitch_type_group.notna()].copy()
    for name, group in [
        ("type", ["pitch_type_group"]),
        ("type_count", ["pitch_type_group", "balls_before", "strikes_before"]),
    ]:
        tab = residual_table(src_known, group)
        exp_d = expected_delta(tgt, proba, tab, by_count=name == "type_count")
        main_exp_d = expected_delta(tgt, main_proba, tab,
                                    by_count=name == "type_count")
        ora_d = oracle_delta(tgt, tab, by_count=name == "type_count")
        base = tgt.pred.to_numpy(np.float64)
        y = tgt.y.to_numpy(np.float64)
        # Oracle changes only rows whose true type is confidently joined.
        ora_all = np.where(known, ora_d, 0.0)
        exp_score = bss(y, np.clip(base + exp_d, 0, 1))
        main_exp_score = bss(y, np.clip(base + main_exp_d, 0, 1))
        ora_score = bss(y, np.clip(base + ora_all, 0, 1))
        report["arms"][name] = {
            "source_delta": {str(k): float(v) for k, v in tab.items()},
            "expected_delta_mean": float(exp_d.mean()),
            "expected_delta_sd": float(exp_d.std()),
            "expected_bss": exp_score,
            "expected_bss_gain": exp_score - report["baseline_bss_2024"],
            "oracle_known_rows_bss": ora_score,
            "oracle_known_rows_bss_gain": ora_score - report["baseline_bss_2024"],
            "expected_delta_target_resid_corr": float(np.corrcoef(
                exp_d, y - base)[0, 1]),
            "main_expected_delta_mean": float(main_exp_d.mean()),
            "main_expected_delta_sd": float(main_exp_d.std()),
            "main_expected_bss": main_exp_score,
            "main_expected_bss_gain": (
                main_exp_score - report["baseline_bss_2024"]),
            "main_expected_delta_target_resid_corr": float(np.corrcoef(
                main_exp_d, y - base)[0, 1]),
        }

    report["pitch_head_feature_importance"] = dict(zip(
        PITCH_FEATURES, map(float, pitch_model.get_feature_importance())))
    report["main_pitch_head_top_importance"] = dict(sorted(
        zip(main_pitch_model.feature_names_,
            map(float, main_pitch_model.get_feature_importance())),
        key=lambda x: x[1], reverse=True)[:15])
    path = OUT / "pitch_type_control_gate.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
