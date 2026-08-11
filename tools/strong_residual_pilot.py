"""Learn only residual structure left by an honest K0 champion analogue.

The residual model is selected on 2022->2023 R rows, refitted on the honest
2021->2022 and 2022->2023 target residuals, then evaluated once on 2024.  F is
left untouched because its 2022->2023 label regime is discontinuous.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from pbmf_transfer_audit import fit_map, middle_keys, pair_keys, post


ROOT = Path(__file__).resolve().parents[1]
CAT = ["top_bottom", "game_type", "base_state", "pitcher_hand",
       "batter_hand", "pitcher_team_id", "batter_team_id"]
WEIGHTS = (.02, .05, .10, .20, .50, 1.0)


def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0, 1)
    r = y.mean()
    return 1e5 * (1-np.mean((p-y)**2)/(r*(1-r)))


def pred(stem, kind):
    z = np.load(ROOT/"out"/f"{stem}_{kind}_preds.npz", allow_pickle=True)
    return z["y"].astype(float), z["pred"].astype(float), \
        (z["row_id"] if "row_id" in z.files else None)


def transition(data, source, target, base_stem, cell_stem):
    ys, bv, _ = pred(base_stem, "val")
    ysc, cv, _ = pred(cell_stem, "val")
    yt, bt, rid = pred(base_stem, "test")
    ytc, ct, ridc = pred(cell_stem, "test")
    if not (np.array_equal(ys, ysc) and np.array_equal(yt, ytc)
            and np.array_equal(rid, ridc)):
        raise ValueError(f"prediction mismatch {source}->{target}")
    src = data[data.season.eq(source)].reset_index(drop=True)
    tgt = data[data.season.eq(target)].reset_index(drop=True)
    tgt = tgt.set_index("row_id").loc[rid].reset_index()
    if len(src) != len(ys) or len(tgt) != len(yt):
        raise ValueError(f"row mismatch {source}->{target}")
    ps, pt = post(.45*bv+.55*cv), post(.45*bt+.55*ct)
    ms, mt = middle_keys(
        src.asof_pitcher_prev5_game_middle_rate.to_numpy(float),
        tgt.asof_pitcher_prev5_game_middle_rate.to_numpy(float))
    r = ys-ps
    r -= r.mean()
    ma, mb = fit_map(ms, mt, r, 500.)
    r = ys-(ps+ma)
    r -= r.mean()
    ea, eb = fit_map(pair_keys(src), pair_keys(tgt), r, 500.)
    return tgt, yt, np.clip(pt+mb+eb, 0, 1)


def strong_2024(data):
    def mean_files(stem, seeds):
        zs = [np.load(ROOT/"out"/f"cat_{stem}_s{s}_val_preds.npz")
              for s in seeds]
        y = zs[0]["y"].astype(float)
        if any(not np.array_equal(y, z["y"]) for z in zs[1:]):
            raise ValueError(f"seed mismatch {stem}")
        return y, np.mean([z["pred"].astype(float) for z in zs], axis=0)

    y, base = mean_files("VB2_base", (42, 7, 13, 3, 4, 5, 6, 8))
    yc, cell = mean_files("ZD5", (42, 7, 13, 3, 4, 5))
    if not np.array_equal(y, yc):
        raise ValueError("strong base/cell mismatch")
    src = data[data.season.eq(2023)].reset_index(drop=True)
    tgt = data[data.season.eq(2024)].reset_index(drop=True)
    ys, lb, _ = pred("cat_MVA_native", "val")
    ysc, lc, _ = pred("cat_MVCELL_s42", "val")
    if not np.array_equal(ys, ysc):
        raise ValueError("source local base/cell mismatch")
    ps = post(.45*lb+.55*lc)
    ms, mt = middle_keys(
        src.asof_pitcher_prev5_game_middle_rate.to_numpy(float),
        tgt.asof_pitcher_prev5_game_middle_rate.to_numpy(float))
    r = ys-ps
    r -= r.mean()
    ma, mb = fit_map(ms, mt, r, 500.)
    r = ys-(ps+ma)
    r -= r.mean()
    ea, eb = fit_map(pair_keys(src), pair_keys(tgt), r, 500.)
    return tgt, y, np.clip(post(.45*base+.55*cell)+mb+eb, 0, 1)


def design(frame, champion_p, features):
    x = frame[features].copy()
    x["champion_p"] = np.asarray(champion_p, np.float32)
    for c in CAT:
        x[c] = x[c].astype(str)
    return x


def gains(frame, y, p0, adj, weight):
    masks = {
        "R": frame.game_type.eq("R").to_numpy(),
        "R_early": (frame.game_type.eq("R") & frame.game_month.le(6)).to_numpy(),
        "R_late": (frame.game_type.eq("R") & frame.game_month.gt(6)).to_numpy(),
        "all": np.ones(len(frame), bool),
    }
    q = np.clip(p0+weight*adj, 0, 1)
    return {name: bss(y[m], q[m])-bss(y[m], p0[m]) for name, m in masks.items()}


def main():
    input_cols = list(pd.read_csv(ROOT/"data"/"test.csv", nrows=0).columns)
    features = [c for c in input_cols if c not in
                ("row_id", "pitcher_id", "batter_id")]
    use = list(dict.fromkeys(input_cols+["control_success"]))
    data = pd.read_csv(ROOT/"data"/"train.csv", usecols=use)
    d22, y22, p22 = transition(data, 2021, 2022, "cat_MV21_base",
                               "cat_MV21_cell")
    d23, y23, p23 = transition(data, 2022, 2023, "cat_MVB22_native",
                               "cat_MVCELL22_s42")
    d24, y24, p24 = transition(data, 2023, 2024, "cat_MVA_native",
                               "cat_MVCELL_s42")
    ds, ys, ps = strong_2024(data)
    if not np.array_equal(y24, ys):
        raise ValueError("local/strong 2024 target mismatch")

    r22, r23 = y22-p22, y23-p23
    m22 = d22.game_type.eq("R").to_numpy()
    m23 = d23.game_type.eq("R").to_numpy()
    m24 = d24.game_type.eq("R").to_numpy()
    r22 -= r22[m22].mean()
    r23 -= r23[m23].mean()
    x22 = design(d22, p22, features)
    x23 = design(d23, p23, features)
    params = dict(iterations=1500, learning_rate=.02, depth=4,
                  l2_leaf_reg=100, random_strength=1,
                  loss_function="RMSE", eval_metric="RMSE",
                  task_type="GPU", devices="0", random_seed=42,
                  verbose=200)
    model = CatBoostRegressor(**params, early_stopping_rounds=200)
    model.fit(Pool(x22.loc[m22], r22[m22], cat_features=CAT),
              eval_set=Pool(x23.loc[m23], r23[m23], cat_features=CAT))
    best = model.get_best_iteration()
    a23 = np.zeros(len(d23), float)
    a23[m23] = model.predict(x23.loc[m23])
    rows = []
    for w in WEIGHTS:
        g = gains(d23, y23, p23, a23, w)
        rows.append((min(g["R_early"], g["R_late"]), g["R"], w, g))
        print(f"select w={w:g} " + " ".join(f"{k}={v:+.3f}" for k,v in g.items()))
    eligible = [z for z in rows if z[0] > 0 and z[1] > 0]
    chosen = max(eligible or rows, key=lambda z: (z[0], z[1]))
    weight = chosen[2]
    print(f"chosen source-only weight={weight:g} best_iter={best} "
          f"min_half={chosen[0]:+.3f}")

    both = pd.concat([x22.loc[m22], x23.loc[m23]], ignore_index=True)
    rr = np.concatenate([r22[m22], r23[m23]])
    rounds = max(1, int((best+1)*1.5))
    final = CatBoostRegressor(**{**params, "iterations": rounds, "verbose": 0})
    final.fit(Pool(both, rr, cat_features=CAT))

    for label, frame, y, p in (("local", d24, y24, p24),
                               ("strong", ds, ys, ps)):
        x = design(frame, p, features)
        a = np.zeros(len(frame), float)
        mask = frame.game_type.eq("R").to_numpy()
        a[mask] = final.predict(x.loc[mask])
        g = gains(frame, y, p, a, weight)
        print(f"target {label} adj_sd={a[mask].std():.7f} adj_mean={a[mask].mean():+.7f} "
              + " ".join(f"{k}={v:+.3f}" for k,v in g.items()))


if __name__ == "__main__":
    main()
