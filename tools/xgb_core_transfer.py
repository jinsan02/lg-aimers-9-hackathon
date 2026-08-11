"""Fixed-weight XGBoost diversity gate on three unseen-season transitions."""

from pathlib import Path

import numpy as np
import pandas as pd

from weak_signal_salvage import fit_map, middle_keys, pair_keys, post


ROOT = Path(__file__).resolve().parents[1]


def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0, 1)
    r = y.mean()
    return 1e5 * (1 - np.mean((p-y)**2)/(r*(1-r)))


def load(stem, kind="test"):
    z = np.load(ROOT/"out"/f"{stem}_{kind}_preds.npz", allow_pickle=True)
    row_id = z["row_id"] if "row_id" in z.files else None
    return z["y"].astype(float), z["pred"].astype(float), row_id


def one(label, base_stem, xgb_stem):
    y, base, rid = load(base_stem)
    yx, xgb, rix = load(xgb_stem)
    if not np.array_equal(y, yx) or not np.array_equal(rid, rix):
        raise ValueError(f"target/row mismatch: {label}")
    meta = pd.read_csv(ROOT/"data"/"train.csv",
                       usecols=["row_id", "game_type", "game_month"])
    meta = meta.set_index("row_id").loc[rid]
    masks = {
        "all": np.ones(len(y), bool),
        "R": meta.game_type.eq("R").to_numpy(),
        "F": meta.game_type.eq("F").to_numpy(),
        "early": meta.game_month.le(6).to_numpy(),
        "late": meta.game_month.gt(6).to_numpy(),
    }
    print(f"\n=== {label} base={bss(y,base):.3f} xgb={bss(y,xgb):.3f} "
          f"rms={np.sqrt(np.mean((base-xgb)**2)):.7f} ===")
    for w in (.02, .05, .10, .20):
        q = (1-w)*base+w*xgb
        parts = []
        for name, mask in masks.items():
            parts.append(f"{name}={bss(y[mask],q[mask])-bss(y[mask],base[mask]):+.3f}")
        print(f"w={w:.2f} " + " ".join(parts))


def k0_one(label, source, target, base_stem, cell_stem, xgb_stem):
    ys, bv, _ = load(base_stem, "val")
    ysc, cv, _ = load(cell_stem, "val")
    yt, bt, rid = load(base_stem)
    ytc, ct, ridc = load(cell_stem)
    yxv, xv, _ = load(xgb_stem, "val")
    yxt, xt, ridx = load(xgb_stem)
    if not (np.array_equal(ys, ysc) and np.array_equal(ys, yxv)
            and np.array_equal(yt, ytc) and np.array_equal(yt, yxt)
            and np.array_equal(rid, ridc) and np.array_equal(rid, ridx)):
        raise ValueError(f"K0 target/row mismatch: {label}")
    cols = ["season", "row_id", "pitcher_id", "batter_id", "game_type",
            "game_month", "asof_pitcher_prev5_game_middle_rate"]
    d = pd.read_csv(ROOT/"data"/"train.csv", usecols=cols)
    src = d[d.season.eq(source)].reset_index(drop=True)
    tgt = d[d.season.eq(target)].reset_index(drop=True)
    # Validation arrays follow CSV season order; target row_id is explicit.
    tgt = tgt.set_index("row_id").loc[rid].reset_index()
    ps = post(.45*bv+.55*cv)
    pt = post(.45*bt+.55*ct)
    ms, mt = middle_keys(
        src.asof_pitcher_prev5_game_middle_rate.to_numpy(float),
        tgt.asof_pitcher_prev5_game_middle_rate.to_numpy(float))
    r = ys-ps
    r -= r.mean()
    ma, mb = fit_map(ms, mt, r, 500.)
    r = ys-(ps+ma)
    r -= r.mean()
    ea, eb = fit_map(pair_keys(src), pair_keys(tgt), r, 500.)
    k0s, k0t = ps+ma+ea, pt+mb+eb
    masks = {
        "all": np.ones(len(yt), bool),
        "R": tgt.game_type.eq("R").to_numpy(),
        "F": tgt.game_type.eq("F").to_numpy(),
        "early": tgt.game_month.le(6).to_numpy(),
        "late": tgt.game_month.gt(6).to_numpy(),
    }
    print(f"\n=== K0 {label} score={bss(yt,k0t):.3f} "
          f"xgb={bss(yt,xt):.3f} rms={np.sqrt(np.mean((k0t-xt)**2)):.7f} ===")
    for w in (.02, .05, .10, .20):
        q = (1-w)*k0t+w*xt
        parts = [f"{name}={bss(yt[m],q[m])-bss(yt[m],k0t[m]):+.3f}"
                 for name, m in masks.items()]
        print(f"w={w:.2f} " + " ".join(parts))
    return yt, k0t, tgt


def latest_multiseed(yt, k0, meta):
    stems = ["xgb_XCR1"] + [f"xgb_XCR1N5_s{s}" for s in (3, 4, 5, 7, 13)]
    preds, gains = [], []
    for stem in stems:
        y, p, _ = load(stem)
        if not np.array_equal(y, yt):
            raise ValueError(f"multiseed target mismatch: {stem}")
        preds.append(p)
        gains.append(bss(yt, .9*k0+.1*p)-bss(yt, k0))
    p = np.mean(preds, axis=0)
    masks = {
        "all": np.ones(len(yt), bool),
        "R": meta.game_type.eq("R").to_numpy(),
        "F": meta.game_type.eq("F").to_numpy(),
        "early": meta.game_month.le(6).to_numpy(),
        "late": meta.game_month.gt(6).to_numpy(),
    }
    print("\n=== K0 2023->2024 XGB 6-seed fixed w=.10 ===")
    print("per_seed_gain=" + ",".join(f"{g:+.3f}" for g in gains)
          + f" mean={np.mean(gains):+.3f} sd={np.std(gains,ddof=1):.3f}")
    q = .9*k0+.1*p
    print("ensemble " + " ".join(
        f"{name}={bss(yt[m],q[m])-bss(yt[m],k0[m]):+.3f}"
        for name, m in masks.items()))
    y2, p2, _ = load("xgb_XCR2")
    if not np.array_equal(y2, yt):
        raise ValueError("XCR2 target mismatch")
    q2 = .9*k0+.1*p2
    print("refit1.5 " + " ".join(
        f"{name}={bss(yt[m],q2[m])-bss(yt[m],k0[m]):+.3f}"
        for name, m in masks.items()))


def strong_v11_surface():
    def mean_files(paths):
        zs = [np.load(p, allow_pickle=True) for p in paths]
        y = zs[0]["y"].astype(float)
        if any(not np.array_equal(y, z["y"]) for z in zs[1:]):
            raise ValueError("strong-v11 seed target mismatch")
        return y, np.mean([z["pred"].astype(float) for z in zs], axis=0)

    base_files = [ROOT/"out"/f"cat_VB2_base_s{s}_val_preds.npz"
                  for s in (42, 7, 13, 3, 4, 5, 6, 8)]
    cell_files = [ROOT/"out"/f"cat_ZD5_s{s}_val_preds.npz"
                  for s in (42, 7, 13, 3, 4, 5)]
    if not all(p.exists() for p in base_files+cell_files):
        raise FileNotFoundError("strong v11 4070 prediction asset missing")
    y, base = mean_files(base_files)
    yc, cell = mean_files(cell_files)
    if not np.array_equal(y, yc):
        raise ValueError("strong-v11 base/cell target mismatch")
    d = pd.read_csv(ROOT/"data"/"train.csv",
                    usecols=["season", "game_month", "game_type",
                             "asof_pitcher_prev5_game_middle_rate",
                             "pitcher_id", "batter_id"])
    src = d[d.season.eq(2023)].reset_index(drop=True)
    tgt = d[d.season.eq(2024)].reset_index(drop=True)
    # Freeze K0 on the honest local 2023 OOF residual, then apply only its
    # offsets to the 4070 target ensemble.  The 2025 submission constants are
    # fitted on 2024 labels and would leak if reapplied here.
    ys, lb, _ = load("cat_MVA_native", "val")
    ysc, lc, _ = load("cat_MVCELL_s42", "val")
    if not np.array_equal(ys, ysc):
        raise ValueError("local source base/cell mismatch")
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
    k0 = np.clip(post(.45*base+.55*cell)+mb+eb, 0, 1)
    stems = ["xgb_XCR1"] + [f"xgb_XCR1N5_s{s}" for s in (3, 4, 5, 7, 13)]
    xp = np.mean([load(stem)[1] for stem in stems], axis=0)
    masks = {
        "all": np.ones(len(y), bool), "R": tgt.game_type.eq("R").to_numpy(),
        "F": tgt.game_type.eq("F").to_numpy(),
        "early": tgt.game_month.le(6).to_numpy(),
        "late": tgt.game_month.gt(6).to_numpy(),
    }
    print("\n=== strong v11-analogue (4070 Cat assets) + local XGB6 ===")
    print(f"base={bss(y,k0):.3f} xgb={bss(y,xp):.3f} "
          f"rms={np.sqrt(np.mean((k0-xp)**2)):.7f}")
    for w in (.02, .05, .10, .20):
        q = (1-w)*k0+w*xp
        print(f"w={w:.2f} " + " ".join(
            f"{name}={bss(y[m],q[m])-bss(y[m],k0[m]):+.3f}"
            for name, m in masks.items()))
    y2, p2, _ = load("xgb_XCR2")
    if not np.array_equal(y2, y):
        raise ValueError("strong-v11 XCR2 target mismatch")
    for w in (.02, .05, .10):
        q2 = (1-w)*k0+w*p2
        print(f"refit1.5-w{w:.2f} " + " ".join(
            f"{name}={bss(y[m],q2[m])-bss(y[m],k0[m]):+.3f}"
            for name, m in masks.items()))


def main():
    one("2021->2022", "cat_MV21_base", "xgb_XCR21")
    one("2022->2023", "cat_MVB22_native", "xgb_XCR0")
    one("2023->2024", "cat_MVA_native", "xgb_XCR1")
    k0_one("2021->2022", 2021, 2022, "cat_MV21_base",
           "cat_MV21_cell", "xgb_XCR21")
    k0_one("2022->2023", 2022, 2023, "cat_MVB22_native",
           "cat_MVCELL22_s42", "xgb_XCR0")
    latest = k0_one("2023->2024", 2023, 2024, "cat_MVA_native",
                    "cat_MVCELL_s42", "xgb_XCR1")
    latest_multiseed(*latest)
    strong_v11_surface()


if __name__ == "__main__":
    main()
