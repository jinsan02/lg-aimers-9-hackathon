"""14셀 전체 확률을 이용한 저차원 메타모델의 시즌 전이 검문.

스칼라 비교군: base 확률 + 성공셀 합.
신규 팔 1: base + 성공셀 합 + 실패모드별 주변확률(middle/ball/reverse).
신규 팔 2: base + 14셀 확률 전체.

alpha와 팔 선택은 source 시즌 전·후반 양방향 전이만으로 한다. 그 뒤 source 전체에
재적합해 미학습 target 시즌에 한 번 적용한다. target의 전역 편향은 제거하므로
비교값은 순위/해상도 증분이다.
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd


ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0)


def load_base(tag: str, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    paths = sorted(glob.glob(f"./out/*_{tag}_s*_{kind}_preds.npz"))
    if not paths:
        raise FileNotFoundError(f"base 없음: {tag} {kind}")
    z = [np.load(p, allow_pickle=True) for p in paths]
    y = z[0]["y"].astype(np.float64)
    if any(not np.array_equal(y, q["y"]) for q in z[1:]):
        raise ValueError(f"base 시드 y 불일치: {tag} {kind}")
    row_id = z[0]["row_id"] if "row_id" in z[0] else None
    return np.mean([q["pred"] for q in z], axis=0), y, row_id


def load_cell(tag: str, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if kind == "val":
        paths = sorted(glob.glob(f"./out/*_{tag}_s*_cell_val.npz"))
        proba_key, success_key = "proba", "success"
    else:
        paths = sorted(glob.glob(f"./out/*_{tag}_s*_test_preds.npz"))
        proba_key, success_key = "cell_proba", "cell_success"
    if not paths:
        raise FileNotFoundError(f"cell 없음: {tag} {kind}")
    z = [np.load(p, allow_pickle=True) for p in paths]
    y = z[0]["y"].astype(np.float64)
    row_id = z[0]["row_id"]
    if any(not np.array_equal(y, q["y"]) or not np.array_equal(row_id, q["row_id"])
           for q in z[1:]):
        raise ValueError(f"cell 시드 행 불일치: {tag} {kind}")
    names = z[0]["names"] if "names" in z[0] else None
    if names is None:
        # test 파일에는 names를 중복 저장하지 않는다. 14셀 기본 순서는 val에서 읽는다.
        val_path = sorted(glob.glob(f"./out/*_{tag}_s*_cell_val.npz"))[0]
        names = np.load(val_path, allow_pickle=True)["names"]
    if any(q[proba_key].shape[1] != len(names) for q in z):
        raise ValueError("셀 확률 열 수/이름 불일치")
    success = z[0][success_key].astype(int)
    P = np.mean([q[proba_key].astype(np.float64) for q in z], axis=0)
    return P, y, row_id, names.astype(str)


def centered_bss(y: np.ndarray, p: np.ndarray) -> float:
    p = p - np.mean(p - y)
    r = float(y.mean())
    return 1e5 * (1.0 - np.mean((np.clip(p, 0.0, 1.0) - y) ** 2)
                  / (r * (1.0 - r)))


def design(base: np.ndarray, cells: np.ndarray, names: np.ndarray,
           arm: str) -> np.ndarray:
    success = np.array([str(v).startswith("1") for v in names])
    ps = cells[:, success].sum(1)
    if arm == "scalar":
        return np.column_stack([base, ps])
    if arm == "modes":
        modes = []
        for pos in (1, 2, 3):
            pick = np.array([len(str(v)) > pos and str(v)[pos] == "1" for v in names])
            modes.append(cells[:, pick].sum(1) if pick.any() else np.zeros(len(cells)))
        return np.column_stack([base, ps, *modes])
    if arm == "cells":
        return np.column_stack([base, cells])
    raise ValueError(arm)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-10] = 1.0
    Z = (X - mu) / sd
    ym = float(y.mean())
    beta = np.linalg.solve(Z.T @ Z + alpha * len(Z) * np.eye(Z.shape[1]),
                           Z.T @ (y - ym))
    return mu, sd, beta, ym


def ridge_predict(pack: tuple[np.ndarray, np.ndarray, np.ndarray, float],
                  X: np.ndarray) -> np.ndarray:
    mu, sd, beta, ym = pack
    return ym + ((X - mu) / sd) @ beta


def main() -> int:
    base_tag = sys.argv[1] if len(sys.argv) > 1 else "AB_base"
    cell_tag = sys.argv[2] if len(sys.argv) > 2 else "OG1_cell14"
    bs, ys, _ = load_base(base_tag, "val")
    bt, yt, rid_bt = load_base(base_tag, "test")
    Cs, ycs, rid_s, names = load_cell(cell_tag, "val")
    Ct, yct, rid_t, names_t = load_cell(cell_tag, "test")
    if not np.array_equal(ys, ycs) or not np.array_equal(yt, yct):
        raise ValueError("base/cell y 불일치")
    if rid_bt is not None and not np.array_equal(rid_bt, rid_t):
        raise ValueError("target row_id 불일치")
    if not np.array_equal(names, names_t):
        raise ValueError("val/test 셀 이름 불일치")

    meta = pd.read_csv("./data/train.csv", encoding="utf-8-sig",
                       usecols=["row_id", "game_month", "game_type"])
    month_map = pd.Series(meta.game_month.to_numpy(), index=meta.row_id)
    league_map = pd.Series(meta.game_type.astype(str).to_numpy(), index=meta.row_id)
    month = month_map.reindex(rid_s).to_numpy()
    league_s = league_map.reindex(rid_s).to_numpy()
    league_t = league_map.reindex(rid_t).to_numpy()
    if np.isnan(month).any():
        raise ValueError("source month 결측")
    first = month <= 6
    Xs = {arm: design(bs, Cs, names, arm) for arm in ("scalar", "modes", "cells")}
    Xt = {arm: design(bt, Ct, names, arm) for arm in ("scalar", "modes", "cells")}

    print(f"source={len(ys):,} target={len(yt):,} cells={len(names)} "
          f"source halves={first.sum():,}/{(~first).sum():,}")

    def scenario(label: str, source_mask: np.ndarray,
                 target_route: np.ndarray) -> tuple[str, float, float, float]:
        """target_route 밖에서는 신규 팔을 scalar와 같게 둬 그 리그에 외삽하지 않는다."""
        print(f"\n=== {label} source={source_mask.sum():,} route={target_route.sum():,} ===")
        print("source_delta는 같은 alpha의 scalar 대비 양방향 centered BSS 증분")
        rows = []
        for alpha in ALPHAS:
            transfer = {arm: [] for arm in Xs}
            for fit, test in ((source_mask & first, source_mask & ~first),
                              (source_mask & ~first, source_mask & first)):
                for arm0 in Xs:
                    pack = ridge_fit(Xs[arm0][fit], ys[fit], alpha)
                    transfer[arm0].append(centered_bss(
                        ys[test], ridge_predict(pack, Xs[arm0][test])))
            for arm0 in ("modes", "cells"):
                d = np.array(transfer[arm0]) - np.array(transfer["scalar"])
                rows.append((float(d.min()), float(d.mean()), alpha, arm0))
                print(f"alpha={alpha:<4g} {arm0:<5} source_delta "
                      f"{d[0]:+7.3f}/{d[1]:+7.3f} "
                      f"min={d.min():+7.3f} mean={d.mean():+7.3f}")
        eligible = [r for r in rows if r[0] > 0]
        chosen = max(eligible or rows, key=lambda r: (r[0], r[1]))
        source_min, _, alpha, arm0 = chosen
        scalar = ridge_predict(ridge_fit(Xs["scalar"][source_mask],
                                         ys[source_mask], alpha), Xt["scalar"])
        candidate = ridge_predict(ridge_fit(Xs[arm0][source_mask],
                                            ys[source_mask], alpha), Xt[arm0])
        candidate = np.where(target_route, candidate, scalar)
        delta = centered_bss(yt, candidate) - centered_bss(yt, scalar)
        print(f"선택(source only): arm={arm0} alpha={alpha:g} source_min={source_min:+.3f}")
        print(f"target centered scalar={centered_bss(yt, scalar):.3f} "
              f"routed={centered_bss(yt, candidate):.3f} delta={delta:+.3f}")
        return arm0, alpha, source_min, delta

    # 전체는 진단용이다. 2023 F는 신체제 과거가 없어 정직한 source가 아니다.
    scenario("ALL (진단용: 2023 F 체제 공백)", np.ones(len(ys), dtype=bool),
             np.ones(len(yt), dtype=bool))
    result = scenario("R->R route (주 판정)", league_s == "R", league_t == "R")
    print("\n승격 게이트(주 판정): R source 양방향 모두 양수 + "
          "R-only route의 전체 target delta >= +3.0")
    print(f"RESULT arm={result[0]} alpha={result[1]:g} "
          f"source_min={result[2]:+.3f} target_delta={result[3]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
