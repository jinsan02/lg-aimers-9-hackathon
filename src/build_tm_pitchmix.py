"""trackman 기반 **볼카운트별 구종 성향** (E114) — 계획 ② 재설계.

계획 ②의 원래 전제(링키지 86% 가 tm 피처를 희석한다)는 **틀렸다**.
src/link_pitchers.py 로 상황 튜플 공기 매칭을 새로 만들어 보니 기존 매핑과
99.9% 일치했다. 즉 링키지는 이미 정확했고, E101/E103 이 0 이었던 건
그 피처들 자체에 신호가 없어서다.

그런데 링키지를 파다가 진짜 미개발 자산이 보였다.

  train/test : asof_pitcher_fastball/breaking/offspeed_rate = **시즌 집계뿐**
  trackman   : 투구마다 구종 + **그때의 볼카운트**

즉 trackman 으로만 P(구종 | 투수, 볼카운트) 를 만들 수 있다. 0-2 에서 변화구로
유인하는 투수와 직구로 승부하는 투수는 제구 성공률이 다를 수밖에 없고,
이건 신호감사 오라클 1위인 **투수x카운트(2740.9)** 축에 물리적 근거를 얹는 것이다.

추가로 구종별 릴리스 산포(= 그 구종의 커맨드)를 곱해
  '이 카운트에서 그가 던질 법한 구종의, 그가 가진 커맨드'
라는 기대 커맨드를 만든다. E103 은 구종별 산포를 **카운트와 무관하게** 넣어서
실패했다 - 카운트와 엮이지 않으면 시즌 집계와 다를 게 없기 때문이다.

누수 차단: (투수, 시즌) 단위로 **그 시즌 이전까지만** 누적한다.
  2025 행에는 2019~2024 전체 누적이 붙는다 (train 만으로 만든 사전 정보).
  개별 투구의 실측값을 그 행에 붙이는 것이 아니므로 규정 위반이 아니다.

메모리: usecols 5개만 읽는다. 노트북 대신 4070/A100 에서 실행할 것.

실행: python src/build_tm_pitchmix.py
산출: data/processed/tm_pitchmix.csv  (pitcher_id, season, balls, strikes, tmx_*)
"""

import os
import sys

import numpy as np
import pandas as pd

DATA = "./data"
OUT = f"{DATA}/processed/tm_pitchmix.csv"
K = 40.0          # 카운트별 표본이 얇으므로 투수 전체 구종비율로 수축


def main():
    use = ["pitcher_trackman_id", "season", "balls_before", "strikes_before",
           "pitch_type_group", "rel_height", "rel_side"]
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=use)
    tm = tm[tm.pitch_type_group.notna()]
    tm["b"] = tm["balls_before"].clip(0, 3).astype(np.int8)
    tm["s"] = tm["strikes_before"].clip(0, 2).astype(np.int8)
    print(f"trackman {len(tm):,}행 | 구종 {sorted(tm.pitch_type_group.unique())}")

    # 링키지 (새 매핑 우선)
    p2 = f"{DATA}/processed/pitcher_map2.csv"
    pmap = pd.read_csv(p2 if os.path.exists(p2)
                       else f"{DATA}/processed/pitcher_map.csv")
    if "margin" in pmap.columns and "score" in pmap.columns and \
            pmap.columns.tolist()[:2] != ["pitcher_id", "tm_id"]:
        pmap = pmap[["pitcher_id", "tm_id"]]
    pmap = pmap[["pitcher_id", "tm_id"]].drop_duplicates("tm_id")
    tm = tm.merge(pmap, left_on="pitcher_trackman_id", right_on="tm_id",
                  how="inner")
    print(f"링키지 후 {len(tm):,}행 | 투수 {tm.pitcher_id.nunique()}명")

    types = ["fastball", "breaking", "offspeed"]

    # ---- ① (투수, 시즌, 카운트) 구종 카운트 + 구종별 릴리스 산포
    g = tm.groupby(["pitcher_id", "season", "b", "s", "pitch_type_group"])
    cnt = g.size().rename("n").reset_index()
    piv = cnt.pivot_table(index=["pitcher_id", "season", "b", "s"],
                          columns="pitch_type_group", values="n",
                          fill_value=0).reset_index()
    for t in types:
        if t not in piv.columns:
            piv[t] = 0
    piv["tot"] = piv[types].sum(1)

    # 투수 전체(카운트 무관) 구종비율 = 수축 목표
    gp = tm.groupby(["pitcher_id", "season", "pitch_type_group"]).size() \
        .rename("n").reset_index()
    base = gp.pivot_table(index=["pitcher_id", "season"],
                          columns="pitch_type_group", values="n",
                          fill_value=0)
    for t in types:
        if t not in base.columns:
            base[t] = 0
    base = base[types]
    basr = base.div(base.sum(1).replace(0, np.nan), axis=0)
    basr.columns = [f"base_{t}" for t in types]

    piv = piv.merge(basr.reset_index(), on=["pitcher_id", "season"], how="left")
    for t in types:
        # 카운트별 비율을 그 투수 전체 비율로 수축 (얇은 카운트 보호)
        piv[f"tmx_{t}"] = ((piv[t] + K * piv[f"base_{t}"].fillna(1 / 3))
                           / (piv["tot"] + K))
        # 전체 대비 편차 = **이 카운트에서만의 성향** (E95 _dev 와 같은 모양)
        piv[f"tmx_{t}_dev"] = piv[f"tmx_{t}"] - piv[f"base_{t}"].fillna(1 / 3)

    # ---- ② 구종별 릴리스 산포 = 그 구종의 커맨드
    sc = tm.groupby(["pitcher_id", "season", "pitch_type_group"]) \
        .agg(h=("rel_height", "std"), sd=("rel_side", "std"),
             n=("rel_height", "size")).reset_index()
    sc = sc[sc["n"] >= 40]
    sc["sct"] = np.sqrt(sc["h"] ** 2 + sc["sd"] ** 2)
    scp = sc.pivot_table(index=["pitcher_id", "season"],
                         columns="pitch_type_group", values="sct")
    scp.columns = [f"sct_{c}" for c in scp.columns]
    piv = piv.merge(scp.reset_index(), on=["pitcher_id", "season"], how="left")

    # 기대 커맨드 = sum_t P(구종 t | 카운트) x (그 구종의 산포)
    exp = np.zeros(len(piv))
    wsum = np.zeros(len(piv))
    for t in types:
        c = f"sct_{t}"
        if c not in piv.columns:
            continue
        v = piv[c].to_numpy(np.float64)
        w = piv[f"tmx_{t}"].to_numpy(np.float64)
        m = np.isfinite(v)
        exp[m] += w[m] * v[m]
        wsum[m] += w[m]
    piv["tmx_exp_scatter"] = np.where(wsum > 0, exp / np.maximum(wsum, 1e-9),
                                      np.nan)

    cols = [c for c in piv.columns if c.startswith(("tmx_", "sct_"))]
    piv = piv[["pitcher_id", "season", "b", "s"] + cols]

    # ---- ③ 그 시즌 **이전까지**만 누적 (누수 차단) + 2025 행 생성
    piv = piv.sort_values(["pitcher_id", "b", "s", "season"])
    out = []
    for _, sub in piv.groupby(["pitcher_id", "b", "s"], sort=False):
        c = sub[cols].expanding().mean().shift(1)
        c[["pitcher_id", "b", "s"]] = sub[["pitcher_id", "b", "s"]].to_numpy()
        c["season"] = sub["season"].to_numpy()
        out.append(c)
        last = c.tail(1).copy()
        # 직전 시즌까지 전부 누적한 값 = 2025 행
        full = sub[cols].mean()
        for k2 in cols:
            last[k2] = full[k2]
        last["season"] = sub["season"].max() + 1
        out.append(last)
    res = pd.concat(out).dropna(subset=cols, how="all")
    res = res[["pitcher_id", "season", "b", "s"] + cols]
    res = res.rename(columns={"b": "balls_before", "s": "strikes_before"})

    # ---- ④ **결측이 정보를 담지 않게** 만든다 (E115 실패 원인 수정)
    # 실측: tmx 결측률이 2019 100% / 나머지 13~23% 이고, 결측 행 성공률 0.5441 vs
    # 비결측 0.5145 (차이 +0.0295). 즉 '결측'이 사실상 '2019년'의 대리 표지였고,
    # 트리가 "결측 = 성공률 높음"을 학습해 2025 로 이전되지 않았다 (-9.5).
    # std 피처가 시즌 리그평균으로 채워 결측 0% 인 것과 같은 처리를 한다:
    # **카운트별 리그 평균 프로필**을 모든 (투수 x 시즌 x 카운트) 격자에 채운다.
    lg = res.groupby(["balls_before", "strikes_before"])[cols].mean()
    pids = sorted(res["pitcher_id"].unique())
    # 2019 는 expanding shift 때문에 표가 없다. 그런데 '2019 = 전부 결측'이
    # 바로 트리가 오용하던 교란이므로, 2019 도 격자에 넣어 리그평균으로 채운다.
    # ("정보 없으면 리그 평균" - std 피처와 같은 원칙)
    seasons = range(min(2019, int(res["season"].min())),
                    int(res["season"].max()) + 1)
    grid = pd.MultiIndex.from_product(
        [pids, seasons, range(4), range(3)],
        names=["pitcher_id", "season", "balls_before", "strikes_before"])
    res = res.set_index(["pitcher_id", "season", "balls_before",
                         "strikes_before"]).reindex(grid)
    fill = lg.reindex(
        pd.MultiIndex.from_arrays(
            [res.index.get_level_values("balls_before"),
             res.index.get_level_values("strikes_before")]))
    fill.index = res.index
    res = res.fillna(fill).reset_index()
    print(f"  결측 채움 후: {res[cols].isna().mean().max() * 100:.2f}% (최대)")

    os.makedirs(f"{DATA}/processed", exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"\n저장 {OUT}: {len(res):,}행, 피처 {len(cols)}개")
    print(f"  투수 {res.pitcher_id.nunique()}명 | 시즌 "
          f"{res.season.min():.0f}~{res.season.max():.0f}")
    d = res[[c for c in cols if c.endswith("_dev")]].describe()
    print(d.loc[["mean", "std", "min", "max"]].round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
