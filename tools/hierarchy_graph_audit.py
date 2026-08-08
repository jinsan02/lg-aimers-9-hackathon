"""동적 계층모형/GNN 설계 전 pitcher-batter 그래프의 유효 구조를 감사한다.

시즌 S를 볼 때 그래프와 모든 통계는 season < S로만 만든다. 평가 시즌 행끼리는
절대 연결하지 않는다. GNN 후보가 기존 asof/TE와 다른 정보를 가질 수 있는지 보는
구조 게이트이며 모델 성능 판정은 아니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DATA = "./data/train.csv"
T = "control_success"


def q(v, ps=(.1, .5, .9)):
    return "/".join(f"{x:.0f}" for x in np.quantile(v, ps)) if len(v) else "-"


def q3(v, ps=(.1, .5, .9)):
    return "/".join(f"{x:.3f}" for x in np.quantile(v, ps)) if len(v) else "-"


def audit(df: pd.DataFrame, season: int) -> None:
    past = df[df.season < season]
    cur = df[df.season == season]
    p_seen, b_seen = set(past.pitcher_id), set(past.batter_id)
    edges = past[["pitcher_id", "batter_id"]].drop_duplicates()
    edge_idx = pd.MultiIndex.from_frame(edges)
    cur_idx = pd.MultiIndex.from_frame(cur[["pitcher_id", "batter_id"]])
    pdeg = edges.groupby("pitcher_id").batter_id.nunique()
    bdeg = edges.groupby("batter_id").pitcher_id.nunique()
    density = len(edges) / max(past.pitcher_id.nunique() * past.batter_id.nunique(), 1)

    # 1-hop 이웃의 과거 성공률을 다시 평균한 값. 투수 자체 과거율과 거의 같거나
    # 투수 사이 분산이 0이면 GNN 메시지는 새 정보 없이 전역평균으로 붕괴한다.
    br = past.groupby("batter_id")[T].agg(["mean", "size"])
    pr = past.groupby("pitcher_id")[T].agg(["mean", "size"])
    ep = edges.merge(br["mean"].rename("b_rate"), left_on="batter_id", right_index=True)
    eb = edges.merge(pr["mean"].rename("p_rate"), left_on="pitcher_id", right_index=True)
    p_neigh = ep.groupby("pitcher_id").b_rate.mean()
    b_neigh = eb.groupby("batter_id").p_rate.mean()
    common_p = pr.join(p_neigh.rename("neigh"), how="inner")
    common_b = br.join(b_neigh.rename("neigh"), how="inner")

    # 투수 두 명의 타자 이웃이 얼마나 겹치는지 표본 Jaccard. 1에 가까우면
    # GraphSAGE 평균 집계가 거의 같은 메시지를 받는다.
    neigh = edges.groupby("pitcher_id").batter_id.agg(set)
    ids = list(neigh.index)
    rng = np.random.default_rng(20260809 + season)
    jac = []
    for _ in range(min(5000, len(ids) * max(len(ids) - 1, 0) // 2)):
        a, b = rng.choice(ids, 2, replace=False)
        A, B = neigh[a], neigh[b]
        jac.append(len(A & B) / max(len(A | B), 1))

    # 실제 target 행 빈도로 투수를 뽑은 Jaccard. 한두 번 나온 저차수 선수가
    # 균등 표본을 지배하는 착시를 제거한다.
    active = cur[cur.pitcher_id.isin(neigh.index)].pitcher_id.value_counts()
    aids = active.index.to_numpy()
    prob = (active / active.sum()).to_numpy()
    jac_w = []
    if len(aids) >= 2:
        for _ in range(5000):
            a, b = rng.choice(aids, 2, replace=False, p=prob)
            A, B = neigh[a], neigh[b]
            jac_w.append(len(A & B) / max(len(A | B), 1))

    print(f"\n=== target season {season} (past rows={len(past):,}, target={len(cur):,}) ===")
    print(f"nodes pitcher={past.pitcher_id.nunique():,} batter={past.batter_id.nunique():,} "
          f"unique_edges={len(edges):,} density={density:.3%}")
    print(f"target row hit pitcher={cur.pitcher_id.isin(p_seen).mean():.1%} "
          f"batter={cur.batter_id.isin(b_seen).mean():.1%} "
          f"exact_pair={cur_idx.isin(edge_idx).mean():.1%}")
    print(f"unique degree q10/50/90 pitcher->{q(pdeg.to_numpy())} "
          f"batter->{q(bdeg.to_numpy())}")
    print(f"pitcher neighbor-rate sd={common_p.neigh.std():.6f} "
          f"own-rate sd={common_p['mean'].std():.6f} "
          f"corr={common_p['mean'].corr(common_p.neigh):+.3f}")
    print(f"batter neighbor-rate sd={common_b.neigh.std():.6f} "
          f"own-rate sd={common_b['mean'].std():.6f} "
          f"corr={common_b['mean'].corr(common_b.neigh):+.3f}")
    print(f"pitcher-neighbor Jaccard uniform={q3(np.asarray(jac))} "
          f"target-row-weighted={q3(np.asarray(jac_w))}")
    active_deg = cur.loc[cur.pitcher_id.isin(pdeg.index), "pitcher_id"].map(pdeg)
    print(f"target-row weighted pitcher degree q10/50/90={q(active_deg.to_numpy())}")
    for lg, g in cur.groupby("game_type"):
        gi = pd.MultiIndex.from_frame(g[["pitcher_id", "batter_id"]])
        print(f"  league {lg}: rows={len(g):,} pitcher_hit={g.pitcher_id.isin(p_seen).mean():.1%} "
              f"pair_hit={gi.isin(edge_idx).mean():.1%}")

    # random slope 표본량: 동적 계층 로짓의 후보 축이 얼마나 관측되는가.
    pc = past.groupby(["pitcher_id", "balls_before", "strikes_before"]).size()
    ph = past.groupby(["pitcher_id", "batter_hand"]).size()
    ps = past.groupby("pitcher_id").size()
    print(f"group n q10/50/90 pitcher={q(ps.to_numpy())} "
          f"pitcher×count={q(pc.to_numpy())} pitcher×hand={q(ph.to_numpy())}")


def main() -> int:
    use = ["season", "game_type", "pitcher_id", "batter_id", "batter_hand",
           "balls_before", "strikes_before", T]
    df = pd.read_csv(DATA, encoding="utf-8-sig", usecols=use)
    for season in (2023, 2024):
        audit(df, season)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
