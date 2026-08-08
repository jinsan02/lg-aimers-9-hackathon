"""피처 감사 — 스케일 / 결측 / 제거 후보를 한 번에 (EDA 재검토).

세 질문에 답한다.

① 스케일링이 필요한가
   **GBDT 에는 무의미하다** (트리는 단조변환 불변). 지금 주력이 CatBoost 이므로
   여기서 나오는 스케일 문제는 전부 **NN 용**이다. NN 이 GBDT 를 못 따라잡는
   전형적 원인이 두꺼운 꼬리와 0 편중이며, 그건 표준화로는 안 고쳐지고
   분위수 변환/구간 임베딩이 필요하다. 그래서 왜도·첨도·0비율을 같이 본다.

② 결측 처리가 끝났는가
   원본은 prev_*(1~5%)뿐이고 CatBoost 가 네이티브로 처리한다. 문제는 파생 피처인데,
   결측률이 **시즌마다 다르면** 트리가 결측을 시즌 표지로 오용한다(E115 가 그렇게 졌다).

③ 제거할 열이 있는가
   두 종류를 본다.
     중복  : |r| > 0.98 인 쌍 (한쪽만 남기면 된다)
     불안정: 시즌마다 타깃 상관 **부호가 뒤집히는** 열 = 다음 시즌으로 이전 안 됨
     드리프트: 학습 구간과 2024 의 분포가 크게 다른 열 (표준화 평균차로 측정)

메모리: 노트북 말고 4070/A100 에서 돌릴 것.
실행: python tools/feature_audit.py
"""

import sys

import numpy as np
import pandas as pd

DATA = "./data"
TARGET = "control_success"


def main():
    sys.path.insert(0, "src")
    tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
    test_cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    # 파생 피처까지 포함해서 본다 (실제 모델이 쓰는 것 기준)
    import season_std as ss
    anchors = ss.build_anchors(tr)
    pri = {c: float(tr[c].mean()) for c in tr.columns if c.startswith("asof_")}
    sp = ss.season_priors(tr)
    tr, std_cols = ss.add_std(tr, anchors, k=80.0, priors=pri,
                              to_career=False, season_prior=sp)
    tr, dom_cols = ss.add_domain(tr)

    feats = [c for c in test_cols if c != "row_id"] + std_cols + dom_cols
    feats = [c for c in feats if c in tr.columns
             and pd.api.types.is_numeric_dtype(tr[c])]
    y = tr[TARGET].to_numpy(np.float64)
    se = tr["season"].to_numpy()
    val = se == 2024
    print(f"train {len(tr):,}행 | 수치 피처 {len(feats)}개\n")

    rows = []
    for c in feats:
        v = tr[c].to_numpy(np.float64)
        m = np.isfinite(v)
        vv = v[m]
        if len(vv) < 1000 or vv.std() < 1e-12:
            continue
        sk = float(((vv - vv.mean()) ** 3).mean() / (vv.std() ** 3 + 1e-12))
        # 꼬리 두께: (99.9분위 - 중앙값) / (75분위 - 중앙값)
        q = np.percentile(vv, [50, 75, 99.9])
        tail = (q[2] - q[0]) / max(q[1] - q[0], 1e-9)
        zero = float((vv == 0).mean())
        # 드리프트: 2024 평균 - 학습구간 평균, 학습구간 표준편차로 정규화
        a, b = v[m & ~val], v[m & val]
        drift = (b.mean() - a.mean()) / (a.std() + 1e-12) if len(b) else 0.0
        # 시즌별 타깃 상관 부호
        signs = []
        for s in sorted(set(se)):
            k = m & (se == s)
            if k.sum() > 5000 and v[k].std() > 1e-12:
                signs.append(np.sign(np.corrcoef(v[k], y[k])[0, 1]))
        flip = len(set(signs)) > 1
        rows.append(dict(col=c, na=1 - m.mean(), sk=sk, tail=tail,
                         zero=zero, drift=drift, flip=flip,
                         rng=vv.max() - vv.min()))
    df = pd.DataFrame(rows)

    print("=== ① 스케일 — NN 이 어려워하는 열 (왜도 |>3| 또는 꼬리 >6) ===")
    bad = df[(df["sk"].abs() > 3) | (df["tail"] > 6)].sort_values(
        "tail", ascending=False)
    print(f"{'열':<40}{'왜도':>9}{'꼬리':>9}{'0비율':>9}{'범위':>12}")
    for _, r in bad.head(18).iterrows():
        print(f"{r['col']:<40}{r['sk']:>9.2f}{r['tail']:>9.1f}"
              f"{r['zero']:>9.1%}{r['rng']:>12.1f}")
    print(f"  해당 {len(bad)}/{len(df)}개. **GBDT 에는 무해**하고 NN 에만 문제다.")
    print("  -> NN 은 표준화 대신 분위수 구간 임베딩을 써야 한다.")

    print("\n=== ② 결측 — 시즌마다 결측률이 다른 열 ===")
    nac = [c for c in feats if tr[c].isna().any()]
    if not nac:
        print("  없음")
    else:
        t = tr.groupby("season")[nac].apply(lambda d: d.isna().mean() * 100)
        rangeby = (t.max() - t.min()).sort_values(ascending=False)
        for c, v in rangeby.head(10).items():
            print(f"  {c:<44} 시즌간 결측률 편차 {v:5.2f}%p "
                  f"({t[c].min():.2f}~{t[c].max():.2f})")
        print("  -> 편차가 크면 트리가 결측을 시즌 표지로 쓴다 (E115 실패 원인).")

    print("\n=== ③-a 제거 후보: 중복 (|r| > 0.98) ===")
    sub = tr[feats].sample(min(200000, len(tr)), random_state=0)
    C = sub.corr().to_numpy()
    names = list(sub.columns)
    seen = set()
    n = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(C[i, j]) > 0.98 and not np.isnan(C[i, j]):
                print(f"  {names[i]:<38} <-> {names[j]:<38} r={C[i, j]:+.4f}")
                seen.add(names[j])
                n += 1
    print(f"  중복쌍 {n}개 -> 제거 후보 {len(seen)}개")

    print("\n=== ③-b 제거 후보: 시즌마다 타깃 상관 부호가 뒤집히는 열 ===")
    fl = df[df["flip"]].sort_values("drift", key=abs, ascending=False)
    print(f"  {len(fl)}/{len(df)}개")
    print("  " + ", ".join(fl["col"].head(20)))

    print("\n=== ③-c 분포 드리프트 상위 (2024 평균 - 학습평균, 학습 sd 단위) ===")
    dr = df.reindex(df["drift"].abs().sort_values(ascending=False).index)
    for _, r in dr.head(12).iterrows():
        print(f"  {r['col']:<44}{r['drift']:+8.3f} sd")
    print("  -> |drift| > 0.5 이면 그 열의 분할 기준이 2025 로 이전되기 어렵다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
