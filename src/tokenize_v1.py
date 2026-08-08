"""이산 토큰화 — 47피처 → 47개 토큰 ID. 노트북(sklearn 1.8.0)에서 실행.

설계(docs/discrete_tokenization.md):
  - 연속형: train 분위 N_BINS 경계로 이산화 (경계는 train만으로 산출)
  - 범주형/저카디널리티: 값 → 인덱스
  - 결측: 컬럼별 전용 <NA> 토큰 (cold-start 정보 보존)
  - 컬럼 순서 고정 → 컬럼명 토큰 불필요. 각 컬럼은 자기 토큰 구간을 가짐(offset)

산출: data/processed/tok_v1.npz (X int32 [N,47], y, season) + model/tok_v1.pkl
"""

import os

import joblib
import numpy as np
import pandas as pd

DATA = "./data"
ID, TARGET = "row_id", "control_success"
N_BINS = 32
MAX_CAT = 64          # 이 이하 카디널리티는 값 그대로 토큰화
BIG_CAT = ["pitcher_id", "batter_id"]  # 전용 토큰 부여

# ⚠️ 순서형(ordinal) 컬럼: 미지 값은 <NA>가 아니라 **가장 가까운 경계 카테고리**로 보낸다.
#    season은 어휘를 2019~2023으로 짓기 때문에, 그냥 두면 2024/2025가 전부 <NA>가 되어
#    모델이 평가 시즌의 기저율을 전혀 못 잡는다 (E42 DTT 발산의 실제 원인).
#    트리가 season>2023을 마지막 리프로 보내는 것과 같은 동작을 재현한다.
ORDINAL = ["season"]


def build_vocab(df, features):
    """컬럼별 토큰 사전 구축. 반환: specs(list), vocab_size"""
    specs, offset = [], 0
    for c in features:
        s = df[c]
        if c in BIG_CAT or s.dtype == object or s.nunique(dropna=True) <= MAX_CAT:
            cats = sorted(s.dropna().unique().tolist())
            n_tok = len(cats) + 1                      # +1 = <NA>/unknown
            specs.append({"col": c, "kind": "cat", "cats": cats, "offset": offset,
                          "n_tok": n_tok})
        else:
            qs = np.linspace(0, 1, N_BINS + 1)[1:-1]
            edges = np.unique(np.quantile(s.dropna(), qs))
            n_tok = len(edges) + 2                     # 구간 수 + <NA>
            specs.append({"col": c, "kind": "num", "edges": edges, "offset": offset,
                          "n_tok": n_tok})
        offset += specs[-1]["n_tok"]
    return specs, offset


def encode(df, specs):
    """각 컬럼을 전역 토큰 ID로 변환. 0번은 각 컬럼의 <NA>."""
    out = np.empty((len(df), len(specs)), dtype=np.int32)
    for j, sp in enumerate(specs):
        s = df[sp["col"]]
        if sp["kind"] == "cat":
            lut = {v: i + 1 for i, v in enumerate(sp["cats"])}   # 0 = NA/unknown
            code = s.map(lut)
            if sp["col"] in ORDINAL and code.isna().any():
                # 미지 값 → 학습 범위의 가장 가까운 끝 카테고리로 클램프
                lo, hi = sp["cats"][0], sp["cats"][-1]
                clamped = s.clip(lower=lo, upper=hi).map(lut)
                code = code.fillna(clamped)
            code = code.fillna(0).to_numpy(np.int32)
        else:
            code = np.searchsorted(sp["edges"], s.to_numpy(np.float64),
                                   side="right").astype(np.int32) + 1
            code[s.isna().to_numpy()] = 0
        out[:, j] = code + sp["offset"]
    return out


def main():
    test_cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID]
    train = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                        usecols=features + [TARGET])

    # 어휘는 검증 누수를 피하려 2019~2023으로만 구축
    fit_mask = train["season"] < 2024
    specs, vocab = build_vocab(train.loc[fit_mask], features)
    print(f"컬럼 {len(features)}개 | 어휘 {vocab} | 시퀀스 길이 {len(features)}")
    for sp in specs[:3]:
        print(f"  {sp['col']:28s} {sp['kind']:3s} tok={sp['n_tok']:4d} off={sp['offset']}")

    X = encode(train, specs)
    os.makedirs(f"{DATA}/processed", exist_ok=True)
    np.savez_compressed(f"{DATA}/processed/tok_v1.npz", X=X,
                        y=train[TARGET].to_numpy(np.int8),
                        season=train["season"].to_numpy(np.int16))
    joblib.dump({"specs": specs, "vocab": vocab, "features": features},
                "./model/tok_v1.pkl", compress=3)
    print(f"저장: tok_v1.npz {X.shape} | 값 범위 [{X.min()}, {X.max()}]")


if __name__ == "__main__":
    main()
