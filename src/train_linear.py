"""전 피처 로지스틱 회귀 — 블렌드 멤버용 (E125).

왜 필요한가.
  블렌드 기여 조건을 풀면    Dmax = 1e5 A / base,  이득 = margin^2 / (4 Dmax)
  이고 이득 상한이 Dmax/4 다. 즉 **+20 을 벌려면 rms >= 0.020 이면서 D <= 50**.
  CatBoost 하이퍼파라미터 변주는 rms 0.0025~0.0036 (Dmax 2.5~5.2) 이라
  원리적으로 불가능하다. 필요한 건 "약하지만 다양한 모델"이 아니라
  **동급 성능 + 근본적으로 다른 함수**다.

  그런데 이 코드베이스에는 전 피처 선형모델이 **하나도 없다**.
  skill.py 의 능형회귀는 10여 열짜리 부분모델이고, 그것조차
  '남은 시즌 성공률' 설명력에서 선형 59.0% vs GBDT 46.5% 로 트리를 이겼다.
  트리는 축 정렬 계단 함수의 합이라 매끄러운 가중평균을 계단으로만 근사한다.
  로짓 가법 모델은 그 지점에서 **계통적으로** 다른 오차를 낸다 (= A_corr).

  ⚠️ 잡음으로 A 를 키우면 안 된다. A = A_corr + A_noise 인데 margin 은
     A_corr 만 집고 A_noise 는 분모만 키워 이득을 **줄인다**.

전처리 (트리와 다르게 가야 하는 부분)
  - 결측: 중앙값으로 채우되 **결측 지시자 열을 같이 넣는다.** 이 프로젝트는
    "결측을 채우면 -3.51" 을 실증했는데, 그건 CatBoost 가 결측을 정보로 쓰기
    때문이다. 선형모델은 결측을 못 먹으므로 채우되 정보는 지시자로 보존한다.
  - 수치: 분위수 변환(정규) 후 표준화. 72개 중 33개가 두꺼운 꼬리라 원값을
    그대로 넣으면 소수 행이 계수를 지배한다.
  - 범주: 원핫 (카디널리티가 전부 낮다).

실행: python src/train_linear.py --tag L1 --seeds 42,7,13
"""

import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch

import fpipe
from train_gbdt2 import CAT_COLS, TARGET, bss, load


def build_matrix(train, features, is_fit):
    """설계행렬 (수치 분위수화 + 결측지시자 + 범주 원핫)."""
    from sklearn.preprocessing import QuantileTransformer

    num = [c for c in features if c not in CAT_COLS]
    cat = [c for c in features if c in CAT_COLS]
    X = train[num].to_numpy(np.float32)
    miss = ~np.isfinite(X)
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan)[is_fit], axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    X = np.where(miss, med, X)

    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             subsample=200_000, random_state=0)
    qt.fit(X[is_fit])
    X = qt.transform(X).astype(np.float32)

    # 결측률이 의미 있는 열만 지시자를 남긴다 (전부 남기면 상수 열이 생긴다)
    keep = miss.mean(0) > 1e-4
    parts = [X, miss[:, keep].astype(np.float32)]
    for c in cat:
        v = train[c].astype(str)
        cats = sorted(v[is_fit].unique())
        oh = np.zeros((len(v), len(cats)), np.float32)
        idx = v.map({k: i for i, k in enumerate(cats)}).to_numpy()
        ok = pd.notna(idx)
        oh[np.arange(len(v))[ok], idx[ok].astype(int)] = 1.0
        parts.append(oh)
    M = np.concatenate(parts, 1)
    print(f"설계행렬 {M.shape} (수치 {len(num)} + 결측지시자 {int(keep.sum())} "
          f"+ 원핫 {M.shape[1] - X.shape[1] - int(keep.sum())})")
    return M


def fit_logistic(M, y, tr_idx, va_idx, seed, epochs, lr, wd):
    """전체배치 L-BFGS. 볼록 문제이므로 SGD 로 대충 도는 건 낭비이고, 실제로
    Adam 8에폭은 시드 간 90점이 벌어졌다(= 수렴 안 함). 볼록이라 시드 의존도
    거의 없어야 정상이고, 남는 편차는 곧 미수렴 신호다."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(M[tr_idx]).to(dev)
    ytr = torch.from_numpy(y[tr_idx].astype(np.float32)).to(dev)
    Xva = torch.from_numpy(M[va_idx]).to(dev)
    w = torch.zeros(M.shape[1], 1, device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=epochs, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    lossf = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = lossf((Xtr @ w).squeeze(1) + b, ytr) + wd * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        final = lossf((Xtr @ w).squeeze(1) + b, ytr).item()
        p = torch.sigmoid((Xva @ w).squeeze(1) + b).cpu().numpy()
    print(f"  학습 Logloss {final:.6f} | |w| {float(w.norm()):.3f}")
    return p.astype(np.float64), (w.detach().cpu().numpy(), b.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lin")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--val-season", type=int, default=2024)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-6)
    # fpipe.fit 이 보는 플래그들 — CatBoost 와 **같은 피처**를 쓴다
    for f, d in (("feat_v2", True), ("feat_std", True), ("te_dev", True),
                 ("feat_domain", True), ("feat_prof", False),
                 ("feat_form", False), ("feat_window", False),
                 ("feat_count", False), ("feat_cross", False),
                 ("feat_skill", False), ("feat_skill_pc", True)):
        ap.add_argument(f"--{f.replace('_', '-')}", action="store_true", default=d)
    ap.add_argument("--te", default="p,pc,ph,b,pi")
    ap.add_argument("--te-k", type=float, default=50.0)
    ap.add_argument("--te-halflife", type=float, default=0.0)
    ap.add_argument("--te-flat", action="store_true")
    ap.add_argument("--feat-k", type=float, default=50.0)
    ap.add_argument("--std-k", type=float, default=80.0)
    ap.add_argument("--std-to-prior", action="store_true", default=True)
    ap.add_argument("--std-season-prior", action="store_true", default=True)
    ap.add_argument("--std-multi-k", default="")
    ap.add_argument("--std-excess", action="store_true")
    ap.add_argument("--std-ratio", action="store_true")
    ap.add_argument("--std-k-mix", type=float, default=0.0)
    ap.add_argument("--std-k-bat", type=float, default=0.0)
    ap.add_argument("--prof-k", type=float, default=60.0)
    ap.add_argument("--prof-lags", type=int, default=2)
    ap.add_argument("--skill-axes", default="")
    args = ap.parse_args()

    train, features, tm = load()
    is_val = (train["season"] == args.val_season).to_numpy()
    is_fit = ~is_val
    train, new_cols, new_cats, art = fpipe.fit(train, args,
                                               pd.Series(is_fit, index=train.index),
                                               tm)
    features = features + [c for c in new_cols if c not in features]
    CAT_COLS[:] = [c for c in CAT_COLS + new_cats if c in features]
    print(f"피처 총 {len(features)}개 (범주형 {len(CAT_COLS)})")

    M = build_matrix(train, features, is_fit)
    y = train[TARGET].to_numpy(np.float64)
    tr_idx = np.flatnonzero(is_fit)
    va_idx = np.flatnonzero(is_val)
    os.makedirs("./out", exist_ok=True)
    os.makedirs("./model", exist_ok=True)

    for sd in [int(x) for x in args.seeds.split(",") if x.strip()]:
        t0 = time.time()
        p, wb = fit_logistic(M, y, tr_idx, va_idx, sd, args.epochs,
                             args.lr, args.wd)
        tag = f"{args.tag}_s{sd}"
        print(f"[lin {tag}] val{args.val_season} BSS "
              f"{bss(y[va_idx], p):.2f} | 예측평균 {p.mean():.4f} "
              f"| {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(f"./out/lin_{tag}_val_preds.npz",
                            y=y[va_idx], pred=p)
        joblib.dump({"w": wb[0], "b": wb[1], "features": features,
                     "cat_cols": list(CAT_COLS), "fpipe": art},
                    f"./model/lin_{tag}.pkl", compress=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
