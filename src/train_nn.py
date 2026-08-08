"""MLP / TabM 스타일 NN — **블렌드 다양성 확보용** (계획 ①).

왜 NN 인가 (tools/blend_value.py 계산):
  블렌드 이득은 후보의 성능 격차 D 와 예측 불일치 A=E[(p2-p1)^2] 로 결정되고
      w* = (A - delta)/(2A),  이득 = (delta - A)^2/(4A),  delta = D*base/1e5
  즉 **delta >= A 면 기여가 0** 이다. 우리 CatBoost 변형끼리는 rms 불일치가
  0.0024~0.0039 밖에 안 돼 기여 상한이 2~6점이다(실측). 설정을 더 늘려도 소용없다.
  반면 NN 은 GBDT 와 통상 rms 0.02~0.03 다르게 예측하므로, **866점(-60)만 나와도
  +15~63** 이 나온다. 그래서 성능 자체보다 '충분히 다르게 틀리는가'가 목표다.

설계
  - 피처는 train_gbdt2 --dump-npz 로 받은 **CatBoost와 동일한 행렬**을 쓴다.
    파이프라인을 두 번 구현하면 반드시 어긋난다.
  - 수치는 분위수 클리핑 + 표준화, 결측은 학습구간 중앙값. 통계는 **학습 구간만**으로.
  - 범주 7개는 임베딩. 카디널리티가 작아(팀 10여개) 차원 4~8이면 충분.
  - 손실은 MSE(=Brier 직접) 와 BCE 중 선택. 지표가 Brier 이므로 기본 MSE.
  - TabM 스타일: 공유 트렁크 + k개 헤드를 각자 다른 초기화로 두고 평균.
    한 번 학습으로 앙상블 효과를 얻어 시드를 덜 돌려도 된다.

실행:
  python src/train_gbdt2.py --model cat --dump-npz out/nn_v1.npz <피처 플래그들>
  python src/train_nn.py --npz out/nn_v1.npz --tag nn1 --seed 42
"""

import argparse
import os
import sys
import time

import joblib
import numpy as np
import torch
import torch.nn as nn


def bss(y, p, base):
    return float(100000 * (1 - ((p - y) ** 2).mean() / base))


class TabMLP(nn.Module):
    """공유 트렁크 + k개 헤드 (TabM 스타일 경량판)."""

    def __init__(self, n_num, cards, emb=6, hidden=(512, 256, 128),
                 heads=4, drop=0.1, norm="bn", qbins=0, qdim=4):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, min(emb, max(2, c // 2)))
                                   for c in cards])
        # 분위수 구간 임베딩 (tools/feature_audit.py: 72개 중 33개가 두꺼운 꼬리).
        # 트리는 각 피처를 **구간으로 쪼개** 쓰는데 MLP 는 선형 결합만 한다.
        # 표준화로는 꼬리가 안 고쳐지므로 구간 인덱스를 따로 임베딩해 준다.
        self.qbins = qbins
        if qbins:
            self.qemb = nn.Embedding(n_num * qbins, qdim)
            nn.init.normal_(self.qemb.weight, std=0.05)
            self.register_buffer("qoff",
                                 torch.arange(n_num).view(1, -1) * qbins)
        d = n_num + sum(e.embedding_dim for e in self.embs)             + (n_num * qdim if qbins else 0)
        layers = []
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if norm == "bn":
                layers.append(nn.BatchNorm1d(h))
            elif norm == "ln":
                layers.append(nn.LayerNorm(h))
            layers += [nn.SiLU(), nn.Dropout(drop)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(heads)])

    def forward(self, xn, xc, xq=None):
        parts = [xn] + [e(xc[:, i]) for i, e in enumerate(self.embs)]
        if self.qbins and xq is not None:
            parts.append(self.qemb(xq + self.qoff).flatten(1))
        h = self.trunk(torch.cat(parts, 1))
        return torch.cat([hd(h) for hd in self.heads], 1)      # (B, heads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tag", default="nn1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hidden", default="512,256,128")
    ap.add_argument("--loss", default="mse", choices=["mse", "bce"])
    ap.add_argument("--norm", default="bn", choices=["bn", "ln", "none"],
                    help="BatchNorm 은 시즌 간 분포 이동에 취약할 수 있다")
    ap.add_argument("--sched", default="onecycle",
                    choices=["onecycle", "cosine", "const"])
    ap.add_argument("--qbins", type=int, default=0,
                    help="분위수 구간 임베딩 구간 수 (0=미사용). 32~64 권장")
    ap.add_argument("--qdim", type=int, default=4)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(args.npz)
    Xn, Xc, y, is_val = z["Xn"], z["Xc"], z["y"], z["is_val"]
    tr, va = ~is_val, is_val
    print(f"학습 {tr.sum():,} | 검증 {va.sum():,} | 수치 {Xn.shape[1]} "
          f"범주 {Xc.shape[1]}")

    # --- 전처리 통계는 학습 구간만으로 (검증 누수 차단)
    med = np.nanmedian(Xn[tr], 0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    Xn = np.where(np.isfinite(Xn), Xn, med)
    lo = np.percentile(Xn[tr], 0.5, axis=0).astype(np.float32)
    hi = np.percentile(Xn[tr], 99.5, axis=0).astype(np.float32)
    Xn = np.clip(Xn, lo, hi)
    mu = Xn[tr].mean(0)
    sd = Xn[tr].std(0)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    Xn = ((Xn - mu) / sd).astype(np.float32)
    cards = [int(Xc[:, i].max()) + 1 for i in range(Xc.shape[1])]

    Xq_t = None
    if args.qbins:
        # 구간 경계도 **학습 구간만**으로 (검증 누수 차단)
        qs = np.linspace(0, 1, args.qbins + 1)[1:-1]
        edges = np.quantile(Xn[tr], qs, axis=0)
        Xq = np.empty(Xn.shape, np.int16)
        for j in range(Xn.shape[1]):
            Xq[:, j] = np.searchsorted(edges[:, j], Xn[:, j]).astype(np.int16)
        Xq_t = torch.from_numpy(Xq)
        print(f"  분위수 임베딩 {args.qbins}구간 x {args.qdim}차원 "
              f"-> +{Xn.shape[1] * args.qdim}차원")

    Xn_t = torch.from_numpy(Xn)
    Xc_t = torch.from_numpy(Xc.astype(np.int64))
    y_t = torch.from_numpy(y)
    itr = np.where(tr)[0]
    iva = torch.from_numpy(np.where(va)[0])
    y_va = y[va].astype(np.float64)
    r = y_va.mean()
    base = r * (1 - r)

    model = TabMLP(Xn.shape[1], cards, hidden=tuple(int(x) for x in
                                                    args.hidden.split(",")),
                   heads=args.heads, drop=args.drop, norm=args.norm,
                   qbins=args.qbins, qdim=args.qdim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    steps = args.epochs * (len(itr) // args.batch + 1)
    if args.sched == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                    total_steps=steps)
    elif args.sched == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    else:
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0,
                                                    total_iters=1)
    lossf = nn.MSELoss() if args.loss == "mse" else nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def predict(idx):
        model.eval()
        out = []
        for i in range(0, len(idx), 65536):
            b = idx[i:i + 65536]
            # 헤드별로 시그모이드를 먼저 씌우고 **확률을 평균**한다.
            # (학습 손실도 sigmoid(o) 에 걸리므로 여기서 빼먹으면 로짓이 그대로
            #  [0,1] 로 클리핑돼 예측이 붕괴한다 - 실제로 val BSS -58,276 이 났다)
            xq = Xq_t[b].to(dev).long() if Xq_t is not None else None
            o = torch.sigmoid(model(Xn_t[b].to(dev), Xc_t[b].to(dev),
                                    xq)).mean(1)
            out.append(o.float().cpu().numpy())
        return np.clip(np.concatenate(out), 0.0, 1.0).astype(np.float64)

    best, best_p, bad = -1e9, None, 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(itr)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = torch.from_numpy(perm[i:i + args.batch])
            xb_n, xb_c = Xn_t[b].to(dev), Xc_t[b].to(dev)
            yb = y_t[b].to(dev).unsqueeze(1).expand(-1, args.heads)
            xb_q = Xq_t[b].to(dev).long() if Xq_t is not None else None
            o = model(xb_n, xb_c, xb_q)
            if args.loss == "mse":
                o = torch.sigmoid(o)      # predict() 와 반드시 동일해야 한다
            loss = lossf(o, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
        p = predict(iva)
        s = bss(y_va, p, base)
        mark = ""
        if s > best:
            best, best_p, bad, mark = s, p, 0, "  *"
            torch.save({"sd": model.state_dict(), "cards": cards,
                        "n_num": Xn.shape[1], "hidden": args.hidden,
                        "heads": args.heads, "drop": args.drop,
                        "loss": args.loss, "qbins": args.qbins,
                        "qdim": args.qdim,
                        "med": med, "lo": lo, "hi": hi,
                        "mu": mu.astype(np.float32), "sd": sd},
                       f"./model/nn_{args.tag}.pt")
        else:
            bad += 1
        print(f"  ep{ep + 1:3d} loss {tot / len(perm):.5f} | val BSS {s:8.2f}{mark}")
        if bad >= args.patience:
            print(f"  조기종료 (patience {args.patience})")
            break

    os.makedirs("./out", exist_ok=True)
    np.savez_compressed(f"./out/nn_{args.tag}_val_preds.npz",
                        y=y_va, pred=best_p)
    print(f"[nn {args.tag}] val BSS {best:.2f} | {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    os.makedirs("./model", exist_ok=True)
    sys.exit(main())
