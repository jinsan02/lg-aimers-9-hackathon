"""다중과제 NN (E140) — 블렌드 멤버용.

## 왜 이걸 만드는가 (수치로)

블렌드 기여식:  Dmax = 400,320 x rms^2,  margin = Dmax - D,  이득 = margin^2/(4 Dmax)

지금까지 후보들:
  CatBoost 변주      rms 0.003  -> 최대 이득 0.9    (실측 ~0)
  절제 멤버          rms 0.006  -> 최대 3.6         (실측 +1.9)
  셀 다중분류 d5     rms 0.011  -> 최대 12.1        (실측 +10.75) <- 천장 근접
  피처 부분공간 0.5  rms 0.020  -> 최대 40 이지만 D 210 -> margin 음수 (기각)
  **이전 NN**       rms 0.023  -> 최대 53 인데 D 219 -> margin -7.2 (기여 0)

즉 CatBoost 계열은 rms 0.011 에서 이미 천장을 쳤다. 남은 이득은 NN 에만 있다.
그런데 **NN 을 더 다르게 만들 필요가 없다 — 더 좋게만 만들면 된다**:

    점수 711 (이전), rms 0.023 -> margin -7    기여 0
    점수 800,        rms 0.023 -> margin +82   이득  7.9
    점수 850,        rms 0.023 -> margin +132  이득 20.4
    점수 880,        rms 0.023 -> margin +162  이득 30.8

## 이전 NN 이 711 에 그친 이유 (진단)

CatBoost 와 **똑같은 행렬**(TE 포함 121열)을 먹였다. TE 는 이미 '투수 x 상황'의
수축 추정치라 신호를 다 담고 있고, 그 위에 MLP 를 얹으면 같은 함수를 재구성만 한다.
게다가 학습 설정이 GBDT 감각으로 맞춰져 있었다.

## 이번 설계

  ① **보조 라벨 4개를 헤드로 단다** (실패모드 복원, E124).
     한 행에 이진 타깃 1개가 아니라 상관된 라벨 4개가 있다 — 이 대회에서 우리가
     가진 유일한 '남은 감독 신호'다. 보조 과제는 트렁크 표현을 정규화한다.
     ⚠️ 라벨은 train 행끼리만 만든다. 추론은 성공 헤드 하나만 쓴다.
  ② 분위수 구간 임베딩 — 수치 72개 중 33개가 두꺼운 꼬리다. 트리는 구간으로
     쪼개 쓰지만 MLP 는 선형결합만 한다. 구간 인덱스를 따로 임베딩해 보정.
  ③ 코사인 스케줄 + 충분한 에폭. 이전엔 조기종료가 너무 빨리 걸렸다.

실행: python src/train_mtnn.py --npz out/mt.npz --tag M1 --seeds 42,7
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn


def bss(y, p):
    r = y.mean()
    return float(1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / (r * (1 - r))))


class MTNet(nn.Module):
    """공유 트렁크 + (주 1 + 보조 k) 헤드."""

    def __init__(self, n_num, cards, n_aux, n_pitch=0, emb=8,
                 hidden=(768, 384, 192), drop=0.15, qbins=32, qdim=4):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, min(emb, max(2, c // 2)))
                                   for c in cards])
        self.qbins = qbins
        if qbins:
            self.qemb = nn.Embedding(n_num * qbins, qdim)
            nn.init.normal_(self.qemb.weight, std=0.05)
            self.register_buffer("qoff", torch.arange(n_num).view(1, -1) * qbins)
        d = (n_num + sum(e.embedding_dim for e in self.embs)
             + (n_num * qdim if qbins else 0))
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(),
                       nn.Dropout(drop)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.main = nn.Linear(d, 1)
        self.aux = nn.Linear(d, n_aux) if n_aux else None
        self.pitch = nn.Linear(d, n_pitch) if n_pitch else None

    def forward(self, xn, xc, xq):
        parts = [xn] + [e(xc[:, i]) for i, e in enumerate(self.embs)]
        if self.qbins:
            parts.append(self.qemb(xq + self.qoff).flatten(1))
        h = self.trunk(torch.cat(parts, 1))
        return (self.main(h).squeeze(1),
                self.aux(h) if self.aux is not None else None,
                self.pitch(h) if self.pitch is not None else None)


class PLRNet(nn.Module):
    """Per-feature periodic-linear-ReLU embeddings followed by a small MLP."""

    def __init__(self, n_num, cards, n_aux=0, n_pitch=0, freq=32, dim=8,
                 sigma=0.1, hidden=(512, 256), drop=0.15):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_num, freq) * sigma)
        self.phase = nn.Parameter(torch.rand(n_num, freq))
        self.mix = nn.Parameter(torch.randn(n_num, freq, dim) / np.sqrt(freq))
        self.mix_bias = nn.Parameter(torch.zeros(n_num, dim))
        self.embs = nn.ModuleList([nn.Embedding(c, min(8, max(2, c // 2)))
                                   for c in cards])
        width = n_num * dim + sum(e.embedding_dim for e in self.embs)
        layers = []
        for h in hidden:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.SiLU(),
                       nn.Dropout(drop)]
            width = h
        self.trunk = nn.Sequential(*layers)
        self.main = nn.Linear(width, 1)
        self.aux = nn.Linear(width, n_aux) if n_aux else None
        self.pitch = nn.Linear(width, n_pitch) if n_pitch else None

    def forward(self, xn, xc, xq=None):
        z = torch.cos(2 * np.pi * (xn.unsqueeze(-1) * self.freq + self.phase))
        z = torch.relu(torch.einsum("bnf,nfd->bnd", z, self.mix)
                       + self.mix_bias).flatten(1)
        parts = [z] + [e(xc[:, i]) for i, e in enumerate(self.embs)]
        h = self.trunk(torch.cat(parts, 1))
        return (self.main(h).squeeze(1),
                self.aux(h) if self.aux is not None else None,
                self.pitch(h) if self.pitch is not None else None)


class PitchMoENet(nn.Module):
    """Legal latent pitch-type mixture; inference returns only control P.

    The pitch head estimates q(type|row-local x).  Three type-conditional
    command heads share a global control logit and may use the recovered pitch
    label only in their *training loss*.  Inference always marginalizes over q.
    """

    def __init__(self, n_num, cards, emb=8, hidden=(768, 384, 192),
                 drop=0.15, qbins=32, qdim=4, delta_scale=0.5):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, min(emb, max(2, c // 2)))
                                   for c in cards])
        self.qbins = qbins
        self.delta_scale = delta_scale
        if qbins:
            self.qemb = nn.Embedding(n_num * qbins, qdim)
            nn.init.normal_(self.qemb.weight, std=0.05)
            self.register_buffer("qoff", torch.arange(n_num).view(1, -1) * qbins)
        d = (n_num + sum(e.embedding_dim for e in self.embs)
             + (n_num * qdim if qbins else 0))
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(),
                       nn.Dropout(drop)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.global_control = nn.Linear(d, 1)
        self.type_delta = nn.Linear(d, 3)
        self.pitch = nn.Linear(d, 3)

    def forward(self, xn, xc, xq):
        parts = [xn] + [e(xc[:, i]) for i, e in enumerate(self.embs)]
        if self.qbins:
            parts.append(self.qemb(xq + self.qoff).flatten(1))
        h = self.trunk(torch.cat(parts, 1))
        pitch_logits = self.pitch(h)
        q = torch.softmax(pitch_logits, 1)
        # Center residual experts so the global head owns the level.
        delta = torch.tanh(self.type_delta(h)) * self.delta_scale
        delta = delta - (q * delta).sum(1, keepdim=True)
        cond = torch.sigmoid(self.global_control(h) + delta)
        mix = (q * cond).sum(1)
        return mix, pitch_logits, cond


class CellNet(nn.Module):
    """실패모드 셀 softmax. 성공확률은 성공 셀 확률의 합으로만 만든다."""

    def __init__(self, n_num, cards, n_cells, emb=8,
                 hidden=(768, 384, 192), drop=0.15, qbins=32, qdim=4):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, min(emb, max(2, c // 2)))
                                   for c in cards])
        self.qbins = qbins
        if qbins:
            self.qemb = nn.Embedding(n_num * qbins, qdim)
            nn.init.normal_(self.qemb.weight, std=0.05)
            self.register_buffer("qoff", torch.arange(n_num).view(1, -1) * qbins)
        d = (n_num + sum(e.embedding_dim for e in self.embs)
             + (n_num * qdim if qbins else 0))
        layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(),
                       nn.Dropout(drop)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.cell = nn.Linear(d, n_cells)

    def forward(self, xn, xc, xq):
        parts = [xn] + [e(xc[:, i]) for i, e in enumerate(self.embs)]
        if self.qbins:
            parts.append(self.qemb(xq + self.qoff).flatten(1))
        return self.cell(self.trunk(torch.cat(parts, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tag", default="mt")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--qbins", type=int, default=32)
    ap.add_argument("--plr", action="store_true",
                    help="수치 피처별 periodic-linear-ReLU 임베딩 사용")
    ap.add_argument("--plr-freq", type=int, default=32)
    ap.add_argument("--plr-dim", type=int, default=8)
    ap.add_argument("--plr-sigma", type=float, default=0.1)
    ap.add_argument("--aux-w", type=float, default=0.3,
                    help="보조 과제 손실 가중. 0 이면 단일과제 대조군")
    ap.add_argument("--pitch-w", type=float, default=0.0,
                    help="strict 1:1 현재구종 3분류 masked CE 가중")
    ap.add_argument("--pitch-moe", action="store_true",
                    help="P(success|x)=sum_type P(type|x)P(success|x,type) 구조")
    ap.add_argument("--pitch-cond-w", type=float, default=0.1,
                    help="MoE에서 복원된 실제 구종 expert의 조건부 BCE 가중")
    ap.add_argument("--pitch-delta-scale", type=float, default=0.5,
                    help="MoE type expert의 global logit 대비 최대 잔차")
    ap.add_argument("--cell-consistent", action="store_true",
                    help="실패모드 셀 CE와 성공 marginal BCE를 공동 최적화")
    ap.add_argument("--cell-w", type=float, default=0.5,
                    help="cell-consistent 모드의 셀 CE 가중")
    ap.add_argument("--refit-mult", type=float, default=1.5,
                    help="검증 최적 에폭을 전체 과거 시즌 재학습에 적용할 배수")
    ap.add_argument("--fixed-refit-epochs", type=int, default=0,
                    help="0보다 크면 검증 선택과 무관하게 전체 재학습 에폭을 고정")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    Xn, Xc, y = z["Xn"].astype(np.float32), z["Xc"].astype(np.int64), z["y"]
    is_val = z["is_val"]
    is_test = z["is_test"] if "is_test" in z else np.zeros_like(is_val)
    aux = z["aux"].astype(np.float32) if "aux" in z else None
    pitch = z["pitch"].astype(np.int64) if "pitch" in z else None
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 수치: 분위수 정규화(두꺼운 꼬리) + 구간 인덱스. 통계는 학습 구간에서만.
    tr, va, te = ~(is_val | is_test), is_val, is_test
    if args.cell_consistent and not te.any():
        raise ValueError("--cell-consistent 판정은 is_test가 있는 npz가 필요합니다")
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             subsample=300_000, random_state=0)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    qt.fit(Xn[tr])
    Xq = np.clip((qt.transform(Xn) * 4 + args.qbins / 2).astype(np.int64),
                 0, args.qbins - 1) if args.qbins else np.zeros_like(Xc)
    Xn = qt.transform(Xn).astype(np.float32)
    cards = [int(Xc[:, i].max()) + 2 for i in range(Xc.shape[1])]

    n_aux = 0
    A = None
    if aux is not None and args.aux_w > 0:
        m = np.isfinite(aux).all(1)
        A = np.nan_to_num(aux, nan=0.0)
        Am = m.astype(np.float32)          # 복원 실패 행은 보조 손실에서 제외
        n_aux = A.shape[1]
        print(f"보조 라벨 {n_aux}개 | 사용 가능 {m.mean() * 100:.2f}%")

    n_pitch = 3 if (pitch is not None and
                    (args.pitch_w > 0 or args.pitch_moe)) else 0
    if (args.pitch_w > 0 or args.pitch_moe) and pitch is None:
        raise ValueError("구종 보조학습은 --dump-pitch-aux NPZ가 필요합니다")
    if n_pitch:
        print(f"구종 보조라벨 3개 | 사용 가능 {(pitch >= 0).mean() * 100:.2f}% | "
              f"pitch_w {args.pitch_w} | moe {args.pitch_moe}")

    T = lambda a, i: torch.from_numpy(a[i]).to(dev)   # noqa: E731
    ytr, yva = y[tr].astype(np.float32), y[va].astype(np.float64)
    print(f"학습 {tr.sum():,} / 검증 {va.sum():,} | 수치 {Xn.shape[1]} "
          f"범주 {Xc.shape[1]} | aux_w {args.aux_w}")

    cell = z["cell"].astype(np.int64) if args.cell_consistent else None
    succ = z["cell_success"].astype(np.int64) if args.cell_consistent else None
    n_cells = int(cell.max()) + 1 if args.cell_consistent else 0
    if args.cell_consistent:
        print(f"확률 일관성 셀 {n_cells}개 | 성공 셀 {succ.tolist()} | "
              f"test {int(te.sum()):,}행")

    for sd in [int(s) for s in args.seeds.split(",") if s.strip()]:
        torch.manual_seed(sd)
        net = (CellNet(Xn.shape[1], cards, n_cells, drop=args.drop,
                       qbins=args.qbins).to(dev) if args.cell_consistent else
               PitchMoENet(Xn.shape[1], cards, drop=args.drop,
                           qbins=args.qbins,
                           delta_scale=args.pitch_delta_scale).to(dev)
               if args.pitch_moe else
               PLRNet(Xn.shape[1], cards, n_aux, n_pitch,
                      freq=args.plr_freq,
                      dim=args.plr_dim, sigma=args.plr_sigma,
                      drop=args.drop).to(dev) if args.plr else
               MTNet(Xn.shape[1], cards, n_aux, n_pitch, drop=args.drop,
                     qbins=args.qbins).to(dev))
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                weight_decay=args.wd)
        idx = np.flatnonzero(tr)
        steps = args.epochs * (len(idx) // args.bs + 1)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
        bce = nn.BCEWithLogitsLoss()
        bce_n = nn.BCEWithLogitsLoss(reduction="none")
        ce = nn.CrossEntropyLoss()
        Xn_t, Xc_t, Xq_t = T(Xn, tr), T(Xc, tr), T(Xq, tr)
        y_t = torch.from_numpy(ytr).to(dev)
        A_t = torch.from_numpy(A[tr]).to(dev) if n_aux else None
        Am_t = torch.from_numpy(Am[tr]).to(dev) if n_aux else None
        P_t = torch.from_numpy(pitch[tr]).to(dev) if n_pitch else None
        Pm_t = P_t >= 0 if n_pitch else None
        Xn_v, Xc_v, Xq_v = T(Xn, va), T(Xc, va), T(Xq, va)
        C_t = torch.from_numpy(cell[tr]).to(dev) if args.cell_consistent else None
        succ_t = torch.from_numpy(succ).to(dev) if args.cell_consistent else None

        def predict(net_, xn, xc, xq):
            net_.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(xn), 65536):
                    if args.cell_consistent:
                        logits = net_(xn[i:i + 65536], xc[i:i + 65536],
                                      xq[i:i + 65536])
                        o = torch.softmax(logits, 1)[:, succ_t].sum(1)
                    elif args.pitch_moe:
                        o, _, _ = net_(xn[i:i + 65536], xc[i:i + 65536],
                                       xq[i:i + 65536])
                    else:
                        o, _, _ = net_(xn[i:i + 65536], xc[i:i + 65536],
                                       xq[i:i + 65536])
                        o = torch.sigmoid(o)
                    out.append(o.cpu().numpy())
            return np.concatenate(out).astype(np.float64)

        t0 = time.time()
        # **에폭마다 검증하고 최고점을 남긴다.** 이걸 빼먹었다가 BSS -9194 가 나왔다:
        # 25 에폭을 그냥 돌리면 미래 시즌 검증에서 자신있게 틀린 예측을 하게 된다.
        # GBDT 는 early stopping 이 기본이라 이 함정이 안 보였다.
        best, best_p, hist = -1e9, None, []
        for ep in range(args.epochs):
            net.train()
            perm = torch.randperm(len(idx), device=dev)
            for i in range(0, len(idx), args.bs):
                j = perm[i:i + args.bs]
                opt.zero_grad()
                if args.cell_consistent:
                    logits = net(Xn_t[j], Xc_t[j], Xq_t[j])
                    ps = torch.softmax(logits, 1)[:, succ_t].sum(1)
                    loss = nn.functional.binary_cross_entropy(
                        ps.clamp(1e-6, 1 - 1e-6), y_t[j])
                    loss = loss + args.cell_w * ce(logits, C_t[j])
                elif args.pitch_moe:
                    pmix, op, pcond = net(Xn_t[j], Xc_t[j], Xq_t[j])
                    loss = nn.functional.binary_cross_entropy(
                        pmix.clamp(1e-6, 1 - 1e-6), y_t[j])
                    mp = Pm_t[j]
                    if mp.any():
                        loss = loss + args.pitch_w * ce(op[mp], P_t[j][mp])
                        cp = pcond[mp].gather(1, P_t[j][mp, None]).squeeze(1)
                        loss = loss + args.pitch_cond_w * \
                            nn.functional.binary_cross_entropy(
                                cp.clamp(1e-6, 1 - 1e-6), y_t[j][mp])
                else:
                    o, oa, op = net(Xn_t[j], Xc_t[j], Xq_t[j])
                    loss = bce(o, y_t[j])
                if n_aux and not args.cell_consistent and not args.pitch_moe:
                    la = bce_n(oa, A_t[j]).mean(1) * Am_t[j]
                    loss = loss + args.aux_w * la.sum() / Am_t[j].sum().clamp(min=1)
                if n_pitch and not args.cell_consistent and not args.pitch_moe:
                    mp = Pm_t[j]
                    if mp.any():
                        loss = loss + args.pitch_w * ce(op[mp], P_t[j][mp])
                loss.backward()
                opt.step()
                sch.step()
            p = predict(net, Xn_v, Xc_v, Xq_v)
            v = bss(yva, p)
            hist.append(v)
            if v > best:
                best, best_p = v, p
                torch.save({"state": net.state_dict(), "cards": cards,
                            "n_num": Xn.shape[1], "n_aux": n_aux,
                            "n_pitch": n_pitch, "pitch_moe": args.pitch_moe,
                            "pitch_delta_scale": args.pitch_delta_scale,
                            "qbins": args.qbins, "cell_consistent": args.cell_consistent,
                            "plr": args.plr, "plr_freq": args.plr_freq,
                            "plr_dim": args.plr_dim,
                            "plr_sigma": args.plr_sigma,
                            "n_cells": n_cells, "cell_success": succ},
                           f"./model/mtnn_{args.tag}_s{sd}.pt")
        print("  에폭 곡선 " + " ".join(f"{v:.0f}" for v in hist), flush=True)
        p = best_p
        print(f"[mtnn {args.tag}_s{sd}] val BSS {best:.2f} (최고 에폭) | "
              f"예측평균 {p.mean():.4f} | {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(f"./out/mtnn_{args.tag}_s{sd}_val_preds.npz",
                            y=yva, pred=p)

        if te.any():
            # CatBoost 판정 표면과 동일하게 검증으로 에폭을 고른 뒤, 검증 시즌까지
            # 학습에 포함해 새 모델을 처음부터 재학습하고 미학습 test 시즌을 본다.
            refit = tr | va
            refit_epochs = max(1, int(round((int(np.argmax(hist)) + 1)
                                            * args.refit_mult)))
            if args.fixed_refit_epochs > 0:
                refit_epochs = args.fixed_refit_epochs
            torch.manual_seed(sd)
            final = (CellNet(Xn.shape[1], cards, n_cells, drop=args.drop,
                             qbins=args.qbins).to(dev) if args.cell_consistent else
                     PitchMoENet(Xn.shape[1], cards, drop=args.drop,
                                 qbins=args.qbins,
                                 delta_scale=args.pitch_delta_scale).to(dev)
                     if args.pitch_moe else
                     PLRNet(Xn.shape[1], cards, n_aux, n_pitch,
                            freq=args.plr_freq,
                            dim=args.plr_dim, sigma=args.plr_sigma,
                            drop=args.drop).to(dev) if args.plr else
                     MTNet(Xn.shape[1], cards, n_aux, n_pitch, drop=args.drop,
                           qbins=args.qbins).to(dev))
            opt = torch.optim.AdamW(final.parameters(), lr=args.lr,
                                    weight_decay=args.wd)
            ridx = np.flatnonzero(refit)
            rsteps = refit_epochs * (len(ridx) // args.bs + 1)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr,
                                                       total_steps=rsteps)
            Xn_r, Xc_r, Xq_r = T(Xn, refit), T(Xc, refit), T(Xq, refit)
            y_r = torch.from_numpy(y[refit].astype(np.float32)).to(dev)
            A_r = torch.from_numpy(A[refit]).to(dev) if n_aux else None
            Am_r = torch.from_numpy(Am[refit]).to(dev) if n_aux else None
            P_r = torch.from_numpy(pitch[refit]).to(dev) if n_pitch else None
            Pm_r = P_r >= 0 if n_pitch else None
            C_r = (torch.from_numpy(cell[refit]).to(dev)
                   if args.cell_consistent else None)
            final.train()
            for _ in range(refit_epochs):
                perm = torch.randperm(len(ridx), device=dev)
                for i in range(0, len(ridx), args.bs):
                    j = perm[i:i + args.bs]
                    opt.zero_grad()
                    if args.cell_consistent:
                        logits = final(Xn_r[j], Xc_r[j], Xq_r[j])
                        ps = torch.softmax(logits, 1)[:, succ_t].sum(1)
                        loss = nn.functional.binary_cross_entropy(
                            ps.clamp(1e-6, 1 - 1e-6), y_r[j])
                        loss = loss + args.cell_w * ce(logits, C_r[j])
                    elif args.pitch_moe:
                        pmix, op, pcond = final(Xn_r[j], Xc_r[j], Xq_r[j])
                        loss = nn.functional.binary_cross_entropy(
                            pmix.clamp(1e-6, 1 - 1e-6), y_r[j])
                        mp = Pm_r[j]
                        if mp.any():
                            loss = loss + args.pitch_w * ce(op[mp], P_r[j][mp])
                            cp = pcond[mp].gather(
                                1, P_r[j][mp, None]).squeeze(1)
                            loss = loss + args.pitch_cond_w * \
                                nn.functional.binary_cross_entropy(
                                    cp.clamp(1e-6, 1 - 1e-6), y_r[j][mp])
                    else:
                        o, oa, op = final(Xn_r[j], Xc_r[j], Xq_r[j])
                        loss = bce(o, y_r[j])
                        if n_aux:
                            la = bce_n(oa, A_r[j]).mean(1) * Am_r[j]
                            loss = loss + args.aux_w * la.sum() / \
                                Am_r[j].sum().clamp(min=1)
                        if n_pitch:
                            mp = Pm_r[j]
                            if mp.any():
                                loss = loss + args.pitch_w * \
                                    ce(op[mp], P_r[j][mp])
                    loss.backward()
                    opt.step()
                    sch.step()
            pt = predict(final, T(Xn, te), T(Xc, te), T(Xq, te))
            yt = y[te].astype(np.float64)
            tv = bss(yt, pt)
            print(f"  -> 미학습 test BSS {tv:.2f} | refit {refit_epochs}에폭 | "
                  f"예측평균 {pt.mean():.4f} vs 실제 {yt.mean():.4f}", flush=True)
            np.savez_compressed(f"./out/mtnn_{args.tag}_s{sd}_test_preds.npz",
                                y=yt, pred=pt)


if __name__ == "__main__":
    main()
