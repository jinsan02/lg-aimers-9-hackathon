"""현재 fpipe 피처 NPZ를 쓰는 TabM 시간분할 판정기.

검증 시즌으로 최적 에폭을 고른 뒤 검증 시즌까지 포함해 처음부터 재학습하고,
미학습 test 시즌을 평가한다. test 행은 전처리 통계와 범주 사전에 쓰지 않는다.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import rtdl_num_embeddings
from tabm_reference import Model, make_parameter_groups


def bss(y, p):
    r = float(y.mean())
    return float(1e5 * (1 - ((p - y) ** 2).mean() / (r * (1 - r))))


def make_model(n_num, cards, bins, args, dev, n_cells=0):
    num_embeddings = ({
        "type": "PiecewiseLinearEmbeddings",
        "d_embedding": args.d_embedding,
        "activation": False,
        "version": "B",
    } if bins is not None else None)
    return Model(
        n_num_features=n_num,
        cat_cardinalities=cards,
        n_classes=n_cells or None,
        backbone={"type": "MLP", "n_blocks": args.n_blocks,
                  "d_block": args.d_block, "dropout": args.dropout},
        bins=bins, num_embeddings=num_embeddings,
        arch_type="tabm-mini", k=args.k,
    ).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tag", default="TM1")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--eval-batch", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n-bins", type=int, default=48)
    ap.add_argument("--d-embedding", type=int, default=8)
    ap.add_argument("--n-blocks", type=int, default=3)
    ap.add_argument("--d-block", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--refit-mult", type=float, default=1.5)
    ap.add_argument("--cell-consistent", action="store_true",
                    help="TabM 출력을 실패모드 셀 softmax로 바꾸고 성공 marginal을 사용")
    ap.add_argument("--cell-w", type=float, default=0.5)
    args = ap.parse_args()

    z = np.load(args.npz)
    Xn = z["Xn"].astype(np.float32)
    Xc = z["Xc"].astype(np.int64)
    y = z["y"].astype(np.float32)
    va = z["is_val"].astype(bool)
    te = z["is_test"].astype(bool)
    tr = ~(va | te)
    if not va.any() or not te.any():
        raise ValueError("시간분할 판정에는 is_val과 is_test가 모두 필요합니다")

    # 통계는 초기 학습 구간에서만 적합한다. 미관측 범주는 dump 단계에서 이미
    # unknown 코드 하나로 모였으므로 test 분포를 훑지 않는다.
    med = np.nanmedian(Xn[tr], axis=0).astype(np.float32)
    med = np.where(np.isfinite(med), med, 0.0)
    Xn = np.where(np.isfinite(Xn), Xn, med).astype(np.float32)
    lo = np.percentile(Xn[tr], 0.5, axis=0).astype(np.float32)
    hi = np.percentile(Xn[tr], 99.5, axis=0).astype(np.float32)
    Xn = np.clip(Xn, lo, hi)
    mu = Xn[tr].mean(0).astype(np.float32)
    sd = Xn[tr].std(0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xn = ((Xn - mu) / sd).astype(np.float32)
    cards = [int(Xc[tr, j].max()) + 2 for j in range(Xc.shape[1])]
    cell = z["cell"].astype(np.int64) if args.cell_consistent else None
    succ = z["cell_success"].astype(np.int64) if args.cell_consistent else None
    n_cells = int(cell.max()) + 1 if args.cell_consistent else 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xn_t, Xc_t, y_t = (torch.from_numpy(Xn), torch.from_numpy(Xc),
                        torch.from_numpy(y))
    cell_t = torch.from_numpy(cell) if args.cell_consistent else None
    succ_t = torch.from_numpy(succ).to(dev) if args.cell_consistent else None
    print(f"학습 {int(tr.sum()):,} / 검증 {int(va.sum()):,} / "
          f"미학습 {int(te.sum()):,} | num {Xn.shape[1]} cat {Xc.shape[1]}")
    t0 = time.time()
    bins = rtdl_num_embeddings.compute_bins(torch.from_numpy(Xn[tr]),
                                             n_bins=args.n_bins)
    print(f"PLE bins {args.n_bins} | {time.time() - t0:.1f}s")

    def predict(model, mask):
        model.eval()
        idx = np.flatnonzero(mask)
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), args.eval_batch):
                j = idx[i:i + args.eval_batch]
                with torch.autocast(dev, dtype=torch.bfloat16,
                                    enabled=dev == "cuda"):
                    q = model(Xn_t[j].to(dev), Xc_t[j].to(dev))
                if args.cell_consistent:
                    p = torch.softmax(q.float(), -1)[..., succ_t].sum(-1).mean(1)
                else:
                    p = torch.sigmoid(q.squeeze(-1).float()).mean(1)
                out.append(p.cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def fit_epochs(model, mask, epochs, seed, watch=False):
        torch.manual_seed(seed)
        opt = torch.optim.AdamW(make_parameter_groups(model), lr=args.lr,
                                weight_decay=args.wd)
        idx = np.flatnonzero(mask)
        steps = math.ceil(len(idx) / args.batch)
        best, best_p, best_ep, bad = -1e12, None, -1, 0
        hist = []
        for ep in range(epochs):
            model.train()
            perm = idx[torch.randperm(len(idx)).numpy()]
            for s in range(steps):
                j = perm[s * args.batch:(s + 1) * args.batch]
                xn, xc, yy = Xn_t[j].to(dev), Xc_t[j].to(dev), y_t[j].to(dev)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(dev, dtype=torch.bfloat16,
                                    enabled=dev == "cuda"):
                    q = model(xn, xc)
                    if args.cell_consistent:
                        # 확률 marginal BCE는 autocast에서 금지된다. 셀 softmax와
                        # 두 손실만 fp32로 계산하고 트렁크는 bf16 이점을 유지한다.
                        with torch.autocast(dev, enabled=False):
                            q32 = q.float()
                            ps = torch.softmax(q32, -1)[..., succ_t].sum(-1)
                            loss = F.binary_cross_entropy(
                                ps.clamp(1e-6, 1 - 1e-6),
                                yy.float()[:, None].expand_as(ps))
                            cc = cell_t[j].to(dev)[:, None].expand(-1, args.k)
                            loss = loss + args.cell_w * F.cross_entropy(
                                q32.reshape(-1, n_cells), cc.reshape(-1))
                    else:
                        q = q.squeeze(-1)
                        loss = F.binary_cross_entropy_with_logits(
                            q, yy[:, None].expand_as(q))
                loss.backward()
                opt.step()
            if watch:
                p = predict(model, va)
                sc = bss(y[va], p)
                hist.append(sc)
                print(f"  ep{ep + 1:02d} val {sc:.2f}", flush=True)
                if sc > best:
                    best, best_p, best_ep, bad = sc, p, ep, 0
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
        return best, best_p, best_ep, hist

    os.makedirs("out", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        torch.manual_seed(seed)
        model = make_model(Xn.shape[1], cards, bins, args, dev, n_cells)
        score, pv, best_ep, hist = fit_epochs(model, tr, args.epochs, seed,
                                              watch=True)
        np.savez_compressed(f"out/tabm_{args.tag}_s{seed}_val_preds.npz",
                            y=y[va], pred=pv)

        refit_epochs = max(1, int(round((best_ep + 1) * args.refit_mult)))
        torch.manual_seed(seed)
        final = make_model(Xn.shape[1], cards, bins, args, dev, n_cells)
        fit_epochs(final, tr | va, refit_epochs, seed, watch=False)
        pt = predict(final, te)
        test_score = bss(y[te], pt)
        print(f"[tabm {args.tag}_s{seed}] val {score:.2f} ep{best_ep + 1} -> "
              f"test {test_score:.2f} refit{refit_epochs} | "
              f"mean {pt.mean():.4f}/{y[te].mean():.4f}", flush=True)
        np.savez_compressed(f"out/tabm_{args.tag}_s{seed}_test_preds.npz",
                            y=y[te], pred=pt)
        torch.save({"state_dict": final.state_dict(), "cards": cards,
                    "bins": [b.cpu() for b in bins], "args": vars(args),
                    "cell_success": succ, "n_cells": n_cells,
                    "med": med, "lo": lo, "hi": hi, "mu": mu, "sd": sd,
                    "best_epoch": best_ep},
                   f"model/tabm_{args.tag}_s{seed}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
