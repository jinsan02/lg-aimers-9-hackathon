"""Train the validated masked-pitch auxiliary network on all labelled seasons.

The current pitch type is used only as a masked auxiliary *training label*.
At inference the exported TorchScript model receives the ordinary row-local
competition features and returns the control-success logit; its pitch head is
discarded.  No Trackman table or pitch label is packaged for inference.
"""

from __future__ import annotations

import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import QuantileTransformer

import fpipe
from train_mtnn import MTNet


def encode_categories(df, columns, vocab):
    out = []
    for col in columns:
        cats = vocab[col]
        unknown = len(cats)
        mapping = {value: i for i, value in enumerate(cats)}
        out.append(df[col].astype(str).map(mapping).fillna(unknown)
                   .to_numpy(np.int64))
    return (np.stack(out, 1) if out else
            np.zeros((len(df), 0), dtype=np.int64))


def predict(net, xn, xc, xq, device, batch=65536):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xn), batch):
            logit, _, _ = net(
                torch.from_numpy(xn[i:i + batch]).to(device),
                torch.from_numpy(xc[i:i + batch]).to(device),
                torch.from_numpy(xq[i:i + batch]).to(device),
            )
            out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--historical-test-season", type=int, default=0,
                    help="0보다 크면 data/train.csv의 해당 시즌을 미래 test로 변환해 평가")
    ap.add_argument("--train-all", action="store_true",
                    help="lagged-artifact 최종 refit: NPZ의 val 표시도 학습에 포함")
    ap.add_argument("--out-dir", default="model/pitch_mtl")
    ap.add_argument("--pred-out", default="out/pitch_mtnn_full_test_preds.npz")
    ap.add_argument("--seeds", default="3,4,5,6,8,13")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--qbins", type=int, default=32)
    ap.add_argument("--qt-max-season", type=int, default=0,
                    help="진단용: QuantileTransformer만 이 시즌 이하에서 fit")
    ap.add_argument("--pitch-w", type=float, default=0.1)
    args = ap.parse_args()

    if args.pitch_w < 0:
        raise ValueError("--pitch-w must be nonnegative")
    z = np.load(args.npz, allow_pickle=True)
    if (z["is_val"].any() or z["is_test"].any()) and not args.train_all:
        raise ValueError("final trainer requires a --dump-full-fit NPZ")
    xn_raw = np.nan_to_num(z["Xn"].astype(np.float32), nan=0.0,
                           posinf=0.0, neginf=0.0)
    xc = z["Xc"].astype(np.int64)
    y = z["y"].astype(np.float32)
    pitch = z["pitch"].astype(np.int64)
    meta_path = args.npz.replace(".npz", "_meta.pkl")
    meta = joblib.load(meta_path)

    print(f"full train {len(y):,} | pitch coverage {(pitch >= 0).mean()*100:.2f}%")
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             subsample=300_000, random_state=0)
    if args.qt_max_season:
        qt_fit = z["season"].astype(int) <= args.qt_max_season
        if not qt_fit.any():
            raise ValueError("--qt-max-season selected no rows")
        qt.fit(xn_raw[qt_fit])
        xn = qt.transform(xn_raw).astype(np.float32)
        print(f"QT fit through {args.qt_max_season}: {qt_fit.sum():,} rows")
    else:
        xn = qt.fit_transform(xn_raw).astype(np.float32)
    xq = np.clip((xn * 4 + args.qbins / 2).astype(np.int64),
                 0, args.qbins - 1)
    cards = [len(meta["vocab"][c]) + 1 for c in meta["cat_cols"]]
    if any(xc[:, i].max(initial=-1) >= card for i, card in enumerate(cards)):
        raise ValueError("categorical code exceeds frozen vocabulary")

    if args.historical_test_season:
        test = pd.read_csv("data/train.csv", encoding="utf-8-sig")
        test = test[test["season"] == args.historical_test_season].reset_index(drop=True)
        test_y = test["control_success"].to_numpy(np.float64)
    else:
        test = pd.read_csv(args.test, encoding="utf-8-sig")
        test_y = None
    test_f = fpipe.transform(test.copy(), meta["fpipe"])
    xt_raw = np.nan_to_num(test_f[meta["num"]].to_numpy(np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0)
    xt = qt.transform(xt_raw).astype(np.float32)
    xqt = np.clip((xt * 4 + args.qbins / 2).astype(np.int64),
                  0, args.qbins - 1)
    xct = encode_categories(test_f, meta["cat_cols"], meta["vocab"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.pred_out) or ".", exist_ok=True)
    joblib.dump({"qt": qt, "feature_meta": meta, "qbins": args.qbins,
                 "cards": cards}, os.path.join(args.out_dir, "preprocess.pkl"),
                compress=3)

    xn_t = torch.from_numpy(xn).to(device)
    xc_t = torch.from_numpy(xc).to(device)
    xq_t = torch.from_numpy(xq).to(device)
    y_t = torch.from_numpy(y).to(device)
    p_t = torch.from_numpy(pitch).to(device)
    mask_t = p_t >= 0
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    indices = np.arange(len(y))
    preds = []

    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        torch.manual_seed(seed)
        n_pitch = 3 if args.pitch_w > 0 else 0
        net = MTNet(xn.shape[1], cards, n_aux=0, n_pitch=n_pitch,
                    drop=args.drop, qbins=args.qbins).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                weight_decay=args.wd)
        steps = args.epochs * (len(indices) // args.bs + 1)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr,
                                                   total_steps=steps)
        t0 = time.time()
        net.train()
        for epoch in range(args.epochs):
            perm = torch.randperm(len(indices), device=device)
            running = 0.0
            for i in range(0, len(indices), args.bs):
                j = perm[i:i + args.bs]
                opt.zero_grad()
                logit, _, pitch_logit = net(xn_t[j], xc_t[j], xq_t[j])
                loss = bce(logit, y_t[j])
                mp = mask_t[j]
                if n_pitch and mp.any():
                    loss = loss + args.pitch_w * ce(pitch_logit[mp], p_t[j][mp])
                loss.backward()
                opt.step()
                sch.step()
                running += float(loss.detach()) * len(j)
            print(f"seed {seed} epoch {epoch+1}/{args.epochs} "
                  f"loss={running/len(indices):.6f}", flush=True)

        pt = predict(net, xt, xct, xqt, device)
        preds.append(pt)
        net = net.cpu().eval()
        scripted = torch.jit.script(net)
        scripted.save(os.path.join(args.out_dir, f"pitch_aux_s{seed}.pt"))
        print(f"seed {seed} done {time.time()-t0:.0f}s | test mean={pt.mean():.6f}",
              flush=True)

    pred = np.mean(preds, axis=0)
    payload = dict(row_id=test["row_id"].astype(str).to_numpy(), pred=pred,
                   members=np.stack(preds),
                   seeds=np.asarray([int(s) for s in args.seeds.split(",")
                                     if s.strip()]))
    if test_y is not None:
        payload["y"] = test_y
    np.savez_compressed(args.pred_out, **payload)
    print(f"saved {args.pred_out} | ensemble mean={pred.mean():.6f} "
          f"sd={pred.std():.6f}")


if __name__ == "__main__":
    main()
