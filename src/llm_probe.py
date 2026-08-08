"""LLM 결정 실험 — **사전학습이 이 과제에 실제로 기여하는가**.

같은 아키텍처를 (a) 사전학습 가중치 (b) 랜덤 초기화 로 각각 파인튜닝해 비교한다.
  (a) > (b) 면 → 언어 사전학습 지식이 실제로 이전됨 = LLM 트랙 가치 있음
  (a) ≈ (b) 면 → 사전학습은 무의미, 그냥 작은 트랜스포머 = TabM/DTT와 동급

직렬화는 **한 줄 compact** — 투수 의도(trackman 상황 집계)까지 포함.

실행(A100): ~/.venvs/tabicl/bin/python src/llm_probe.py --model Qwen/Qwen2.5-0.5B
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

DATA = "./data"
TARGET = "control_success"
DEV = "cuda"


def bss(y, p):
    r = y.mean()
    return float(100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r))))


def serialize(df, ctx=None):
    """한 줄 직렬화 — 상황 + 투수/타자 이력 + (있으면) 투수 의도."""
    b = df.balls_before.astype(str)
    s = df.strikes_before.astype(str)
    hand = np.where(df.pitcher_hand == 2, "RHP", "LHP")
    bh = np.where(df.batter_hand == 2, "RHB", "LHB")
    base = df.base_state.astype(str)
    out = (df.season.astype(str) + "/" + df.game_month.astype(str)
           + " inn" + df.inning.astype(str) + df.top_bottom.astype(str)
           + " " + b + "-" + s + " out" + df.outs_before.astype(str)
           + " base" + base
           + " diff" + df.score_diff_pitcher_team.astype(str)
           + " li" + df.li.round(1).astype(str)
           + " " + hand + "v" + bh
           + " Pn" + (df.asof_pitcher_n // 100).astype(str)
           + " Psr" + df.asof_pitcher_success_rate.round(3).astype(str)
           + " Prev" + df.asof_pitcher_reverse_rate.round(3).astype(str)
           + " P5" + df.asof_pitcher_prev5_game_success_rate.round(3).astype(str)
           + " Bsr" + df.asof_batter_success_rate.round(3).astype(str))
    if ctx is not None:
        out = out + " FB" + ctx.round(2).astype(str)   # 이 상황 리그 속구 비율 = 투수 의도
    return out.tolist()


def build(model_name, pretrained):
    if pretrained:
        m = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, torch_dtype=torch.float32)
    else:
        cfg = AutoConfig.from_pretrained(model_name, num_labels=1)
        m = AutoModelForSequenceClassification.from_config(cfg)
    return m.to(DEV)


def run(model, tr_loader, va_ids, va_mask, y_va, epochs, lr, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best = -1e9
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for ids, mask, yb in tr_loader:
            ids, mask, yb = ids.to(DEV), mask.to(DEV), yb.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEV, dtype=torch.bfloat16):
                logit = model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logit.float(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        model.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(va_ids), 256):
                with torch.autocast(DEV, dtype=torch.bfloat16):
                    lg = model(input_ids=va_ids[i:i + 256].to(DEV),
                               attention_mask=va_mask[i:i + 256].to(DEV)
                               ).logits.squeeze(-1)
                ps.append(torch.sigmoid(lg.float()).cpu())
        p = torch.cat(ps).numpy().astype(np.float64)
        sc = bss(y_va, p)
        best = max(best, sc)
        print(f"  [{tag}] ep{ep} loss {tot / n:.5f} | val BSS {sc:9.2f} "
              f"| {time.time() - t0:.0f}s", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--n-val", type=int, default=50_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=80)
    args = ap.parse_args()

    test_cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
    feats = [c for c in test_cols if c != "row_id"]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=feats + [TARGET])
    ctx = pd.read_csv(f"{DATA}/processed/tm_context.csv")
    ck = ["balls_before", "strikes_before", "outs_before", "top_bottom",
          "pitcher_hand", "batter_hand"]
    cagg = ctx.groupby(ck)["tmc_pt_fastball"].mean().rename("fb")
    df = df.merge(cagg, on=ck, how="left")
    df["fb"] = df["fb"].fillna(df["fb"].mean())

    rng = np.random.default_rng(0)
    tr = df[df.season < 2024]
    va = df[df.season == 2024]
    tr = tr.iloc[rng.choice(len(tr), args.n_train, replace=False)]
    va = va.iloc[rng.choice(len(va), args.n_val, replace=False)]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    s_tr, s_va = serialize(tr, tr.fb), serialize(va, va.fb)
    print("한 줄 직렬화 예시:")
    print(" ", s_tr[0])
    print(f"  토큰 수: {len(tok(s_tr[0])['input_ids'])}\n")

    def enc(texts):
        e = tok(texts, padding="max_length", truncation=True,
                max_length=args.max_len, return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    tr_ids, tr_mask = enc(s_tr)
    va_ids, va_mask = enc(s_va)
    y_tr = torch.as_tensor(tr[TARGET].to_numpy(np.float32))
    y_va = va[TARGET].to_numpy(np.float64)
    loader = DataLoader(TensorDataset(tr_ids, tr_mask, y_tr),
                        batch_size=args.batch, shuffle=True, drop_last=True)

    results = {}
    for pre in [True, False]:
        name = "사전학습" if pre else "랜덤초기화"
        print(f"=== {name} ===")
        torch.manual_seed(0)
        m = build(args.model, pre)
        m.config.pad_token_id = tok.pad_token_id
        results[name] = run(m, loader, va_ids, va_mask, y_va,
                            args.epochs, args.lr, name)
        del m
        torch.cuda.empty_cache()

    print("\n=== 판정 ===")
    for k, v in results.items():
        print(f"  {k:8s} best BSS {v:9.2f}")
    gap = results["사전학습"] - results["랜덤초기화"]
    print(f"  사전학습 이득: {gap:+.2f}")
    print("  → 이득이 유의미하면 LLM 트랙 가치 있음, 아니면 그냥 작은 트랜스포머")


if __name__ == "__main__":
    main()
