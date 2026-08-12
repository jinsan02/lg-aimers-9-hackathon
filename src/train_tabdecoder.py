"""Column-token causal decoder for row-independent control prediction.

Each legal pre-pitch column becomes exactly one token.  Categorical values use
field-local vocabularies; numerical values use a field-local quantile token
plus a robust continuous projection.  A final prediction token attends
causally to the 47 feature tokens and emits one control-success probability.

The historical gate mirrors deployment:
  stage 1: fit <=2022 (dropping old-regime F), early-stop on 2023
  stage 2: rebuild tokenizer on <=2023, refit for the selected epoch count,
           evaluate the untouched 2024 season.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ID = "row_id"
TARGET = "control_success"
N_BINS = 64
MAX_CAT = 64
BIG_CAT = {"pitcher_id", "batter_id"}

# Fixed baseball scenario order.  The prediction token sees all fields, while
# intermediate tokens only see the scenario prefix (GPT-style causal mask).
SCENARIO_ORDER = [
    "season", "game_month", "game_dayofweek", "game_type",
    "inning", "top_bottom", "outs_before",
    "balls_before", "strikes_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "num_runners_on", "base_state",
    "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_id", "pitcher_hand", "pitcher_team_id",
    "batter_id", "batter_hand", "batter_team_id",
    "asof_pitcher_n", "asof_batter_n",
    "asof_pitcher_success_rate", "asof_batter_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]


def bss(y, p):
    y = np.asarray(y, np.float64)
    p = np.clip(np.asarray(p, np.float64), 0, 1)
    r = float(y.mean())
    return 1e5 * (1 - np.mean((p-y)**2)/(r*(1-r)))


@dataclass
class Spec:
    col: str
    kind: str
    offset: int
    cats: list | None = None
    edges: list | None = None
    center: float = 0.0
    scale: float = 1.0


def build_specs(fit: pd.DataFrame):
    specs, offset = [], 0
    for col in SCENARIO_ORDER:
        s = fit[col]
        categorical = (col in BIG_CAT or s.dtype == object or
                       (col != "season" and s.nunique(dropna=True) <= MAX_CAT))
        if categorical:
            cats = sorted(s.dropna().unique().tolist())
            specs.append(Spec(col, "cat", offset, cats=cats))
            offset += len(cats)+1             # zero is field-local missing/UNK
        else:
            x = pd.to_numeric(s, errors="coerce").to_numpy(np.float64)
            good = x[np.isfinite(x)]
            edges = np.unique(np.quantile(good, np.linspace(0,1,N_BINS+1)[1:-1]))
            center = float(np.median(good))
            q25, q75 = np.quantile(good, [.25,.75])
            scale = float(max(q75-q25, np.std(good)*.25, 1e-6))
            specs.append(Spec(col, "num", offset, edges=edges.tolist(),
                              center=center, scale=scale))
            offset += len(edges)+2            # missing + finite bins
    return specs, offset


def encode(frame: pd.DataFrame, specs):
    n, m = len(frame), len(specs)
    token = np.empty((n,m), np.int32)
    cont = np.zeros((n,m), np.float32)
    numeric = np.zeros(m, np.float32)
    for j,sp in enumerate(specs):
        s = frame[sp.col]
        if sp.kind == "cat":
            lut = {v:i+1 for i,v in enumerate(sp.cats)}
            token[:,j] = s.map(lut).fillna(0).to_numpy(np.int32)+sp.offset
        else:
            x = pd.to_numeric(s, errors="coerce").to_numpy(np.float64)
            ok = np.isfinite(x)
            code = np.zeros(n, np.int32)
            code[ok] = np.searchsorted(np.asarray(sp.edges), x[ok], side="right")+1
            token[:,j] = code+sp.offset
            z = np.zeros(n, np.float32)
            z[ok] = np.clip((x[ok]-sp.center)/sp.scale, -8, 8).astype(np.float32)
            cont[:,j] = z
            numeric[j] = 1
    return token, cont, numeric


class TabDecoder(nn.Module):
    def __init__(self, vocab, n_cols, numeric_mask, dim=96, layers=3,
                 heads=8, dropout=.1, causal=True):
        super().__init__()
        self.value = nn.Embedding(vocab, dim)
        self.column = nn.Embedding(n_cols+1, dim)
        self.cont_weight = nn.Parameter(torch.zeros(n_cols,dim))
        self.pred = nn.Parameter(torch.zeros(1,1,dim))
        block = nn.TransformerEncoderLayer(
            dim, heads, dim*4, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim,1)
        self.register_buffer("positions", torch.arange(n_cols), persistent=False)
        self.register_buffer("numeric", torch.as_tensor(numeric_mask), persistent=False)
        # A causal mask over table columns has never been justified -- SCENARIO_ORDER
        # is a chosen order, not a natural one. --no-causal-mask makes every feature
        # token see every other, which is the only clean structural difference from
        # the closed ft-transformer axis (that one was bidirectional with a CLS token).
        mask = (torch.triu(torch.full((n_cols+1, n_cols+1), float("-inf")), diagonal=1)
                if causal else None)
        self.register_buffer("causal_mask", mask, persistent=False)
        nn.init.normal_(self.value.weight, std=.02)
        nn.init.normal_(self.column.weight, std=.02)
        nn.init.normal_(self.cont_weight, std=.01)
        nn.init.normal_(self.pred, std=.02)

    def forward(self, token, cont):
        h = self.value(token)+self.column(self.positions)[None]
        h = h+cont[:,:,None]*self.cont_weight[None]*self.numeric[None,:,None]
        pred = self.pred.expand(len(token),-1,-1)+self.column.weight[-1][None,None]
        h = torch.cat([h,pred],1)
        h = self.decoder(h, mask=self.causal_mask)
        return self.head(self.norm(h[:,-1])).squeeze(-1)


class Arrays:
    def __init__(self, frame, specs):
        self.token, self.cont, self.numeric = encode(frame, specs)
        self.y = frame[TARGET].to_numpy(np.float32)


def train_model(train, valid, specs, vocab, args, epochs, early_stop=True):
    tr, va = Arrays(train,specs), Arrays(valid,specs)
    device = torch.device("cuda")
    model = TabDecoder(vocab,len(specs),tr.numeric,args.dim,args.layers,
                       args.heads,args.dropout,
                       causal=not args.no_causal_mask).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.wd)
    steps = math.ceil(len(tr.y)/args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,args.lr,total_steps=max(steps*epochs,1),pct_start=args.pct_start)
    best = {"score":-1e30,"epoch":0,"state":None,"pred":None}
    bad = 0
    for ep in range(epochs):
        model.train(); order=np.random.permutation(len(tr.y)); total=0.; t0=time.time()
        for start in range(0,len(order),args.batch):
            ix=order[start:start+args.batch]
            tok=torch.as_tensor(tr.token[ix],device=device,dtype=torch.long)
            con=torch.as_tensor(tr.cont[ix],device=device)
            y=torch.as_tensor(tr.y[ix],device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=F.binary_cross_entropy_with_logits(model(tok,con),y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); sched.step(); total+=float(loss)*len(ix)
        pred = predict(model,va,args.eval_batch)
        score=bss(va.y,pred)
        print(f"epoch {ep+1:02d}/{epochs} loss={total/len(tr.y):.6f} "
              f"BSS={score:.3f} sec={time.time()-t0:.1f}",flush=True)
        # In refit mode the epoch count is already frozen by stage 1.  Keep
        # the final epoch unconditionally; never select on the 2024 labels.
        if (not early_stop) or score>best["score"]:
            best={"score":score,"epoch":ep+1,
                  "state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                  "pred":pred}; bad=0
        else:
            bad+=1
            if early_stop and bad>=args.patience: break
    model.load_state_dict(best["state"])
    return model,best,va


@torch.no_grad()
def predict(model, arrays, batch):
    model.eval(); out=[]; device=next(model.parameters()).device
    for start in range(0,len(arrays.y),batch):
        tok=torch.as_tensor(arrays.token[start:start+batch],device=device,dtype=torch.long)
        con=torch.as_tensor(arrays.cont[start:start+batch],device=device)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(tok,con).float()).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def spec_json(specs):
    return [{"col":s.col,"kind":s.kind,"offset":s.offset,
             "n_values":len(s.cats)+1 if s.kind=="cat" else len(s.edges)+2,
             "center":s.center,"scale":s.scale} for s in specs]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/train.csv")
    ap.add_argument("--tag",default="TDEC1")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--dim",type=int,default=96)
    ap.add_argument("--layers",type=int,default=3)
    ap.add_argument("--heads",type=int,default=8)
    ap.add_argument("--dropout",type=float,default=.1)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--wd",type=float,default=1e-2)
    ap.add_argument("--batch",type=int,default=2048)
    ap.add_argument("--eval-batch",type=int,default=8192)
    ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--patience",type=int,default=3)
    # TDEC1 stopped at epoch 2 with pct_start=.1 over 15 epochs -- warmup is 1.5
    # epochs, so it died at peak LR and never annealed. patience 3 read that
    # warmup wobble as divergence.
    ap.add_argument("--pct-start",type=float,default=.1)
    ap.add_argument("--no-causal-mask",action="store_true")
    args=ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    # row_id is carried only so the saved npz matches the cat-family layout --
    # tools/loss_map.py skips any prediction file without it, silently.
    use=SCENARIO_ORDER+[TARGET,ID]
    data=pd.read_csv(args.data,usecols=use)
    old_f=data.game_type.eq("F")&data.season.le(2022)
    fit1=data[data.season.le(2022)&~old_f].reset_index(drop=True)
    val1=data[data.season.eq(2023)].reset_index(drop=True)
    specs1,vocab1=build_specs(fit1)
    print(f"stage1 train={len(fit1):,} val2023={len(val1):,} "
          f"cols={len(specs1)} vocab={vocab1}")
    _,best,_=train_model(fit1,val1,specs1,vocab1,args,args.epochs,True)
    print(f"stage1 BEST epoch={best['epoch']} BSS={best['score']:.3f}")
    torch.cuda.empty_cache()

    fit2=data[data.season.le(2023)&~old_f].reset_index(drop=True)
    test2=data[data.season.eq(2024)].reset_index(drop=True)
    specs2,vocab2=build_specs(fit2)
    # No target labels affect epoch selection; the selected count is frozen.
    model2,_,va2=train_model(fit2,test2,specs2,vocab2,args,best["epoch"],False)
    arr2=Arrays(test2,specs2); p2=predict(model2,arr2,args.eval_batch)
    score2=bss(arr2.y,p2)
    os.makedirs("out",exist_ok=True); os.makedirs("model",exist_ok=True)
    np.savez_compressed(f"out/tdec_{args.tag}_s{args.seed}_val_preds.npz",
                        y=val1[TARGET].to_numpy(np.float64),pred=best["pred"],
                        row_id=val1[ID].to_numpy())
    np.savez_compressed(f"out/tdec_{args.tag}_s{args.seed}_test_preds.npz",
                        y=arr2.y.astype(np.float64),pred=p2,
                        row_id=test2[ID].to_numpy())
    torch.save({"state_dict":model2.state_dict(),"vocab":vocab2,
                "numeric":Arrays(fit2.head(1),specs2).numeric,
                "config":vars(args),"specs":spec_json(specs2)},
               f"model/tdec_{args.tag}_s{args.seed}.pt")
    with open(f"out/tdec_{args.tag}_mapping.json","w",encoding="utf-8") as f:
        json.dump({"scenario_order":SCENARIO_ORDER,"specs":spec_json(specs2),
                   "selected_epoch":best["epoch"],"val2023_bss":best["score"],
                   "test2024_bss":score2},f,ensure_ascii=False,indent=2)
    print(f"FINAL unseen2024 BSS={score2:.3f} mean={p2.mean():.6f}")
    ledger(args, best, score2)


def ledger(args, best, test_bss):
    """LEDGER.tsv 에 한 행 남긴다 — train_gbdt2.py 와 같은 11열 형식.

    기록이 멈추면 대화 기억으로 일하게 되고 대화는 압축된다(2026-08-07 에
    30개 실험을 잃었다). 기록 실패로 학습 결과를 날리지는 않는다.
    """
    try:
        with open("./LEDGER.tsv", "a", encoding="utf-8") as lg:
            lg.write("\t".join([
                time.strftime("%Y-%m-%d %H:%M"), socket.gethostname(),
                f"{args.tag}_s{args.seed}", "tdec", "2023", "2024",
                str(args.seed), str(best["epoch"]),
                f"{best['score']:.2f}", f"{test_bss:.2f}",
                " ".join(sys.argv[1:])]) + "\n")
    except Exception as e:                 # 기록 실패로 학습을 죽이지 않는다
        print(f"  (LEDGER 기록 실패: {e})")


if __name__=="__main__":
    main()
