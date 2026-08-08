#!/bin/bash
# 미학습 표면(val 2023 -> 미학습 2024) 재판정 B조 — A100
cd ~/aimers || exit 1
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022"
V="--val-season 2023 --test-season 2024"
S="--seeds 3,4,5,6,8,13"
$P src/train_gbdt2.py --model cat $B $V $S --std-k 120 --tag SB_k120 2>&1 | grep -E "^\[cat|미학습"
$P src/train_gbdt2.py --model cat $B $V $S --te-k 100 --tag SB_tek100 2>&1 | grep -E "^\[cat|미학습"
$P src/train_gbdt2.py --model cat $B $V $S --lr 0.02 --tag SB_lr02 2>&1 | grep -E "^\[cat|미학습"
$P src/train_gbdt2.py --model cat $B $V $S --depth 9 --tag SB_d9 2>&1 | grep -E "^\[cat|미학습"
echo SURF_B_DONE
