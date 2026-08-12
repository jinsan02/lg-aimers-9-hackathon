#!/bin/bash
# OG1 — 기존 depth5 셀 모델의 14차원 확률분포를 보존한다.
# 모델·손실·피처는 DW_cell과 동일하며 --dump-cell-proba는 산출물만 추가한다.
set -euo pipefail
cd ~/aimers
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 5 --l2 10 --iters 3000 --refit-mult 1.5 --drop-f-pre 2022"
V="--val-season 2023 --test-season 2024"
S="--seeds 3,4,5,6,8,13"
$P src/train_gbdt2.py --model cat $B $V $S --failmode-cells --dump-cell-proba \
  --tag OG1_cell14
