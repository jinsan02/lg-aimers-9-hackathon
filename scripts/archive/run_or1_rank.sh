#!/bin/bash
# OR1 — pairwise ranking 목적함수 파일럿. 한 시드로 해상도·다양성 게이트만 본다.
set -euo pipefail
cd ~/aimers
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 300 --depth 8 --l2 10 --iters 2000 --refit-mult 1.5 --drop-f-pre 2022"
$P src/train_gbdt2.py --model rank $B --val-season 2023 --test-season 2024 \
  --seed 42 --rank-group-size 64 --tag OR1_rank
