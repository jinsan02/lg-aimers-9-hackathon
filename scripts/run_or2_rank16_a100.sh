#!/usr/bin/env bash
set -euo pipefail
cd ~/aimers
P=~/venv451/bin/python
$P src/train_gbdt2.py --model rank --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --iters 2000 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 --val-season 2023 --test-season 2024 --rank-group-size 16 --seed 42 --tag OR2_rank16
