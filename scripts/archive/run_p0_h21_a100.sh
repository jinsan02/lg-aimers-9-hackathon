#!/usr/bin/env bash
set -euo pipefail
cd ~/aimers
PY=~/venv451/bin/python
COMMON="--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2022 --val-season 2021 --test-season 2022 --seed 3"
$PY src/train_gbdt2.py $COMMON --depth 8 --tag H21_base
$PY src/train_gbdt2.py $COMMON --depth 5 --iters 3000 --failmode-cells --tag H21_cell

