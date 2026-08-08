#!/usr/bin/env bash
set -euo pipefail
cd ~/aimers
P=~/venv451/bin/python
$P src/train_gbdt2.py --model cat --feat-v2 --feat-std --std-k 80 \
  --std-to-prior --std-season-prior --feat-domain \
  --drop-f-pre 2022 --val-season 2023 --test-season 2024 \
  --dump-npz out/plr1_raw.npz --tag PLR1_dump
$P src/train_mtnn.py --npz out/plr1_raw.npz --tag PLR1 --seeds 3 \
  --epochs 18 --bs 8192 --lr 0.002 --drop 0.15 --qbins 0 --aux-w 0 \
  --plr --plr-freq 32 --plr-dim 8 --refit-mult 1.5
