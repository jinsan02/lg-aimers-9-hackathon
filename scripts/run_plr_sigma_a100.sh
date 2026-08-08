#!/usr/bin/env bash
set -euo pipefail
cd ~/aimers
P=~/venv451/bin/python
for S in 1 10; do
  $P src/train_mtnn.py --npz out/plr1_raw.npz --tag PLR_sigma${S} --seeds 3 \
    --epochs 18 --bs 8192 --lr 0.002 --drop 0.15 --qbins 0 --aux-w 0 \
    --plr --plr-freq 32 --plr-dim 8 --plr-sigma $S --refit-mult 1.5
done
