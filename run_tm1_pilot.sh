#!/bin/bash
set -euo pipefail

"$HOME/venv451/bin/python" src/train_tabm_npz.py \
  --npz out/mtc_judge.npz \
  --tag TM1_pilot \
  --seeds 42 \
  --epochs 15 \
  --patience 4 \
  --k 32 \
  --n-bins 48 \
  --d-embedding 8 \
  --n-blocks 3 \
  --d-block 384 \
  --refit-mult 1.5
