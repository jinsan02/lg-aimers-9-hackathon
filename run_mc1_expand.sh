#!/bin/bash
set -euo pipefail

"$HOME/venv451/bin/python" src/train_mtnn.py \
  --npz out/mtc_judge.npz \
  --tag MC1_expand \
  --seeds 5,6,8,13 \
  --epochs 15 \
  --cell-consistent \
  --cell-w 0.5 \
  --refit-mult 1.5
