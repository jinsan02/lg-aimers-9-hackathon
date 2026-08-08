#!/bin/bash
set -euo pipefail

PY="$HOME/venv451/bin/python"
BASE=(
  --model cat
  --feat-v2
  --te p,pc,ph,b,pi
  --te-dev
  --feat-std
  --std-k 80
  --std-to-prior
  --std-season-prior
  --feat-domain
  --feat-skill-pc
  --drop-f-pre 2022
  --val-season 2023
  --test-season 2024
)

"$PY" src/train_gbdt2.py "${BASE[@]}" --dump-npz out/mtc_judge.npz
"$PY" src/train_mtnn.py \
  --npz out/mtc_judge.npz \
  --tag MC1_pilot \
  --seeds 3,4 \
  --epochs 15 \
  --cell-consistent \
  --cell-w 0.5 \
  --refit-mult 1.5
