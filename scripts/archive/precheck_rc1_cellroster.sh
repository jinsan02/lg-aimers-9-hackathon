#!/bin/bash
set -euo pipefail
cd ~/aimers
P=~/venv451/bin/python
B=(--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80
   --std-to-prior --std-season-prior --feat-domain --feat-skill-pc
   --lr 0.01 --es 500 --depth 5 --l2 10 --iters 3000 --refit-mult 1.5
   --feat-roster --failmode-cells)
$P tools/precheck.py "${B[@]}" --drop-f-pre 2022 --val-season 2023 \
  --test-season 2024 --seed 3 --tag RC1A_cellroster
$P tools/precheck.py "${B[@]}" --val-season 2024 --seed 42 --tag RC1V_cellroster
