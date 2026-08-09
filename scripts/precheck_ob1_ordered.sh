#!/bin/bash
set -euo pipefail
cd ~/aimers
P=~/venv451/bin/python
B=(--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80
   --std-to-prior --std-season-prior --feat-domain --feat-skill-pc
   --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5
   --boosting-type Ordered)
$P -m py_compile src/train_gbdt2.py
$P tools/precheck.py "${B[@]}" --drop-f-pre 2022 --val-season 2023 \
  --test-season 2024 --seed 3 --tag OB1A_ordered
$P tools/precheck.py "${B[@]}" --val-season 2024 --seed 42 --tag OB1V_ordered
