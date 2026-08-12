#!/usr/bin/env bash
set -euo pipefail
cd ~/aimers
COMMON=(--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80
  --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --feat-id-cohort
  --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 --seed 3)
~/venv451/bin/python src/train_gbdt2.py "${COMMON[@]}" \
  --max-train-season 2023 --val-season 2022 --test-season 2023 --tag IC1A22_idcohort
~/venv451/bin/python src/train_gbdt2.py "${COMMON[@]}" \
  --max-train-season 2022 --val-season 2021 --test-season 2022 --tag IC1A21_idcohort
