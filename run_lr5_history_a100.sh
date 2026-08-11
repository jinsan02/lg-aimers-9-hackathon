#!/usr/bin/env bash
set -euo pipefail
cd ~/aimers
COMMON=(--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.005 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 --seed 3)
~/venv451/bin/python tools/precheck.py "${COMMON[@]}" --max-train-season 2023 --val-season 2022 --test-season 2023 --tag LR5A22
~/venv451/bin/python src/train_gbdt2.py "${COMMON[@]}" --max-train-season 2023 --val-season 2022 --test-season 2023 --tag LR5A22
~/venv451/bin/python tools/precheck.py "${COMMON[@]}" --max-train-season 2022 --val-season 2021 --test-season 2022 --tag LR5A21
~/venv451/bin/python src/train_gbdt2.py "${COMMON[@]}" --max-train-season 2022 --val-season 2021 --test-season 2022 --tag LR5A21
