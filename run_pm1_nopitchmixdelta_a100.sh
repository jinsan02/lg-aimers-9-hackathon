#!/usr/bin/env bash
set -euo pipefail
cd ~/aimers
DROP=std_asof_pitcher_offspeed_rate_delta
COMMON=(--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --drop-cols "$DROP" --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 --seed 3)
~/venv451/bin/python tools/precheck.py "${COMMON[@]}" --val-season 2023 --test-season 2024 --tag PM1A_nooffspeeddelta
~/venv451/bin/python src/train_gbdt2.py "${COMMON[@]}" --val-season 2023 --test-season 2024 --tag PM1A_nooffspeeddelta
