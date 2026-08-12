#!/usr/bin/env bash
set -euo pipefail
cd ~/aimers
DROP=std_asof_pitcher_success_rate_delta,std_asof_pitcher_reverse_rate_delta,std_asof_pitcher_middle_rate_delta,std_asof_pitcher_ball_rate_delta,std_asof_pitcher_strike_rate_delta,std_asof_batter_success_rate_delta,std_asof_batter_middle_rate_delta,std_asof_pitcher_fastball_rate_delta,std_asof_pitcher_breaking_rate_delta,std_asof_pitcher_offspeed_rate_delta
COMMON=(--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --drop-cols "$DROP" --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 --seed 3)
~/venv451/bin/python tools/precheck.py "${COMMON[@]}" --max-train-season 2023 --val-season 2022 --test-season 2023 --tag DA3A22_nodelta
~/venv451/bin/python src/train_gbdt2.py "${COMMON[@]}" --max-train-season 2023 --val-season 2022 --test-season 2023 --tag DA3A22_nodelta
~/venv451/bin/python tools/precheck.py "${COMMON[@]}" --max-train-season 2022 --val-season 2021 --test-season 2022 --tag DA3A21_nodelta
~/venv451/bin/python src/train_gbdt2.py "${COMMON[@]}" --max-train-season 2022 --val-season 2021 --test-season 2022 --tag DA3A21_nodelta
