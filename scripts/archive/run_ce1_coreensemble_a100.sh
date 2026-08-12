#!/usr/bin/env bash
set -euo pipefail
cd ~/aimers
BASE=(--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 --val-season 2023 --test-season 2024 --seeds 4,5,6,8,13)
~/venv451/bin/python tools/precheck.py "${BASE[@]}" --lr 0.005 --tag CE1A_lr5
~/venv451/bin/python src/train_gbdt2.py "${BASE[@]}" --lr 0.005 --tag CE1A_lr5
~/venv451/bin/python tools/precheck.py "${BASE[@]}" --lr 0.01 --random-strength 0.5 --tag CE1A_rs05
~/venv451/bin/python src/train_gbdt2.py "${BASE[@]}" --lr 0.01 --random-strength 0.5 --tag CE1A_rs05
~/venv451/bin/python tools/precheck.py "${BASE[@]}" --lr 0.01 --random-strength 2 --tag CE1A_rs2
~/venv451/bin/python src/train_gbdt2.py "${BASE[@]}" --lr 0.01 --random-strength 2 --tag CE1A_rs2
