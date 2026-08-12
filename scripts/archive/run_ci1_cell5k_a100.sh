#!/usr/bin/env bash
set -euo pipefail
cd ~/aimers
~/venv451/bin/python tools/precheck.py --model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --l2 10 --refit-mult 1.5 --depth 5 --iters 5000 --failmode-cells --drop-f-pre 2022 --val-season 2023 --test-season 2024 --seed 3 --tag CI1A_cell5k
~/venv451/bin/python src/train_gbdt2.py --model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --l2 10 --refit-mult 1.5 --depth 5 --iters 5000 --failmode-cells --drop-f-pre 2022 --val-season 2023 --test-season 2024 --seed 3 --tag CI1A_cell5k
