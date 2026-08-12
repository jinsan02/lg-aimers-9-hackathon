#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

~/venv451/bin/python src/teacher.py --tag T5_seq --max-season 2023 --prev --drop-f-pre 2022

~/venv451/bin/python src/train_gbdt2.py --model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 --val-season 2023 --test-season 2024 --seeds 3,4,5,6,8,13 --soft-target ./out/teacher_T5_seq.npz --tag DT5_seq
