@echo off
cd /d C:\aimers
set PYTHONUTF8=1
.venv\Scripts\python.exe src\train_gbdt2.py --model rank --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 20 --iters 50 --depth 8 --l2 10 --drop-f-pre 2022 --val-season 2023 --test-season 2024 --rank-group-size 16 --no-refit --seed 42 --tag OR1_smoke16
