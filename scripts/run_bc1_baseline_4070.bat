@echo off
cd /d C:\aimers
set PYTHONUTF8=1
.venv\Scripts\python.exe src\train_gbdt2.py --model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --baseline-col skill_pc_hat --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --val-season 2024 --seed 42 --tag BC1_offset
