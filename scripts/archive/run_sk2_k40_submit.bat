@echo off
setlocal
cd /d C:\aimers

.venv\Scripts\python.exe src\train_gbdt2.py --model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 40 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --val-season 2024 --seeds 42,7,13,3,4,5,6,8 --tag SK2_k40
exit /b %errorlevel%
