@echo off
cd /d C:\aimers
set P=.venv\Scripts\python.exe
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5
%P% src\train_gbdt2.py --model cat %B% --boosting-type Ordered --val-season 2024 --seed 42 --tag OB1V_ordered
