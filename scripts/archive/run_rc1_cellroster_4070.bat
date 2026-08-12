@echo off
cd /d C:\aimers
set P=.venv\Scripts\python.exe
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 5 --l2 10 --iters 3000 --refit-mult 1.5
%P% src\train_gbdt2.py --model cat %B% --feat-roster --failmode-cells --val-season 2024 --seed 42 --tag RC1V_cellroster
