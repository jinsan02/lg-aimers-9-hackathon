@echo off
cd /d C:\aimers
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --l2 10 --refit-mult 1.5
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% --failmode-cells --depth 5 --iters 3000 --tag ZD5 --seeds 42,7,13,3,4,5
echo ZD5_DONE
