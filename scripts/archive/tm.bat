@echo off
cd /d C:\aimers
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% --no-refit --tag TM1 --seeds 3,4,5,6,8,13 --tm-feats data/processed/tm_pitcher_feats.csv
echo TM1_DONE
