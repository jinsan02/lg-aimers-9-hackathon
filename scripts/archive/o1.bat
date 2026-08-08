@echo off
cd /d C:\aimers
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc
set O=--lr 0.0081 --depth 8 --l2 41.9097 --border-count 128 --bagging-temp 0.5508 --random-strength 0.6526 --cat-min-leaf 85
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %O% --iters 4000 --es 400 --no-refit --tag O1 --seeds 3,4,5,6,8,13
echo O1_DONE
