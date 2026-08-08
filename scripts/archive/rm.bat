@echo off
cd /d C:\aimers
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10
set V=--val-season 2023 --test-season 2024
for %%M in (1.5 1.7 2.0) do (
  .venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% --refit-mult %%M --tag RM%%M --seeds 3,4,5,6,8,13
)
echo RM_DONE
