@echo off
cd /d C:\aimers
set PYTHONUTF8=1
.venv\Scripts\python.exe src\train_mtnn.py --npz out\tower1_raw.npz --tag PLRV_sigma1 --seeds 42 --epochs 18 --bs 8192 --lr 0.002 --drop 0.15 --qbins 0 --aux-w 0 --plr --plr-freq 32 --plr-dim 8 --plr-sigma 1
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe src\train_mtnn.py --npz out\tower1_raw.npz --tag PLRV_sigma10 --seeds 42 --epochs 18 --bs 8192 --lr 0.002 --drop 0.15 --qbins 0 --aux-w 0 --plr --plr-freq 32 --plr-dim 8 --plr-sigma 10
