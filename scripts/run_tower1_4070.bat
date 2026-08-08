@echo off
cd /d C:\aimers
set PYTHONUTF8=1
C:\Progra~1\Python312\Scripts\uv.exe pip install --python C:\aimers\.venv\Scripts\python.exe torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b %errorlevel%
if not exist out\tower1_raw.npz .venv\Scripts\python.exe src\train_gbdt2.py --model cat --feat-v2 --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --val-season 2024 --dump-npz out\tower1_raw.npz --tag TW1_dump
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe src\train_twotower.py --npz out\tower1_raw.npz --meta out\tower1_raw_meta.pkl --tag TW1_content --seed 42 --epochs 18 --bs 8192 --lr 0.002
