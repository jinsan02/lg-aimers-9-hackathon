@echo off
cd /d C:\aimers
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
.venv\Scripts\python.exe tools\make_matchup_constants.py
