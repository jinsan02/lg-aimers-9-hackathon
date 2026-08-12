@echo off
cd /d C:\aimers
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist out\audit_v12final mkdir out\audit_v12final
tar -xf submissions\v12_career_pb_0809.zip -C out\audit_v12final
.venv\Scripts\python.exe tools\audit_rowindep.py out\audit_v12final\script.py
