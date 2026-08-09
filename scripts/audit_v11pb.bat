@echo off
cd /d C:\aimers
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist out\audit_v11pb mkdir out\audit_v11pb
tar -xf submissions\v11_pb_posix_0809.zip -C out\audit_v11pb
.venv\Scripts\python.exe tools\audit_rowindep.py out\audit_v11pb\script.py
