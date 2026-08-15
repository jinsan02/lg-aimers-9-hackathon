@echo off
REM Supervisor for the RK16S3B stage-12 worker.
REM
REM The stage-1 run proved this is not optional: after announcing, the worker
REM did NOT exit. Its batch never wrote RK16S3B_scout.exit or the .done file, so
REM _hard_exit's TerminateProcess failed for the third time, and the only thing
REM that ended it was this task issuing schtasks /end.
REM
REM Timeout 5400s. Stage 12 fits 952 fixed trees where the scout fit 1452 with
REM early stopping in 21 minutes, so it should be well under half an hour.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe

%PY% -u tools\supervise_stage.py --stage rank12_RK16S3B_sel --run-id 20260815c --task AimersRK16S3B12 --timeout 5400 --poll 5 > out\RK16S3B_sel_sup.log 2>&1
echo %ERRORLEVEL% > out\RK16S3B_sel_sup.exit
echo DONE > out\RK16S3B_sel_sup.done
