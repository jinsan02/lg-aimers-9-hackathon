@echo off
REM Supervisor for the RK16S3B stage-1 worker. Its own scheduled task, on
REM purpose: the worker cannot be trusted to end itself, and a supervisor
REM running inside the worker would share its fate.
REM
REM Waits for out\handoff\rank1_RK16S3B_scout.ready.json under run id 20260815b,
REM re-verifies the model and meta against the sha recorded in it, then ends
REM AimersRK16S3B1 with schtasks /end -- the only termination that has ever
REM worked on this machine -- and writes <stage>.ended.json.
REM
REM Timeout 5400s. The scout fit took about 20 minutes; anything past 90 is not
REM slow, it is stuck, and a timeout does NOT end the task because nothing has
REM been verified and there is no evidence which process to end.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe

%PY% -u tools\supervise_stage.py --stage rank1_RK16S3B_scout --run-id 20260815b --task AimersRK16S3B1 --timeout 5400 --poll 5 > out\RK16S3B_sup.log 2>&1
echo %ERRORLEVEL% > out\RK16S3B_sup.exit
echo DONE > out\RK16S3B_sup.done
