@echo off
REM ANCH alone. The P1C+ANCH chain wrote P1C.done at 06:16 and then stopped
REM without creating ANCH_base.log at all -- the detached cmd survived a full
REM 17-minute P1C run after its launching ssh closed, so detachment held, but
REM the second `call` never produced output. Not diagnosed; rerunning the one
REM missing arm costs less than finding out why, and the failure is visible
REM (a missing log) rather than silent.
REM
REM Writes CHAIN_REST.done so the waiter started for the original chain fires.
cd /d C:\aimers
call scripts\run_judge_5070.bat ANCH "3,4,5" --anchor-last-pitch
echo DONE > out\CHAIN_REST.done
