@echo off
REM One GPU job at a time: the four single-change arms, run back to back.
REM
REM Each isolates one flag against the fixed B0-JL core so the delta attributes
REM to that flag. --p1 bundles the first three and cannot say which one moved.
REM
REM Chaining these through `ssh "cmd /c a & b & c"` does not work -- the nested
REM quotes around the seed list come through as \"3,4,5\" and cmd answers
REM "4 was unexpected at this time", having run nothing. Keep it in a file.
cd /d C:\aimers
call scripts\run_judge_5070.bat P1A "3,4,5" --te-fit-prior
call scripts\run_judge_5070.bat P1B "3,4,5" --skill-neutral-first
call scripts\run_judge_5070.bat P1C "3,4,5" --two-stage-artifact
call scripts\run_judge_5070.bat ANCH "3,4,5" --anchor-last-pitch
echo DONE > out\CHAIN_ATTRIB.done
