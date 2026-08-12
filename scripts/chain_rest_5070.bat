@echo off
REM Remainder of the attribution chain after the ssh connection was reset mid
REM P1C_cell. Launch this DETACHED, never as an ssh foreground command:
REM
REM   ssh desktop-5070 "powershell -NoProfile -Command \"Start-Process -WindowStyle Hidden C:\aimers\scripts\chain_rest_5070.bat\""
REM
REM A foreground `ssh ... cmd /c bat` dies with the connection. On 2026-08-13
REM the link reset with the laptop lid closed -- sleep was disabled but the
REM network adapter still dropped -- and it killed P1C_cell mid-refit and ANCH
REM before it started. Disabling sleep is not enough; the job must not be a
REM child of the session.
cd /d C:\aimers
call scripts\run_judge_5070.bat P1C "3,4,5" --two-stage-artifact
call scripts\run_judge_5070.bat ANCH "3,4,5" --anchor-last-pitch
echo DONE > out\CHAIN_REST.done
