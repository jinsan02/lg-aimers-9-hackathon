#!/usr/bin/env bash
# Launch a detached long job on desktop-5070.
#
# The 5070's schtasks was recorded as broken on 2026-08-12: the task creates,
# `/run` reports success, and `Last Result` stays 267011 while nothing executes.
# 2026-08-15 found the cause, and it is not schtasks.
#
#   `schtasks /create` without `/ru` registers the task **Interactive only**, to
#   run as the creating user when that user is logged on. ssh runs as the local
#   account `jinsan`; the console session belongs to
#   `AzureAD\노진산(컴퓨터공학부)`. Those are different users, so the task waits
#   for a session that never appears. 267011 is SCHED_S_TASK_HAS_NOT_RUN -- it
#   is not an error, it is "still waiting".
#
# Registering under SYSTEM removes the session requirement entirely, so the job
# survives ssh disconnect, logoff and the laptop being shut down -- the property
# the 5070 was thought not to have.
#
# Two escaping traps, both of which cost a launch tonight:
#   * the remote layer eats one level of backslashes, so the path handed to this
#     script must already be doubled. Quote it and use two:
#       bash tools/run5070.sh RANK16 'C:\\aimers\\scripts\\x.bat' 'C:\\aimers\\out\\x.log'
#     Check the result: `schtasks /query /tn <name> /v /fo list | grep "Task To Run"`
#     must show single backslashes. `C:aimersscriptsx.bat` means it was eaten.
#   * SYSTEM has no user PATH. Call the interpreter by absolute path inside the
#     batch (`C:\aimers\.conda\python.exe`), never bare `python`.
#
# Each arm inside the batch must still write its own out\<tag>.log and
# out\<tag>.exit -- exit code 0 from the task says nothing about the arms.
#
# Usage: bash tools/run5070.sh <name> '<bat path>' '<log path>' [host]
set -u

NAME="Aimers$1"
BAT="$2"
LOG="$3"
H="${4:-desktop-5070}"

ssh "$H" "schtasks /delete /tn $NAME /f" > /dev/null 2>&1
ssh "$H" "schtasks /create /tn $NAME /tr \"cmd /c $BAT > $LOG 2>&1\" /sc once /sd 2099/01/01 /st 03:00 /ru SYSTEM /rl HIGHEST /f" | tail -1
echo "registered as:"
ssh "$H" "schtasks /query /tn $NAME /v /fo list" 2>/dev/null | tr -d '\r' \
  | grep -iE "task to run|run as user"
ssh "$H" "schtasks /run /tn $NAME" | tail -1
echo "launched: $NAME   (next automatic fire 2099-01-01, i.e. never)"
echo "watch:  ssh $H 'schtasks /query /tn $NAME /v /fo list' | grep -i 'last result'"
echo "        267009 = running, 267011 = never started, 0 = finished"
echo "clean:  ssh $H 'schtasks /delete /tn $NAME /f'"
