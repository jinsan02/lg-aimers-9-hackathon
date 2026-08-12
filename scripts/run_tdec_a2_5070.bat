@echo off
REM A2 convergence arms. TDEC1 stopped at epoch 2 of a 15-epoch OneCycle schedule
REM (warmup is 1.5 epochs) so it died at peak LR and never annealed. One change per arm.
cd /d C:\aimers
if not exist out mkdir out
if not exist model mkdir model
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe

REM arm 1 -- schedule only: longer patience, shorter warmup
%PY% src\train_tabdecoder.py --tag TDEC2_sched --seed 42 --epochs 15 --patience 8 --pct-start 0.03 > out\TDEC2_sched.log 2>&1
echo %ERRORLEVEL% > out\TDEC2_sched.exit

REM arm 2 -- learning rate only (schedule as arm 1, so the lr effect is isolated)
%PY% src\train_tabdecoder.py --tag TDEC3_lr --seed 42 --epochs 15 --patience 8 --pct-start 0.03 --lr 1e-4 > out\TDEC3_lr.log 2>&1
echo %ERRORLEVEL% > out\TDEC3_lr.exit

REM arm 3 -- mask only (schedule as arm 1). The causal mask over table columns was
REM never justified; removing it is the one clean difference from the closed FT axis.
%PY% src\train_tabdecoder.py --tag TDEC4_nomask --seed 42 --epochs 15 --patience 8 --pct-start 0.03 --no-causal-mask > out\TDEC4_nomask.log 2>&1
echo %ERRORLEVEL% > out\TDEC4_nomask.exit

echo DONE > out\TDEC_A2.done
