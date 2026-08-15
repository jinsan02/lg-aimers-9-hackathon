@echo off
REM Separate the row axis from the tree axis for the unscoreable rank model.
REM
REM Established: the real .cbm (870,752 rows, 1224 trees, 5,209,800 B) segfaults
REM on predict with 100 rows, on CPU, on two machines. Small models are fine.
REM
REM Point 1 is already done and rules out two candidates: at 60,000 rows and
REM 1200 trees the model is 5,067,284 B -- the same tree count and the same size
REM as the failing one -- and it scores in 0.0s. So neither tree count nor model
REM size is the trigger. That leaves the row axis: the CTR tables built from
REM 870k rows.
REM
REM   (870k, 300)   row axis alone
REM   (870k, 1200)  the real configuration -- the positive control. If this one
REM                 also scores, the fault is not the configuration at all but
REM                 the run that produced the shipped .cbm, which spun and was
REM                 force-killed.
REM
REM Detached under SYSTEM so it survives ssh disconnect and the laptop sleeping.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

%PYU% tools\rank_ctr_probe.py --rows 870000 --iters 300  --only full254 > out\rankscale_870k_300.log 2>&1
echo %ERRORLEVEL% > out\rankscale_870k_300.exit

%PYU% tools\rank_ctr_probe.py --rows 870000 --iters 1200 --only full254 > out\rankscale_870k_1200.log 2>&1
echo %ERRORLEVEL% > out\rankscale_870k_1200.exit

echo DONE > out\rankscale.done
