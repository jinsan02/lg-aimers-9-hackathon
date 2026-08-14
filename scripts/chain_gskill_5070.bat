@echo off
REM GENERAL_SKILL_ADD -- add the season-level skill pack alongside the count one.
REM
REM One mechanism: --feat-skill is added, --feat-skill-pc stays. fpipe.skill_axes
REM then returns ['', 'count'] instead of ['count'], so the frame gains
REM skill_hat and skill_hat_vs_std and goes 121 -> 123 features.
REM
REM Controls are CTRL_base and NULLC_cell, already on this host from earlier
REM today: same surface, same six seeds, fingerprint 0.5352282202778269 on all
REM four families, argv differing by --feat-skill alone. They are not re-run.
REM
REM Base runs first, but the cell arm is NOT conditional on it. base and cell
REM have split signs on the same feature change before (H1: base +3.88,
REM cell -1.55), and the pre-registered hypothesis is the pack in both arms.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --val-season 2024
set CELL=--depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005
set SEEDS=3,4,5,6,8,13

%PYU% src\train_gbdt2.py %CORE% --p1 --feat-skill --depth 8 --seeds %SEEDS% --tag GSKILL_base > out\GSKILL_base.log 2>&1
echo %ERRORLEVEL% > out\GSKILL_base.exit

%PYU% src\train_gbdt2.py %CORE% --p1 --feat-skill %CELL% --seeds %SEEDS% --tag GSKILL_cell > out\GSKILL_cell.log 2>&1
echo %ERRORLEVEL% > out\GSKILL_cell.exit

echo DONE > out\GSKILL.done
