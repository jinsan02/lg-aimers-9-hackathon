@echo off
REM FPRE -- does dropping F-league rows before 2022 help on the SUBMISSION
REM surface? Pre-registered in docs/FPRE_PREREGISTRATION_20260816.md.
REM
REM The champion trains on all 1,475,092 rows, including 105,308 F-league rows
REM from before 2022 (7.14% of the fit set) whose label regime shifts
REM .7087 (2022) -> .4729 (2023) while R is flat. The judging surface already
REM carries --drop-f-pre 2022 and collapses without it, so the flag has only
REM ever been validated where it is mandatory. The submission surface omits it
REM and that omission has never been measured at n=6 on a matched footing.
REM
REM Same-place-as---min-season warning: the filter runs before fpipe.fit, so it
REM truncates the TE / std / anchor tables too. --min-season 2021 lost 95.09
REM through exactly that mechanism. A large negative here is pre-accepted.
REM
REM 24 fits, one session, one host, six shared seeds. Fresh controls on both
REM arms: the champion's stored B1S_base / GSKDEP_cell were produced at an
REM earlier commit and training-path parity is not proven (P3-B precedent).
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set COMMON=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --val-season 2024 --p1
set CELL=--feat-skill --max-train-season 2024 --depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005
set SEEDS=--seeds 3,4,5,6,8,13

echo ==== FPRE control base ====
%PYU% src\train_gbdt2.py %COMMON% --depth 8 %SEEDS% --tag FPRECTL_base
echo ==== FPRE control cell ====
%PYU% src\train_gbdt2.py %COMMON% %CELL% %SEEDS% --tag FPRECTL_cell
echo ==== FPRE candidate base ====
%PYU% src\train_gbdt2.py %COMMON% --depth 8 --drop-f-pre 2022 %SEEDS% --tag FPRE_base
echo ==== FPRE candidate cell ====
%PYU% src\train_gbdt2.py %COMMON% %CELL% --drop-f-pre 2022 %SEEDS% --tag FPRE_cell
echo ==== FPRE done ====
