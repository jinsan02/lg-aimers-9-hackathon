@echo off
REM B4/B5 on desktop-5070. Judging surface, 6 seeds, paired against a baseline
REM built on THIS machine -- MVN3 is a laptop artefact and the machine effect is
REM 11 points (AGENTS.md rule 5), so it cannot be the comparison arm here.
cd /d C:\aimers
if not exist out mkdir out
if not exist model mkdir model
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set B=--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 --val-season 2023 --test-season 2024 --seeds 3,4,5,6,8,13

REM [0] same-machine baseline. Everything below is paired against this.
%PY% src\train_gbdt2.py %B% --tag N5_base > out\N5_base.log 2>&1
echo %ERRORLEVEL% > out\N5_base.exit

REM [B4] drop 2019-2020 (481,500 rows, 32.64%). The filter sits before fpipe.fit,
REM so the TE/std/prior tables change too -- a member_fingerprint mismatch here is
REM expected, not an alarm.
%PY% src\train_gbdt2.py %B% --min-season 2021 --tag N5_min2021 > out\N5_min2021.log 2>&1
echo %ERRORLEVEL% > out\N5_min2021.exit

REM [B5-1] drop the three columns measured to carry zero residual information
REM (one unique value per situation, width 0.0000).
%PY% src\train_gbdt2.py %B% --drop-cols li,home_win_expectancy,away_win_expectancy --tag N5_wexp > out\N5_wexp.log 2>&1
echo %ERRORLEVEL% > out\N5_wexp.exit

REM [B5-3] --feat-v4: context stats keyed on (pitcher_id, base_empty). Wired since
REM features.py:223 but never run. Requires --feat-v2 (assert at train_gbdt2.py:1215).
%PY% src\train_gbdt2.py %B% --feat-v4 --tag N5_v4ctx > out\N5_v4ctx.log 2>&1
echo %ERRORLEVEL% > out\N5_v4ctx.exit

echo DONE > out\N5_B4B5.done
