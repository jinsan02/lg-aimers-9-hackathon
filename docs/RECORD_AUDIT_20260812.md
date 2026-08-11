# 제출·실험 기록 정합성 감사 — 2026-08-12

## SR1·MDU1: champion 보존형 core 잔차 검정

### SR1 — 강수축 raw-feature residual learner

R리그만 사용해 2022 K0 잔차를 학습하고 2023에 고정 검증한 뒤, 2022+2023으로
재적합해 2024 local/strong 아날로그를 평가했다. CatBoostRegressor는 `best_iter=0`에서
멈췄다. 고정 2% 결합의 BSS 증분은 다음과 같다.

| 표면 | 전체 | R 전반 | R 후반 |
|---|---:|---:|---:|
| source 2023 | −0.038 | −0.015 | −0.068 |
| target local 2024 | −0.012 | −0.004 | −0.027 |
| target strong 2024 | −0.014 | −0.004 | −0.032 |

공식 raw 피처에 K0가 남긴 일반화 가능한 잔차가 보이지 않는다. 하이퍼파라미터를 늘리면
best-iter 0을 source 선택편의로 바꿀 뿐이므로 종료한다.

### MDU1 — base/cell disagreement 기반 수축·routing

base−cell 차이의 signed q8, absolute q8, sign×abs q4, base probability×signed q4를
전년도 K0 잔차로 맞추고 다음 시즌에 고정 적용했다. signed-q8 25%는 source 2023
`−2.349`, strong 2024 `−2.106`; abs-q8 25%는 source `−0.303`, strong 전체
`+0.252`이나 R `−1.747`, R 후반 `−0.848`이었다. disagreement는 예측 불확실성의
안정된 방향 정보가 아니므로 추가 bin/scale 탐색 없이 종료한다.

## v11/v12 제출 스크립트 소스 감사

- `submissions/v11_pb_posix_0809.zip/script.py`는 정규화 후
  `out/v11_script_reference.py`와 완전 동일하다. `asof_pitcher_prev5_game_middle_rate`와
  `matchup_constants_2024.npz`를 쓰는 실제 recent-middle+PB 챔피언이다.
- `submissions/v12_career_pb_0809.zip/script.py`는 감사 전
  `src/script_blend_v11.py`와 완전 동일했다. `asof_pitcher_middle_rate`와
  `final_constants_2024.npz`를 쓰는 실패한 career-middle+PB다.
- 따라서 기존 LB 점수에는 패키징 오류가 없다. 저장소 소스 이름만 잘못되어 있었고,
  `src/script_blend_v11.py`를 실제 v11로 복원하고 career 버전을
  `src/script_blend_v12.py`로 분리했다. 두 파일은 각각 제출 ZIP과 다시 완전 동일함을 확인했다.

## EV1 — 동일 base 계열 시드 분산

base-cell 차이가 아니라 동일 depth8 base의 random seed 4개가 만드는 행별 표준편차를
epistemic uncertainty로 검정했다. source 2023 R과 target 2024의 평균 분산은
`.003203→.003390`, p90 `.005545→.005814`로 안정적이었다.

| 고정 보정 | 전체 | R | F | 전반 | 후반 |
|---|---:|---:|---:|---:|---:|
| q8 k5000 ×.25 | −0.966 | −1.089 | −0.043 | −0.987 | −0.939 |
| 0.5방향 수축 ×.25 | +1.369 | +0.823 | +5.466 | +4.255 | −2.401 |
| 0.5방향 수축 ×.50 | +1.357 | +0.392 | +8.594 | +7.171 | −6.240 |

분산 크기는 이전되지만 어떤 방향으로 보정해야 하는지는 후반기에 반전한다. 제출형 +3과
전후반 비악화 조건을 모두 못 넘으므로 추가 seed·bin·scale 탐색 없이 종료한다.

재현: `tools/ensemble_variance_audit.py`.

## BTP1 — CatBoost Bernoulli bootstrap

기본 Bayesian bootstrap 대신 Bernoulli subsample 0.8만 단일 변경했다. 로컬 RTX 5060,
seed42, drop-F-pre 2022, val2023→test2024로 MVA_native와 동일 표면이다.

| 비교 | source 2023 | target 2024 |
|---|---:|---:|
| 단독 BTP1−MVA_native | −13.63 | −14.19 |
| K0에서 base 10% 교체 | −0.159 | −0.107 |
| base 25% 교체 | −0.497 | −0.413 |
| base 50% 교체 | −1.324 | −1.314 |
| base 100% 교체 | −3.969 | −4.575 |

10% target 세그먼트도 R `−0.414`, F `+2.192`, 전반 `−0.378`, 후반 `+0.248`이다.
F 다양성은 있으나 행 비중과 크기가 작고 R 손실을 못 갚는다. 기본 bootstrap을 유지한다.

재현: `tools/bootstrap_core_audit.py`; 학습 태그 `BTP1`.

## LB 제출 원장

사용자가 전달한 실제 평가 서버 결과를 기준으로 최근 제출을 다시 대조했다.

| 제출 | 변경 | LB | v11 대비 | 실행 |
|---|---|---:|---:|---:|
| v10 | v18 + frozen recent-5-game middle q8/k500 | 1100.0891947834 | −1.7128724231 | 32초 |
| **v11** | v10 + frozen pitcher×batter residual k500 | **1101.8020672065** | 0 | 33초 |
| v12 | recent-middle을 career-middle q8/k200으로 교체 | 1083.5307698831 | −18.2712973234 | 33초 |
| v13 | masked-pitch auxiliary MLP, full-fit artifact | 1099.4652219091 | −2.3368452974 | 41초 |
| v14_pitch_mtl_lag | masked-pitch auxiliary, artifact≤2023 | 1099.2989837237 | −2.5030834828 | 42초 |

`v14_pitch_mtl_lag`는 과거의 “v14 셀 depth8” 실험과 이름만 겹친다. 이번 문서부터
반드시 전체 태그로 구분한다. v13·v14 모두 ZIP 구조, 압축 해제 smoke, 행 독립 감사와
평가 실행 시간이 정상이므로 제출 오류가 아니다.

## v14 점수 판단

- v14−v13은 `−0.1662381854`로, 전처리 cutoff 수정이 손실을 회복하지 못했다.
- full-fit 과거 재현은 이미 2022→23 `−198.662`, 2023→24 `−123.688`이었다.
- lagged-artifact는 과거 rolling에서 양수였지만 실제 2024→25에서는 v13과 같은 폭으로
  하락했다. 따라서 “cutoff만 원래 구조로 되돌리면 일반화한다”는 설명도 반증됐다.
- LB 두 점으로 NN 가중치나 gate를 다시 맞추는 것은 정답 세트 적합이므로 하지 않는다.
- 현행 챔피언은 v11 `1101.8020672065`다.

## 문서 정합성 확인

다음 파일의 v14 `CANDIDATE/LB 미관측` 표현을 실제 점수와 CLOSED 판정으로 교체했다.

- `AGENTS.md`
- `EXPERIMENT.md`
- `HANDOFF.md`
- `docs/EXPERIMENTS_LOG.md`
- `docs/PITCH_MASKED_MTL_20260811.md`
- `docs/PITCH_MTL_LB_FAILURE_20260811.md`
- `docs/SETTLED.md`

## 다음 core-resolution 파일럿 — HFC1

기존 flat 14-cell softmax가 희귀 실패조합을 각 클래스로 독립 취급한다는 점을 겨냥해,
`middle → ball|middle → reverse|middle,ball → success|세 모드` 네 조건부 이진 모델로
joint 확률을 분해했다. 추론에서는 테스트 실패모드를 쓰지 않고 8개 잠재 상태를 행별로
모두 주변화했다. 동일 NPZ·RTX 5060·seed42·depth5·800 iter의 flat 대조군과 비교했다.

| 모델 | 2023 검증 BSS | 미학습 2024 BSS | centered 2024 |
|---|---:|---:|---:|
| flat 14-cell | 592.131 | 586.631 | 588.058 |
| hierarchical chain | 591.982 | 514.976 | 516.307 |
| chain−flat | −0.149 | **−71.655** | **−71.751** |

예측 RMS는 검증/테스트 `.014223/.014322`로 다양성은 충분했다. 그러나 flat에 chain을
25% 고정 결합한 값도 2023 `+15.135`가 2024 `−2.519`로 반전했다. 실패모드 조건부
calibration이 시즌을 넘지 못하므로 1시드 gate에서 즉시 종료하고 다중시드는 돌리지 않는다.

재현: `tools/fm_chain_pilot.py`, `out/HFC1_s42_preds.npz`.

## CatBoost split randomness — CRS0

현행 로컬 기준 `MVA_native`에서 `random_strength=1→0`만 바꿨다. 검증 BSS는
`611.741→603.95`, 미학습 2024는 `876.899→861.12`로 각각 `−7.79/−15.78`이었다.
대규모 표본에서도 split-score 무작위성은 낭비가 아니라 필요한 정규화다. 0과 1 사이를
추가 탐색하면 단일 실패점 뒤의 연속 하이퍼파라미터 튜닝이 되므로 확대하지 않는다.

## 기존 로컬 후보 고정 블렌드 전수 스캔

`tools/local_blend_sweep.py`로 2023 validation과 refit 후 미학습 2024 test 산출물이 모두
존재하고 타깃 배열이 정확히 같은 59개 후보를 `MVA_native`에 5/10/20/50% 고정 결합했다.

- CatBoost 파생 후보의 두 표면 동시 양수 최대치는 20%에서 `QMIN1 +1.756/+0.879`,
  `FCC1 +1.505/+0.867`로 모두 +3 gate 미만이었다.
- 강한 다양성은 masked-pitch MLP뿐이었다. 6시드 평균 PMT0은 20%에서
  `+22.184/+6.198`, PMT1 auxiliary는 `+28.235/+10.004`였다.
- 이 MLP 경로는 v13/v14로 실제 제출돼 두 번 모두 `−2.34~−2.50` 하락했다. 따라서 이
  스캔은 새로운 제출 후보가 아니라, 로컬 base-only 블렌드가 실제 base+cell+후처리
  일반화를 과대평가한다는 재확인이다. LB를 보고 10→20%로 올리지 않는다.

다음 감사에서는 로컬의 정식 depth5 cell 기준선을 재생성해 base-only가 아닌 실제
base+cell core 위에서 후보 기여를 다시 제한한다.

### 정식 로컬 base+cell 재감사

동일 MVA 설정의 depth5 14-cell을 3000 iter, refit 4500 iter로 재생성했다.

| 구성 | 2023 BSS | 미학습 2024 BSS |
|---|---:|---:|
| MVA_native base | 611.741 | 876.899 |
| MVCELL_s42 | 606.76 | 890.63 |
| **base .45 + cell .55** | **625.806** | **900.175** |

이 core 위에서 59개를 다시 고정 결합하자 모든 CatBoost 파생 후보는 2024 음수였다.
PMT1 auxiliary 6시드 평균은 10%에서 2023 `+12.345`, 2024 `+2.793`으로 +3 gate 바로
아래였고, 20%는 `+22.780/+2.261`로 더 약해졌다. 개별 seed3의 큰 값은 선택편의다.
v13/v14 실제 LB 반증까지 있으므로 masked MLP 재제출 근거가 아니다.

## 예측 구종확률 파생피처 — PUG1/PUP2

Trackman≤2022로 2023, Trackman≤2023으로 2024의 행단독 3구종 확률을 만들었다. 실제
구종이나 평가행 간 정보는 쓰지 않았다. 2023 전반↔후반 교차적합 잔차와 2024 전이를
동시에 비교했다.

| 신호·고정 scale | 2023 반분 CV | 2024 | centered 2024 |
|---|---:|---:|---:|
| entropy 1.0 | −3.095 | +0.407 | +0.342 |
| top1 margin .25 | −1.480 | −0.137 | −0.138 |
| breaking probability .5 | +1.664 | +1.601 | +1.244 |
| offspeed probability .5 | +1.376 | +3.212 | +2.803 |
| fastball×breaking q4, .25 | **+2.797** | **+2.602** | **+2.384** |
| breaking×offspeed q4, .25 | +2.722 | +1.115 | +0.914 |

구종 선택 불확실성 자체는 제구 난이도의 안정 신호가 아니었다. 구종확률 구성에는 작고
같은 부호의 관계가 있지만 최선도 두 표면 +3을 동시에 넘지 못한다. 여기서 q·k·scale을
더 고르는 것은 검증 최적화이므로 CatBoost 피처 재학습과 제출로 승격하지 않는다.

재현: `tools/pitch_uncertainty_gate.py`, `tools/pitch_proba_pair_gate.py`,
`out/pitch_uncertainty_gate.json`, `out/pitch_proba_pair_gate.json`.

## 챔피언 cell iteration 상한 — MVCELL5K

과거 depth5 cell 대부분과 이번 MVCELL이 best_iter 2990~2999로 3000 상한에 걸린 점을
근거로, 다른 설정을 고정하고 `iters 3000→5000`만 바꿨다. 검증 최적점은 실제로 3928로
이동했지만 refit 후 미학습 시즌에서 반전했다.

| 비교 | 2023 | 미학습 2024 |
|---|---:|---:|
| MVCELL5K−MVCELL 단독 | +1.63 | **−8.67** |
| core 내 새 cell 10% 치환 | +0.357 | −0.178 |
| core 내 새 cell 50% 치환 | +1.564 | −1.093 |
| core 내 새 cell 100% 치환 | +2.575 | −2.688 |

old/new cell 예측 RMS는 val/test `.003022/.002882`로 차이는 작지만 모든 고정 치환이
다음 시즌 음수다. 3000 상한은 최적점을 못 찾은 것처럼 보여도 시즌 전이 관점에서는
유효한 정규화다. 기존 3000 tree cell을 유지하고 추가 iteration 탐색은 닫는다.

## 약신호 salvage · 새 core geometry 후속

현행 recent-middle+exact-PB의 정직한 rolling 아날로그 `K0` 위에서만 증분을 다시
측정했다. 5000-iter cell을 source-2023 양수 세그먼트에만 적용한 10개 gate는 target
최고가 7회 이후 inning 그룹의 `+0.133`뿐이었다. 예측 구종확률도 K0 뒤에는
offspeed 최대 `+0.822`, fastball×breaking `+1.346`, 고정 등가결합 `+1.526`으로
축소됐다. 약한 신호끼리 합치거나 세그먼트 routing해도 제출 gate를 넘지 못한다.

새 출력 기하로 compact FT-Transformer(수치 field token, 범주 embedding token,
CLS, dim32·2 layers·4 heads)를 현재 121피처 exact NPZ에 학습했다. val2023 최고는
epoch2 `499.903`, refit 후 미학습 2024는 `712.940`; K0와 RMS `.022445`였지만
2% blend가 source `−8.566`, target `+0.098`, 5%부터 target도 음수였다. 다양성보다
성능 격차가 커서 단일 파일럿으로 종료한다.

## Brier early-stop · 현재 121피처 XGBoost

Logloss 학습은 그대로 두고 early stopping metric만 `BrierScore`로 바꾼 BSE1은
기준과 best_iter가 정확히 같은 `826`이었다. val/test도 `611.77/876.87` 대
`611.74/876.90`으로 사실상 동일해 조기종료 목적 불일치는 남은 레버가 아니다.

XGBoost는 과거 feat-v2 시절 블렌드 기여 기록은 있었지만 E99/E95/skill_pc가 포함된
현재 121피처로는 미실행이었다. 테스트 예측에 pandas를 직접 넘겨 죽던 DMatrix 버그를
수정하고 세 rolling 전이를 새로 만들었다. 또한 refit이 `refit_mult`를 무시하던 결함을
수정했다(`1.0→1.5`: 2024 단독 `816.318→828.82`).

K0에 XGB seed42를 10% 고정 결합하면 2021→22/2022→23/2023→24가 각각
`+2.640/+20.294/+3.867`; 최신 6시드 개별 증분은 `+3.092~+4.157`, 평균
`+3.711`, 앙상블 `+3.749`였다. 그러나 이는 약한 로컬 single-seed Cat core에서의
보완 효과였다. 실제 출하 자산과 가까운 4070 `VB2_base` 8시드 + `ZD5` 6시드에
source-2023 K0를 동결 적용한 강한 아날로그는 BSS `973.157`이었다. 그 위 XGB6은
2/5/10%가 `−0.053/−0.369/−1.526`; refit1.5 XGB도 2% `+0.094`뿐이며
R `−0.287`, late `−0.247`였다. 따라서 다중시드 착시가 아니라 챔피언이 이미 같은
resolution을 더 잘 먹은 것이며 제출 후보로 승격하지 않는다.

주의: 중간에 `matchup_constants_2024.npz`를 2024 검증행에 되붙여 BSS `1402`가 나온
계산은 **무효**다. 이 파일은 2025 제출용으로 2024 라벨/OOF 잔차에 적합된 상수다.
역사 검증에는 반드시 source-2023에서 새로 동결한 K0만 사용한다.

재현: `tools/weak_signal_salvage.py`, `src/train_fttransformer.py`,
`tools/xgb_core_transfer.py`, `out/weak_signal_salvage.json`.
