# 제출·실험 기록 정합성 감사 — 2026-08-12

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
