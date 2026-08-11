# 전수 데이터 감사 — 2026-08-11

## 결론

공개된 대회 데이터를 전부 다시 읽고 시간 전이 기준으로 조사했다. 공식 입력의 원본
결측은 **0, -1, 평균, 중앙값으로 채우지 않고 CatBoost native NaN으로 두는 것이 현재
최선**이다. 파생 누적 통계만 학습구간 prior와 표본수 0으로 수축하는 현행 혼합 정책을
유지한다.

새 관계 가운데 가장 확실한 것은 `pitcher_id × batter_id` 잔차다. 독립적인 두 전이에서
각각 `+1.874`, `+4.595 BSS`이고 전반/후반/R/F가 모두 양수였다. 다만 이것은 이미 v11의
exact-PB(k500)에 들어 있다. 최근 성공률과 통산 success/reverse의 쌍 관계도 통계적으로
강하지만 명시적 곱/차 피처로 넣으면 약 `-9.3`점이었다. 3중 조합, 결측 대체, Trackman
전년도 요약, CatBoost 고차 CTR에서는 새 제출 후보가 나오지 않았다.

즉 이번 조사의 실질적인 결과는 “쓸 피처가 없다”가 아니라 다음 세 가지다.

1. v11의 PB와 v10의 recent-middle이 우연한 선택이 아님을 원자료 전수 감사가 재발견했다.
2. 단순 파생변수 추가가 아니라 현행 모델의 **조건부 해상도(resolution)**를 개선해야 한다.
3. 결측치를 임의 숫자로 바꾸거나 결측 표시자를 강조하면 오히려 일반화가 나빠진다.

## 조사 범위와 원칙

- `train.csv`: 1,475,092행, 공식 입력 47개 + `row_id` + target, 2019~2024 전 행
- `trackman_history.csv`: 1,793,078행, 30개 컬럼 전 행
- `test.csv`: 실제 2025 데이터가 아니라 5행 스키마 샘플이므로 입력 컬럼 확정에만 사용
- 시간 전이: `≤2021→2022`, `≤2022→2023`, `≤2023→2024`
- 단변수 47개 전부, 2024 기준 1,081개 모든 쌍, 상위 235개 쌍의 3개 전이 재검증,
  안정 쌍 × 야구 맥락 11개의 선택적 3중 조합
- 모델 잔차에서는 동일 로컬 RTX 5060, 동일 시드끼리만 비교
- 2025 다른 평가행, row 순서, 외부 데이터, LB 상수 맞추기는 사용하지 않음

실제 2025 평가값은 평가 서버에만 있으므로 “대회 데이터 전부”는 **공개된 공식 파일의
모든 행**을 뜻한다. 5행 샘플을 2025 분포처럼 해석하지 않았다.

## 1. 전체 분포와 체제 변화

전체 target 평균은 `0.523766`이다. 성공률은 시즌과 리그에 따라 크게 움직인다.

| 시즌 | F 성공률 | R 성공률 |
|---:|---:|---:|
| 2019 | .68925 | .54949 |
| 2020 | .58777 | .52692 |
| 2021 | .70384 | .51276 |
| 2022 | .70875 | .50369 |
| 2023 | .47290 | .50312 |
| 2024 | .45928 | .48971 |

따라서 과거 전체 평균으로 만든 단변수 lookup은 2024 수준 편향 때문에 음수 BSS가 쉽게
나온다. 이 문서에서 `raw`는 실제 배포 가능한 점수이고, `centered`는 target 시즌 평균
오차만 진단용으로 제거해 순수 분별력을 보는 값이다. 후보 채택에는 raw와 모델 잔차를
사용하고 centered만으로 채택하지 않았다.

## 2. 결측치 전수 감사

결측이 있는 공식 입력은 47개 중 16개다.

| 결측 패턴 | 열 수 | 전체 결측률 | 의미 |
|---|---:|---:|---|
| prev1/3/5 success·middle | 6 | 1.9785% | 최근 경기 이력이 없는 투수 |
| batter success·middle | 2 | 0.0563% | 타자 cold-start |
| pitcher success/reverse/middle/ball/strike + pitch mix | 8 | 0.0537% | 투수 cold-start |

최근 경기 결측률은 2019 `5.17%`, 2020 `1.58%`, 2021 `1.51%`, 2022 `1.19%`,
2023 `1.11%`, 2024 `1.44%`다. 결측 자체의 target 차이는 최근 경기군이 시즌별
`-0.0244~+0.0266`, 투수 cold-start가 `-0.0787~+0.0764`로 부호가 바뀐다.
타자 cold-start는 평균 `+0.0702`이지만 전체의 0.056%뿐이다. 즉 결측 여부를 성공/실패
상수로 해석할 수 없다.

### 현재 코드의 실제 처리

- 공식 원본 수치 NaN: CatBoost에 그대로 전달(`native`). 0/평균 대체가 아니다.
- `feat-v2` shrink 통계: 학습구간의 계층 prior로 채우고 관측수 `n=0`을 함께 제공.
- `feat-std`: 복원 불가능한 count는 0, rate는 학습 prior로 채움.
- TE cold-start: NaN을 유지해 CatBoost가 분기.
- 선형 skill 보조 추정기: 해당 보조모델 내부에서 학습 중앙값을 사용.

### 같은 머신·같은 seed42 대체값 비교

기준 `native=876.90`에 대한 2024 BSS 차이다.

| 방식 | BSS | 기준 대비 |
|---|---:|---:|
| native NaN | 876.90 | 0.00 |
| native + 결측 표시자 | 870.32 | -6.58 |
| -1 | 869.85 | -7.05 |
| 계층 의미 대체 + 표시자 | 869.43 | -7.47 |
| 0 | 869.11 | -7.79 |
| 학습 평균 | 867.00 | -9.90 |
| 최근→통산 fallback + 손/리그 prior | 866.73 | -10.17 |
| 학습 중앙값 | 864.83 | -12.07 |

가장 덜 나빴던 `native + 표시자`를 seed 3/4/5로 다시 비교한 차이는
`-4.520/+1.348/-1.824`, 평균 `-1.665`, 3시드 앙상블 `-1.348`이다. 명시적 대체나
표시자가 native보다 낫다는 증거가 없으므로 현행을 유지한다.

## 3. 컬럼과 쌍 관계

시간 전이 3개에서 raw 점수와 단변수 대비 synergy가 모두 양수인 핵심 쌍은 아래다.

| 쌍 | 평균 raw | 최소 raw | 평균 synergy | 최소 synergy |
|---|---:|---:|---:|---:|
| reverse rate × prev3 success | 432.16 | 96.48 | 128.36 | 65.46 |
| career success × prev3 success | 432.87 | 32.39 | 83.71 | 63.27 |
| reverse rate × prev5 success | 427.12 | 87.14 | 123.32 | 56.12 |
| career success × prev1 success | 416.51 | 20.01 | 67.34 | 50.89 |
| reverse rate × prev1 success | 392.23 | 65.31 | 88.42 | 34.29 |

해석은 “최근 성공 상태의 의미가 투수의 통산 실패 형태에 조건부”라는 것이다. 그러나
관계가 존재하는 것과 새 피처가 현행 CatBoost에 증분을 주는 것은 다르다. 명시적
`recent success × reverse`는 `-9.29`, `recent success −/× career success`도 `-9.30`
이었다. depth-8 트리와 기존 shrink/form 피처가 이미 이 관계를 흡수하고 있으며, 곱 피처가
학습 경로만 흔든 것으로 본다.

안정 쌍에 볼카운트·아웃·이닝·리그·손·주자·월 등을 붙인 3중 조합은 **세 전이에서
쌍 대비 최소 증분이 양수인 후보가 하나도 없었다**. 조건을 더 쪼개면 표본 분산이 커졌다.

## 4. 현행 모델 잔차에서 남은 관계

121피처 CatBoost의 이전 시즌 검증 잔차를 k500으로 수축해 다음 시즌에 고정 적용했다.

| 관계 | 2022→2023 | 2023→2024 | 전반/후반/R/F |
|---|---:|---:|---|
| pitcher_id × batter_id | +1.874 | +4.595 | 두 전이 모두 전 구간 양수 |
| game_type × num_runners | -303.5 | +3.758 | 체제 반전, 기각 |
| prev5 middle | -18.319 | +10.680 | 전이 반전; v10 외 추가 확장 금지 |
| career middle | -51.891 | +9.093 | 전이 반전, 기각 |

모든 1,081쌍 가운데 두 전이의 전체·전반·후반·R·F를 전부 통과한 것은
`pitcher_id × batter_id` 하나뿐이다. 이 감사가 v11 exact-PB를 독립적으로 재발견했다.
새로운 PB 저랭크/적응 수축 확장은 과거 실험에서 이미 닫혔으므로 현재 k500을 유지한다.

## 5. 고차 범주 조합과 모델 해상도

쌍 관계를 모델이 더 직접 배우게 CatBoost `max_ctr_complexity`를 올렸다.

- complexity 2, seed42: `+1.389`; 과거 R 전이는 `+3.214`, 최신 R은 `+0.763`
- complexity 3, seed42: `+3.338`
- complexity 3 추가 seed 3/4/5: `+4.541/-0.690/-1.479`
- 3시드 평균 `+0.791`, `t=0.419`; 앙상블 증분 `+0.471`
- 과거 R 전반은 `-5.357`로 부호 반전

첫 시드 선택편의였고 제출 기준을 통과하지 못했다. `game_type × num_runners`를 단일 범주
피처로 추가한 실험도 `-11.91`이었다.

## 6. Trackman 179만 행

30개 원본 컬럼을 모두 프로파일했다. 수치 결측은 spin `0.695%`, horizontal break
`0.576%`, IVB `0.561%`, 나머지 주요 수치 약 `0.42~0.44%`다. 현재 투구의 Trackman
측정값은 투구 전 예측에 쓸 수 없으므로, 합법적인 **직전 시즌 투수 요약** 21개
(구속·회전·무브먼트·릴리스 평균/표준편차·구종군 비율·표본수)을 만들었다.

다음 시즌 행 커버리지는 약 77%였다. 모든 단변수의 3전이 최소 raw BSS가 음수였고,
가장 나은 `offspeed mix`도 평균 centered `+6.41`에 그쳤지만 raw는 크게 음수였다.
기존 정식 `--tm-feats` 실험 `-6.04`와 일치한다. 결측 대체가 문제가 아니라 전년도
기계 특성과 다음 시즌 제구 성공의 비정상성·시즌 변화가 병목이다.

## 7. 중복·저가치 컬럼

강한 중복은 다음과 같다.

- `asof_pitcher_n == asof_pitcher_pitchmix_n`: 상관 1.000
- home/away win expectancy: -0.9999998
- score diff와 win expectancy: 절대 상관 약 .904
- prev3/prev5 success: .882, prev3/prev5 middle: .847
- career success와 reverse: -.810
- ball rate와 strike rate: -.774

그렇다고 중복 열을 제거하지는 않는다. 과거 E46 중복 제거가 약 `-1.0`이었고 CatBoost가
같은 정보를 서로 다른 split 경로에서 활용할 수 있기 때문이다.

seed42 기준 CatBoost gain importance 상위는 `season(3.83)`, `batter_team_id(3.03)`,
`pitcher success shrink(3.01)`, `std pitcher success(2.70)`, `pitcher_team_id(2.56)`,
`std batter success(2.25)`, `pitcher×batter-hand TE deviation(2.11)`, `form_delta5(1.97)`다.
반대로 `prev_missing`, `is_monday`, `is_new_pitcher`는 0이었다. 중요도는 인과효과가
아니지만 결측/신인 표시자를 더 강조할 근거가 약하다는 결론과 일치한다.

## 최종 결정

- **유지:** v11 recent-middle + exact-PB, 원본 NaN native 처리, 파생 prior+n0 수축
- **제출 후보 없음:** 이번 감사에서 새 피처나 결측 정책을 챔피언에 추가하지 않음
- **기각/보류:** 숫자 대체 전부, 결측 indicator, 명시적 recent×career, league×runner,
  3중 lookup, prior-year Trackman, CTR complexity 2/3
- **방향:** 후처리나 야구 proxy를 더 붙이기보다 champion OOF 잔차를 이용한 저용량
  core-resolution 개선만 검토. 단, exact-PB와 중복되지 않는 구조여야 함

## 재현 산출물

- 컬럼별 통합표: `out/full_audit/column_audit_summary.csv`
- 결측/분포: `main_profile.csv`, `main_missing_by_season.csv`,
  `missing_target_by_season.csv`
- 단/쌍/삼중: `singleton_transfer*.csv`, `pair_transfer*.csv`,
  `triple_transfer*.csv`
- 모델 잔차: `residual_single_two_transition.csv`,
  `residual_pair_two_transition.csv`
- Trackman: `trackman_profile.csv`, `trackman_missing_by_season.csv`,
  `trackman_singleton_transfer*.csv`
- 실행 코드: `tools/full_dataset_audit.py`, `tools/full_residual_audit.py`
- 모델 실행 전문과 seed별 수치: `LEDGER.tsv`의 `MVA_*`, `MIN3`, `RR_*`,
  `LRUN1`, `CTR2*`, `CTR3*`
