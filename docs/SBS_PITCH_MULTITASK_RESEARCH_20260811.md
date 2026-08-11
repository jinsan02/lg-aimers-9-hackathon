# SBS 다음 구종 모델과 제구 멀티태스크 조사 — 2026-08-11

## 결론

SBS가 2017년에 공개한 모델은 **다음 구종 예측기**이지 제구 성공 예측기가 아니다.
우리 데이터에서도 실제 구종은 제구 잔차에 매우 강한 정보였지만, 평가 시점에 합법적으로
예측한 구종확률로 주변화하면 오히려 `−13.04 ~ −28.87 BSS`였다. 따라서 SBS식
시퀀스 모델을 그대로 만들거나 큰 멀티태스크 신경망으로 바로 확장하지 않는다.

남은 연구 가치는 두 가지로 제한한다.

1. 제공 데이터의 1:1 Trackman 구종 라벨 복원은 재사용 가능한 인프라다.
2. 향후 행 단독 입력에서 다음 구종 예측력이 현 53.61%를 **새 정보로** 크게 넘는
   경우에만 작은 residual multitask head를 다시 연다.

현재 챔피언과 제출 후보는 바뀌지 않는다.

## SBS 모델에서 공개된 사실

[SBS 원문](https://programs.sbs.co.kr/sports/sbssportsgolf/article/56051/S10008823953)에
따르면 SBS Sports와 Intel이 2017년 정규시즌 약 22만 구를 학습해 한국시리즈에서 다음
구종을 예측했다. 공개 설명은 전통적 통계·수학 모델과 시계열 학습 신경망을 포함한
3개 알고리즘이라는 수준이며, 세부 구조·구종 클래스·기준선·신뢰구간은 공개되지 않았다.
기사의 60~80%는 경기 중 다음 구종 적중률이라 이 대회의 Brier Skill Score 또는 제구
성공확률과 직접 비교할 수 없다.

학술 연구도 직전 구종·위치와 경기 문맥이 다음 구종 예측에 중요하다고 보고한다.
예: [pitch type/location DNN ensemble](https://journals.sagepub.com/doi/pdf/10.3233/JSA-200559),
[MLB pitch-type ML/DL 연구](https://journals.sagepub.com/doi/10.3233/ATDE251162).
그러나 이 대회 test에는 game_id, pitch_no, pitch_of_pa, 직전 구종이 없고 다른 평가행을
읽어 상태를 갱신하는 것도 행 독립 규칙 위반이다.

## 데이터 연결 감사

기존 `src/linkage_study.py`는 선수 ID 없이 상황 키만 비교해 Trackman 후보가 정확히
1개인 행을 8.9%로 보고했다. 이번에는 이미 검증된 `pitcher_map2.csv`와
`batter_map2.csv`를 적용한 뒤 아래 행 단독 키를 사용했다.

```text
season, month, weekday, inning, top/bottom, balls, strikes, outs,
pitcher_hand, batter_hand, pitcher_id, batter_id
```

전수 감사 결과:

| 항목 | train 행 비율 |
|---|---:|
| Trackman 후보 하나 이상 | 86.4147% |
| Trackman 후보 정확히 1개 | 76.0114% |
| main 1행 : Trackman 1행이며 3분류 구종 라벨 존재 | **75.1668%** |
| 양쪽 그룹 크기 동일 | 86.3575% |
| 동일크기이며 그룹 구종도 단일 | 78.2876% |

실제 gate에 쓴 가장 엄격한 1:1 라벨 커버는 2023 `77.1225%`, 2024 `76.5947%`다.
그룹 크기만 같은 복수행은 공통 game/pitch ID가 없으므로 순서를 억지로 맞추지 않았다.

재현: `tools/audit_joint_pitch_labels.py`, `out/joint_pitch_label_audit.json`.

## SBS 유사 다음 구종 예측 재현

Trackman의 fastball/breaking/offspeed 3분류를 `<=2023`으로 학습하고 미학습 2024에서
평가했다. 현재 투구의 구종·구속·무브먼트·릴리스 값은 입력하지 않았다.

| 구종 모델 | 사용 정보 | 정확도 | Logloss |
|---|---|---:|---:|
| 2024 다수 클래스 기준 | fastball 고정 | 47.044% | 1.05327 |
| Trackman row-local | 행 단독 문맥+선수 | 50.925% | 0.96671 |
| Trackman sequence | 위 + pitch_of_pa + 직전 2구 | **52.701%** | **0.94035** |
| main exact-label head | 대회 입력 47개 전체 | **53.606%** | **0.91551** |

시퀀스의 `+1.776%p`는 실제 야구 신호지만 제출에서 사용할 수 없다. 가장 강한 합법
모델은 `asof_pitcher_offspeed_rate`, 양 선수 손, pitcher_id, season, count와 기존
구종배합을 주로 사용했다. 즉 새 센서 정보가 아니라 현행 제구 모델에도 이미 들어가는
행 단독 정보의 재표현이다.

재현: `tools/pitch_aux_feasibility.py`, `tools/pitch_type_control_gate.py`,
`out/pitch_aux_feasibility.json`, `out/pitch_type_control_gate.json`.

## “구종과 제구를 같이 예측하고 제구만 사용” 검정

정확한 확률 분해는 다음이다.

```text
P(success | x) = Σ_type P(type | x) P(success | x, type)
```

2023의 현행 로컬 기준선 OOF 잔차에서 실제 구종별 강수축 보정을 학습하고, 2024에는
두 경로를 비교했다.

- oracle: 1:1 복원한 **실제 2024 구종**을 넣음. 연구 진단일 뿐 제출 불가.
- deployable: `P(type|x)`로 보정을 가중평균. 오직 현재 행 입력만 사용.

기준은 같은 노트북·seed42의 `MVA_native=876.899`다.

| 보정 구조 | 실제 구종 oracle | Trackman 행단독 확률 | 대회 47열 구종확률 |
|---|---:|---:|---:|
| 구종 3개 | **+120.693** | −13.043 | **−14.843** |
| 구종×볼카운트 36개 | **+141.909** | −25.091 | **−28.871** |

oracle은 전체의 약 76.6%에만 적용했는데도 매우 크다. 직구는 2023 기준선 잔차가
약 `+0.0211`, 변화구는 `−0.0289`, 오프스피드는 `−0.0055`였다. 즉 실제로 선택된
구종은 제구성공과 강하게 연결된다.

그러나 합법적인 기대 보정과 2024 실제 잔차의 상관은 `0.0000~0.0017`에 불과하다.
실제 구종 선택 중 현재 행으로 예측 가능한 부분은 현행 CatBoost가 이미 `x`에서 기대값으로
흡수했고, 남은 개별 선택 변동은 직전 구종·구종 시퀀스 없이 복원되지 않는다는 뜻이다.

## 멀티태스크 신경망 판단

표준 구조는 공유 encoder에 제구 BCE head와 구종 CE head를 두고, 구종이 복원되는 행만
보조손실을 마스킹하는 것이다.

```text
L = BCE(control_success) + λ · mask_pitch · CE(pitch_type)
```

이는 규칙상 가능하며 75%의 보조 라벨도 확보했다. 하지만 당장 GPU 실험으로 승격하지
않는 이유는 다음과 같다.

1. 구종확률을 직접 주변화한 가장 저분산 gate가 두 구조에서 모두 크게 음수다.
2. 보조 구종 head는 대회 입력과 같은 `x`만 보므로 새로운 추론 정보를 만들지 않는다.
3. 기존 MLP/TabM/two-tower는 CatBoost보다 190점 이상 뒤져, 표현 정규화만으로 그 격차를
   메울 근거가 없다.
4. 과거 E115의 time-honest `P(type|pitcher,count)` 피처도 6시드 `−0.9`, `t=−0.43`이었다.

따라서 full replacement NN은 기각한다. 나중에 재개한다면 챔피언 확률을 고정한 뒤 출력
크기를 강하게 제한한 residual head, 고정 `λ` 하나, 2022→23과 2023→24 양 전이 양수라는
조건으로만 검정한다. 직전 평가행·row_id 순서·test 배치 집계는 사용하지 않는다.

## 판정

- SBS 시퀀스 직접 재현: **제출 경로 기각** — 필요한 직전 투구 상태가 test에 없다.
- 실제 구종을 현재 행 피처로 사용: **금지** — 투구 후에만 확정되는 정보다.
- 예측 구종확률의 제구 주변화: **CLOSED** — 최신 전이 `−13.04 ~ −28.87`.
- masked multitask residual: 구현 가능하나 **미승격 연구 lane**. 새 행 단독 구종 신호가
  발견되기 전에는 현재 탐색 우선순위보다 낮다.
