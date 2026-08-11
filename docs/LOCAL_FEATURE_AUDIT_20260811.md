# 로컬 피처 엔지니어링 감사 — 2026-08-11

## 결론

노트북 환경을 평가 서버의 핵심 패키지 버전에 맞춘 뒤, 동일 로컬 RTX 5060·동일
seed42·동일 `val2023→test2024` 표면에서 피처 삭제 4개와 피처 추가 5개를 검증했다.
기준 `MVA_native=876.90`을 넘은 후보는 없었고, 따라서 다중시드·제출 패키지로
승격한 후보도 없다. 현행 LB 챔피언 v11 `1101.802`를 유지한다.

## 환경

- Python 3.11.15
- pandas 2.0.3, numpy 1.26.4, scikit-learn 1.8.0, joblib 1.5.3
- CatBoost 1.2.10
- RTX 5060 Laptop GPU 8GB

Python과 평가 기본 패키지는 대회 서버와 일치한다. CatBoost는 제출 zip에 포함해야 하는
추가 패키지이므로 기존 제출 구조를 그대로 따른다.

## 실제 GPU 결과

공통 설정은 현행 121피처 이진 모델이며 `--drop-f-pre 2022`, `--lr .01`, depth 8,
std-k 80, season-prior, domain, skill-pc를 고정했다.

| 태그 | 단일 변경 | 2024 BSS | 기준 대비 | 판정 |
|---|---|---:|---:|---|
| MVA_native | 기준 | 876.90 | 0.00 | 기준 |
| LPR_PD1 | 구종배합 delta 3열 제거 | 871.91 | -4.99 | 기각 |
| LPR_OD1 | 성공/실패 outcome delta 7열 제거 | 860.68 | -16.22 | 기각 |
| LPR_LOW | 중요도 0 부근 indicator 5열 제거 | 865.02 | -11.88 | 기각 |
| RPB_R3Q8 | reverse×prev3 success 동결 q8 범주 | 841.27 | -35.63 | 기각 |
| RPTE_R3Q8K500 | 위 범주의 과거시즌 전용 TE | 869.06 | -7.84 | 기각 |
| PBTE1 | exact pitcher×batter를 모델 입력 TE로 이동 | 859.34 | -17.56 | 기각 |
| LFA_BT1 | batter_team_id 제거 | 831.69 | -45.21 | 기각 |
| FCC1 | 볼-스트라이크 정확한 12상태 범주 | 867.33 | -9.57 | 기각 |
| QMIN1 | std 투수/타자 성공률의 최소값 | 865.95 | -10.95 | 기각 |

`LFA_BT1` 과정에서 `--drop-cols`가 범주형 열을 features에서는 빼고 CatBoost의
`cat_features` 목록에는 남기는 기존 버그를 발견해 두 목록을 동기화했다. 실패한 첫
호출은 학습 전에 종료돼 LEDGER 결과로 기록되지 않았다.

## CPU 사전 게이트

### LossFunctionChange

저장 모델 예측을 최대 오차 0으로 재현한 뒤 2023/2024 두 표면의
`LossFunctionChange`를 계산했다. 공통 음수는 20개였고 `batter_team_id`가 가장 강했지만,
실제 삭제 결과는 -45.21이었다. 이 값은 Logloss 기반의 고정 모델 귀속값이지 재학습 후
Brier 인과효과가 아니므로 이후 삭제 후보 선별에도 사용하지 않는다.

### 야구 proxy

- 과거 시즌의 투수/타자 2스트라이크·깊은 카운트·풀카운트 비율: 두 전이 최선이
  `batter_full_current_gate +0.089/+0.565`, +3 승격선 미달.
- 구종배합/실패모드 entropy와 최대 구성비: 두 전이 전부 음수.
- 홈팀·원정팀·홈×원정: 2023→2024가 약 -29~-175로 불안정.
- 품질 bottleneck `min(std_pitcher_success, std_batter_success)` 잔차 lookup은
  `+9.753/+11.838`이었지만 실제 모델 피처는 -10.95. 사후 잔차 lookup의 이득을
  CatBoost 증분으로 해석할 수 없음을 확인했다.

## 구현과 재현 산출물

- `tools/loss_feature_audit.py`: 모델 예측 비트 재현 후 피처별 LossFunctionChange CSV
- `tools/pa_depth_proxy_audit.py`: prior-season 타석 깊이 proxy의 두 전이 감사
- `src/fpipe.py`: 기본 비활성인 count-category, quality-min, recent-pair-bin 경로
- `src/target_enc.py`: 기본 비활성 recent-pair-bin TE 키
- `src/train_gbdt2.py`: 위 플래그와 범주형 drop 동기화
- `out/full_audit/loss_feature_MVA_2024.csv`
- `out/full_audit/loss_feature_MVB22_2023.csv`
- `out/full_audit/pa_depth_proxy_transfer.csv`
- `out/full_audit/model_interactions_MVA.csv`

## 다음 방향

동일 원재료의 결정론적 곱·범주·삭제는 현행 CatBoost의 분할 경로만 흔들고 resolution을
늘리지 못했다. 다음 실험은 새로 발견된 정보나 독립적인 출력 구조가 있을 때만 연다.
오늘 닫힌 플래그는 재시도하지 않고, v11 recent-middle + exact-PB를 보존한다.
