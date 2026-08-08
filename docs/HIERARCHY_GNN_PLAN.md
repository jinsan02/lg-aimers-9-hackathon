# 동적 계층모형 / 관계 그래프 실험 설계

작성일: 2026-08-09  
목표: 기존 `asof_*`, 계층 TE, skill 피처와 겹치지 않는 **투수-타자 관계 구조**가
미학습 다음 시즌에 일반화되는지 검증한다.

## 1. 결론

첫 실험은 완전한 GNN이 아니라 **G0: 동결 과거 그래프의 구조 피처를 CatBoost에 추가**하는
것으로 한다. G0가 미학습 2024 표면에서 살아남을 때만 **G1: 귀납형 bipartite GraphSAGE**로
확장한다.

동적 계층 로짓모형은 비교 대조군으로만 둔다. 현재 피처를 둔 GPBoost 투수 랜덤효과의 분산이
약 0.0042로 붕괴했고, 투수 피처 42개를 제거해 랜덤효과에 역할을 넘기면 -265였다. 또한
투수×타자손 잔차의 2023→2024 전이는 -191.952였다. 따라서 같은 그룹 절편·랜덤 기울기를
다시 구현하는 것은 우선순위가 낮다.

## 2. 구조 감사

`tools/hierarchy_graph_audit.py`가 목표 시즌 S마다 `season < S`만 사용해 감사했다.

| 목표 시즌 | 과거 고유 edge | 밀도 | 투수 hit | 타자 hit | exact pair hit | 활성 투수 이웃 Jaccard 중앙값 |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | 69,307 | 15.94% | 86.2% | 90.6% | 54.1% | 0.375 |
| 2024 | 82,297 | 15.60% | 80.1% | 90.7% | 51.2% | 0.360 |

2024에서 투수 이웃 타자의 평균 성공률은 투수 자신의 과거 성공률과 상관 +0.611이지만,
표준편차도 0.0328 남는다. 관계 정보가 완전히 기존 투수 평균으로 환원되지는 않는다.

주의점은 두 가지다.

- 2024 행의 19.9%는 과거 미관측 투수다. ID 임베딩 lookup만 쓰는 GCN은 사용할 수 없다.
- F리그 exact-pair hit는 23.6%, R리그는 54.9%다. pair 통계 하나로 밀면 F가 무너질 수 있다.

## 3. 누수와 행 독립 경계

목표 시즌 S의 그래프, 노드 통계, edge 통계는 반드시 `season < S`로만 만든다.

- val 2023: 2019~2022 그래프
- test 2024: 2019~2023 그래프
- 최종 2025: 2019~2024 그래프

2025 테스트 행끼리 edge를 만들거나, 앞 행의 예측·상태로 뒤 행을 갱신하지 않는다. 제출물에는
2019~2024로 만든 동결 lookup과 동결 encoder만 담는다. 각 테스트 행은 자신의 ID·사전정보로
lookup/encoder를 한 번 호출하므로 배치 순서와 무관하다. 최종 스크립트는
`tools/audit_rowindep.py`를 통과해야 한다.

## 4. G0 — topology-only graph feature gate

### 4.1 그래프

- 이분 그래프 노드: `pitcher_id`, `batter_id`
- edge: 과거에 한 번 이상 맞붙은 투수-타자
- 최초 변경은 **구조와 이웃의 사전 요약만** 쓴다. exact-pair 성공률은 넣지 않는다.
  pair target 통계는 별도의 G0-E 실험으로 분리한다.

### 4.2 첫 피처 묶음

`fpipe.fit()`에서 시즌 확장 피처를 만들고 최종 과거 lookup을 artifact에 저장한다.

- `g_pitcher_degree`, `g_batter_degree`, `g_pair_n`, `g_pair_seen`
- 투수 이웃 타자들의 과거 `asof_batter_success/middle/n` 가중 평균·표준편차
- 타자 이웃 투수들의 과거 `asof_pitcher_success/middle/ball/reverse/strike/n` 가중 평균·표준편차
- 자기 통계와 이웃 통계의 차이, `log1p(degree)`, `log1p(pair_n)`
- hit flag: pitcher / batter / pair. 미관측 값은 시즌·리그 prior와 hit flag로 처리

edge 가중치는 첫 실험에서 `sqrt(n_edge)`로 고정한다. recency, 성공률, failure-mode 구성은
한 번에 섞지 않고 후속 단일 변경으로 분리한다.

### 4.3 구현 경계

- 새 모듈: `src/graph_features.py`
- 새 플래그: `--feat-graph-topology`
- `fpipe.fit(train, args, is_fit)`:
  - 각 학습 행은 자기 시즌보다 이전 시즌의 그래프만 보도록 expanding 생성
  - 제출용으로 fit 범위 전체의 최종 lookup 저장
- `fpipe.transform(df, art)`: 저장된 lookup만 행별 merge
- CatBoost의 나머지 설정과 피처는 AB 기준선 그대로 고정

### 4.4 판정

1. 정적 검사: 미래 시즌을 섞었을 때만 값이 달라지는지 cutoff 단위 테스트
2. 1시드 smoke: val 2023→test 2024, `--drop-f-pre 2022`, A100의 같은 설정 기준선과 비교
3. target가 기준선보다 낮거나 예측 RMS가 사실상 0이면 즉시 종료
4. 살아남으면 같은 머신 6시드 paired 비교
5. 채택은 기존 규칙대로 paired t ≥ 2.4. 95% 상한 < +3이면 축을 닫는다
6. 독립 멤버 가치도 보되, 고정 소가중 blend가 현재 AB+DW 대비 +3 이상이어야 G1로 간다

## 5. G1 — inductive bipartite GraphSAGE

G0 통과 후에만 구현한다. 평가 환경에 PyTorch는 있지만 PyG는 없으므로 순수 PyTorch sparse
집계로 만든다.

### 5.1 입력

- 투수 초기 상태: 과거 성공/실패모드, recent 1/3/5, count profile, 손, `log1p(n)`
- 타자 초기 상태: 과거 성공/middle, 손, `log1p(n)`
- edge 상태: `log1p(n_edge)`, 마지막 대결 이후 경과, 대결 count 분포
- ID 자체는 입력하지 않는다. ID는 과거 그래프의 연결과 lookup key로만 쓴다.

### 5.2 encoder

- 1~2층 weighted-mean GraphSAGE
- hidden 16 또는 32, residual update, dropout 0.2, weight decay
- 미관측 노드도 행의 공식 `asof_*` 피처를 node MLP에 넣어 임베딩 생성
- 알려진 노드는 과거 이웃 message와 node MLP를 결합

### 5.3 pitch decoder

행별로 `[z_pitcher, z_batter, z_pitcher*z_batter, |z_pitcher-z_batter|, row_context]`를
작은 MLP에 넣는다. `row_context`는 투구 전 확인 가능한 count·inning·score·hand·league 및
기존 as-of 피처다.

GNN 단독 예측과 AB 잔차(offset) 예측을 각각 보되, 첫 정식 후보는 다음 offset 형태다.

```text
logit(p_graph) = logit(p_AB_OOF) + delta_graph(row)
```

학습의 `p_AB_OOF`는 반드시 rolling-origin/OOF 예측이어야 한다. in-sample AB 예측을 offset으로
쓰면 잔차가 인위적으로 작아진다.

### 5.4 학습과 제출

- val 2023 encoder: ≤2022 graph로 학습/동결, 2023 행 평가
- test 2024 encoder: ≤2023 graph로 refit/동결, 2024 행 평가
- 최종 encoder: ≤2024 graph로 refit, 2025 행별 독립 추론
- seed 42 한 개로 구조·속도·standalone/상관을 본 뒤, G0를 통과한 경우에만 6시드
- 저장물은 state dict, scaler, node lookup. `failmode.py`나 원시 train은 제출 zip에 넣지 않는다

## 6. 동적 계층모형 대조군 H1

기존 GPBoost를 반복하지 않고 AB의 OOF logit을 offset으로 둔 최소 상태공간 모형만 비교한다.

```text
logit(p_i) = logit(p_AB_OOF,i) + u[pitcher_i, season_i] + v[batter_i, season_i]
u[p,s] = rho_p * u[p,s-1] + epsilon[p,s]
v[b,s] = rho_b * v[b,s-1] + epsilon[b,s]
```

- 처음에는 투수·타자 random intercept만 사용
- count/hand random slope는 기존 연도 전이 실패 때문에 제외
- 사후 평균은 표본수로 수축하며 미관측 개체는 0 또는 리그 prior
- H1 단독이 아니라 G0/G1의 graph prior가 `u[p,s]`의 prior mean을 설명하는 H2가 최종적인
  계층-그래프 결합 형태다

H1은 구현비가 작을 때만 실행한다. GPBoost 분산 붕괴를 뒤집을 새 근거가 없으면 G0보다 먼저
GPU를 배정하지 않는다.

## 7. 중단 조건과 우선순위

1. **G0 topology-only**: 가장 싸고 누수 감사가 쉽다.
2. G0가 통과하면 **G1 GraphSAGE**, 실패하면 그래프 축 종료.
3. G0가 약하지만 F/R 반대 부호가 아니면 **G0-E edge recency** 한 번만 분리 검정.
4. H1은 비교 대조군. 기존 랜덤효과 실패를 단순 반복하지 않는다.
5. TGN은 2025 행 사이 상태 갱신이 행 독립을 깨므로 현 단계에서 사용하지 않는다. 2024에서
   완전히 동결한 temporal encoder만 합법이지만, GraphSAGE보다 구현비가 커 후순위다.

## 8. 이 설계가 기존 축과 다른 이유

기존 TE/as-of는 개체와 상황의 과거 평균을 직접 요약한다. G0/G1은 **그 선수가 누구와 상대해
왔는지, 상대군의 구성과 연결 패턴이 어땠는지**를 요약한다. 즉 “투수 평균 성공률”이 같은 두
투수라도 상대 타자군과 연결 구조가 다르면 다른 표현을 만든다. 구조 감사의 이웃률 상관이
1이 아니고 활성 투수 Jaccard가 0.36인 것이 이 축을 한 번 검정할 근거다.
