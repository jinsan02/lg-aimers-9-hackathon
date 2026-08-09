# 2026-08-09 인간 야구 관점 돌파구 브레인스토밍

## 한 줄 결론

현 데이터에서 더 넣을 수 있는 일반 야구 상식은 거의 소진됐다. 다음 돌파구는
**최근 제구 상태**와 실제 LB에서 재현된 **투수×타자 상성**을 더 안전하게 구조화하는
것이며, 이를 고르기 전에 rolling-origin 검증 표면을 늘려야 한다.

## 타깃을 야구 언어로 다시 정의

이 문제는 스트라이크 확률이 아니라 포수가 요구한 코스를 실행할 확률이다. 존 밖 유인구도
요구대로 던지면 성공일 수 있고, 스트라이크라도 한가운데나 반대 방향이면 실패다. 따라서
제구력 하나뿐 아니라 다음 두 잠재요인이 섞여 있다.

1. 상황·카운트·타자가 정하는 **투구 의도**
2. 최근 투수 상태가 정하는 **실행 능력**

현재 CatBoost는 count, runner, hand, asof/TE로 이 둘을 상당 부분 흡수한다. 단순 교차항,
hard routing, 역할 분할을 반복하면 기존 실패를 재현한다.

## 실제 증거

- recent-5-game middle 보정: LB `+6.237`
- exact pitcher×batter 잔차: 과거 전이 `+1.704`, LB `+1.713`
- career-middle 교체: LB `−18.271`
- topology graph: A100 `−1.93`, 4070 `−9.16`
- 동적 계층 잔차: `+8.03 → −19.09`
- lineup-history 신규 감사: 최고 전체 `+0.444`, 전반 `+8.911`, 후반 `−10.591`
- mechanics-history 신규 감사: 최고 전체 `−0.245`, 전반 `+18.801`, 후반 `−25.268`

즉 고정 개인실력·연결구조·과거 타순·과거 투구폼은 시즌을 안정적으로 넘지 못했다. 반면
최근 middle과 exact matchup만 실제 평가에서 재현됐다.

## P0 — 모델보다 먼저 검증 표면 확장

후보를 `v11 recent-middle+PB` 오프라인 아날로그 위 증분으로 비교한다.

- rolling-origin `2021→22`, `2022→23`, `2023→24`
- 각 전이의 후보 경로 자체가 현행 대비 절대 양수
- 마지막 전이 전반기·후반기 모두 0 이상
- 동일 머신 6시드 paired `t ≥ 2.4`, 평균 증분 `+3` 이상
- known pair / known-new-pair / cold player, R/F, 경험, 월별 MSE 보고
- reliability가 아니라 resolution을 올렸는지 Murphy 분해

실패한 두 경로의 상대 비교, 한 전이 raw sweep 최고값, LB 기반 부분가중은 금지한다.

## P1 — outcome-aware PB residual matrix factorization

가장 강한 신규 후보. topology-only GNN이 아니라 **결과가 붙은 matchup edge**를 쓴다.

```text
r = y - p_v11_analogue
delta(p, b) = exact_pair_bias[p,b] + u[p] · v[b]
prediction = clip(p_v11_analogue + delta)
```

- `K=0` arm이 현 exact-PB를 재현하지 못하면 즉시 중단
- 사전 고정한 `K=4` 하나만 검정, 강한 L2/count shrinkage
- 투수·타자 단독 bias는 동적 hierarchy 실패와 겹치므로 제외
- 미관측 선수=0, known player/new pair는 저랭크 항만 적용
- 2025에는 2019~2024에서 동결한 embedding/lookup만 사용

인간적 의미는 반복 대결에서 생기는 타자별 존 압박·사인·익숙함을, 표본이 적은 pair에는
비슷한 상대 패턴으로 부분 수축하는 것이다.

## P2 — recent-state empirical Bayes

career를 다시 쓰지 않고 실제 성공한 recent-middle을 구조화한다.

- prev1/3/5 middle의 중첩창을 1경기 / 2~3경기 / 4~5경기로 분해
- 과거 경기당 투구수 분포로 각 창의 관측 불확실성을 근사
- latent recent level, 변화속도, uncertainty만 생성
- 첫 arm은 middle만 사용; success/reverse는 후속 단일변경
- 비단조 q-bin lookup 대신 선형/단조 spline과 강한 수축

단순 form delta `−5.4`, window `+3.42 (t=1.42)`와 달리 표본 신뢰도를 명시한다.

## P3 — workload pace 저비용 감사

행에 이미 있는 당해 시즌 투구수 `std_pitcher_n`을 월 진행도로 나누어 시즌 사용 강도를
만든다. 같은 6월이라도 정상 로테이션, 과사용, 늦은 콜업·복귀를 구분하려는 피처다.

- `log1p(current-season n) - historical expected log n(game_month, league, prior role)`
- prior-season workload 대비 current pace ratio
- current inning과 prior-season role compatibility

현재행 공식 asof와 train anchor만 쓰므로 행 독립이다. 단, month가 거친 proxy이고 옛 role
모델이 `−5`였으므로 GPU 전에 CPU residual-transfer gate만 한다.

## P4 — soft intent × execution low-rank model

3볼 must-strike, 2스트라이크 chase, 중립 승부라는 야구 의도를 hard 모델로 쪼개지 않는다.

- context encoder: count, runner, batter threat, hand, LI → 3~4 intent weights
- pitcher-state encoder: recent middle/ball/reverse 실행 상태
- low-rank bilinear 결합으로 14 failure cell을 예측
- 성공확률은 성공 cell 합으로 유지

기존 count routing·cell×count가 크게 실패했으므로 모든 데이터를 공유하는 작은 residual head만
허용한다. 비용과 위험이 높아 P1/P2 뒤다.

## P5 — 팀 사인 성향 / PB 제한 확장

- 수비팀×count family×주자상황의 failure-mode composition이 여러 연도에서 안정적인지
  Dirichlet shrinkage audit. 연도별 상관 중앙값이 `.25` 미만이면 즉시 종료.
- 현 PB의 pair_n, 최근 시즌 pair_n, 마지막 대결 gap으로 수축강도만 조절. recency와
  adaptive-k를 동시에 바꾸지 않는다.

팀 성공률 상수, batter×opponent, graph topology는 이미 실패했으므로 재사용하지 않는다.

## 2025 제도 변화의 해석 경계

KBO 공식 2025 운영안에는 ABS 존 상·하단 조정과 피치클락 정식 도입이 있다. 이는
career-middle보다 recent-middle이 더 유효할 수 있다는 **가설의 배경**은 되지만, 2025 실제
통계나 수동 SHIFT를 모델에 넣는 근거는 아니다. 외부데이터·평가시즌 사후 적합은 금지한다.

## 하지 않을 것

- catcher ID, 구장, 날씨, 키, 실제 구종·구속·릴리스, 당일 투구수, 정확한 휴식일
- test 행 순서로 현재 경기 workload·상대순번 복원
- lineup/mechanics/role/graph/hierarchy를 이름만 바꿔 재실행
- 2025 공개 야구 통계를 이용한 calibration 또는 SHIFT

## 권장 기기 순서

1. A100: rolling-origin 표면 보강 → PB matrix factorization K=0/K=4
2. 4070: recent-state와 workload-pace CPU audit → 양 표면 단일시드 gate
3. 둘 중 통과한 축만 6/8시드 확장
4. intent/failure-mode 구조는 앞의 두 축이 실패하거나 통과한 뒤 진행

