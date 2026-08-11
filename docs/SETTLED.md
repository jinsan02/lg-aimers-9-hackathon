# 닫힌 질문 — 다시 하지 말 것

> `tools/precheck.py` 가 이 파일을 읽는다. 실험 전 반드시 통과시킬 것.
> 형식: `FLAG <플래그조각> | <판정> | <수치> | <근거>`
> 판정은 `BANNED`(재실행 금지) / `CLOSED`(닫힘, 새 근거 있을 때만) 둘 중 하나.

## 왜 이 파일이 생겼나

2026-08-07 하루에 **약 30개 실험을 돌리고 docs/EXPERIMENTS_LOG.md 에 하나도 안 적었다.**
그 결과 `season` 제거를 "한 번도 안 해본 축"이라고 두 번 말하고 실제로 큐에 걸었다.
E08 에서 **−580점**으로 이미 끝난 질문이었다. 기록이 멈추면 대화 기억으로 일하게
되고, 대화는 압축된다. 그래서 기록을 **자동화**하고(LEDGER.tsv) 조회를
**강제**한다(precheck).

---

## BANNED — 기전까지 밝혀진 것. 표면이 바뀌어도 안 뒤집힌다

FLAG --drop-cols season | BANNED | E08 −580 | season 은 드리프트 캘리브레이터다. 시즌당 −0.012 씩 떨어지는 수준을 이 피처가 잡는다. 빼면 6년 평균(0.524)을 예측하는데 2025 는 ~0.48 이라 편향 0.04 짜리 재앙이 된다.
FLAG --drop-cols season,season_progress | BANNED | E08 −580 | 위와 동일.
FLAG last-season-weight | BANNED | LB −41.5 | 최신시즌 가중(W). 로컬은 좋아지는데 LB 오프셋이 +133 → +99 로 무너진다. F리그 유사물이 부호를 반대로 예측했다.
FLAG test-row-derived | BANNED | 실격 사유 | test.csv 내부 다른 행을 쓰는 피처·사후보정 전부. 실패모드 라벨 복원(failmode.py)을 test 에 쓰면 2025 타깃이 96.79% 복원된다 — train 전용.
FLAG lb-probing | BANNED | 규칙·심사 | LB 점수를 보고 상수를 되맞추는 것. Public=Private 이라 정답 세트 직접 적합이고, Phase 2 코드 심사에서 근거 없는 상수로 남는다.

## CLOSED — 측정으로 닫힘. 새 근거(표면·구조 변화) 있을 때만 재개

FLAG --model lgb | CLOSED | 미학습표면 margin −7.1 | 자기검증(−9.8)과 미학습(−7.1) 양쪽에서 음수. D 가 9.8 → 59.9 로 **더 나빠졌다** — 표면 효과가 이쪽엔 없다.
FLAG --feat-frac | CLOSED | 미학습표면 이득 +0.60 | 부분공간 0.7. margin +8.8 로 양수지만 무시할 크기.
FLAG gpboost | CLOSED | 랜덤효과 분산 ~0 | 우리 피처(asof·TE·skill)를 다 넣으면 투수 랜덤효과 분산이 0.0042 로 죽는다. 피처 42개를 빼고 랜덤효과로 대체하면 −265.
FLAG --tm-feats | CLOSED | 홀드아웃 6시드 −6.04, t=−2.14 | trackman 투수×시즌 요약 13개. 링키지는 좋다(행 커버 99.6%, ratio>0.9 가 70.8%) — 정보 자체가 안 먹힌다.
FLAG optuna | CLOSED | 홀드아웃 6시드 −7.44, t=−2.65 | 공동 하이퍼 탐색. 2시드(SE 3.2)로 40 trial 중 최고를 고르면 선택 편의가 이득보다 크다.
FLAG segment-drift | CLOSED | 이전 −3.76 ~ +0.96 | 세그먼트별 드리프트 보정. 자기적합은 +11.45 인데 시즌을 못 넘는다. 매핑 성공률 100% 확인함(audit_seg).
FLAG --label-smooth | CLOSED | E118 계열 | 라벨 평활. 손실·링크를 같이 바꿔 가설이 섞인다.
FLAG --resid-col | CLOSED | −17.8 | 2단 잔차. 타깃·손실·링크 6가지를 한 번에 건드렸다. baseline-col 이 같은 가설의 단독 검정본이고 그것도 중립.

## 조건부 — 표면을 바꾸면 뒤집힌 전례가 있다

FLAG --failmode-cells | OPEN | 자기검증 D +2.8 → 미학습 D **−18.0** | 셀 다중분류. **자기검증 표면에서 과소평가되고 있었다.** base 보다 강한 모델이고 미학습 표면 최적 w 0.70 (이득 +22.26). 셀 기하 변형(d4/d6/16셀/다중라벨)은 전부 틀린 표면에서 기각됐으므로 재검정 대상.

## 측정 원칙 (어기면 하루를 날린다)

1. **판정 표면**: `--val-season S-1 --test-season S` (배치 구조). 자기검증 2024 는 참고용.
2. **후처리 상수**는 그 값을 잰 실행의 **학습 데이터·플래그가 제출과 완전히 같을 때만** 쓴다. (v16 −6.15 — 편향은 `--drop-f-pre 2022` 로 재고 제출엔 안 썼다)
3. **구조가 다른 실험의 결론을 옮기지 말 것.** (v17 −53.6 — FL 실험은 학습에 신체제 F 가 0시즌, 제출은 2시즌)
4. **제출 하나에 변경 하나.** (v16 은 SHIFT+SLOPE 동시 변경 → 역산 필요)
5. 채택 t ≥ 2.4 / 기각 95% 상한 < +3, 홀드아웃 시드 n ≥ 6, 페어 비교.
6. **시드를 늘려 SE 를 줄이는 것으로는 보류 축이 채택으로 안 넘어온다.** 6→12→18
   로 가도 결론이 안 바뀌었다. 머신 효과(11점)가 시드 잡음보다 크므로 시드 확장은
   그 편의를 못 줄인다 — n≥6 만 채우고 다음 축으로 갈 것.

## 비교 규칙 (2026-08-08 추가, 실측 근거 있음)

FLAG cross-machine-compare | BANNED | A100 +15.05 vs 4070 +4.27 | **비교군은 같은 머신에서 만들 것.** std-k 40 을 같은 설정·같은 6시드로 두 머신에서 돌렸더니 11점이 벌어졌다. 기준선 RN1.5 가 4070 산인 줄 모르고 A100 결과와 비교해 "채택 t=11.88" 을 만들었고, 하마터면 그 설정으로 제출할 뻔했다. 시드를 늘려도 못 막는다 — 머신 효과는 시드 잡음이 아니다. `tools/surf_report.py` 가 LEDGER 의 hostname 을 대조해 자동으로 '무효' 처리한다.

## 2026-08-08 추가

FLAG --failmode-cells-second | CLOSED | 두 번째 셀 멤버 +0.34 | **셀 축 소진.** base+d5 가 +19.22 인데 d6 를 더해도 +19.56 이다. 셀끼리 rms 0.0032 (base 와는 0.0107) — 같은 라벨·피처·알고리즘이라 서로 닮았다. d4·c16 은 NNLS 가중 0.00. 셀 최적 가중은 0.55~0.65 이고 그 구간이 평평하다.
FLAG --league F | CLOSED | 학습 55,696행에서 크래시 | F 전용 모델. `--league F --drop-f-pre 2022` 면 신체제 F 가 2시즌뿐이라 55,696행만 남고 TE 결측 71.5%, skill 추정기가 적합할 시즌이 없어 matmul 오류. **데이터가 없어서 못 한다.** 그리고 F 는 BSS 가 낮아 보일 뿐 **MSE 는 R 보다 낮다**(.24690 vs .24770) — 기저율이 0.5 에서 멀어 생긴 착시다.
FLAG --std-k 40 | CLOSED | 제출 표면 -1.954, SE 2.377, t=-.822 | 판정 표면에서는 4070 +4.27(t=1.30) / A100 +10.31(t=3.55)이었지만, 실제 제출 학습집합(val2024, drop 없음, 4070 8시드, 지문 0.5401750413)에서는 `SK2_k40−VB2_base`가 음수였다. 현행 base+cell의 base 완전교체 -0.564, 절반교체 +0.194이고 절반교체도 전반기 +0.417/후반기 -0.305로 전이되지 않는다. **작은 k의 과거 표면 이득은 제출 구조에서 재현되지 않는다.**
FLAG isotonic | CLOSED | E30 −21 | 확률 보정 전반. 2019~22 학습 → 2023 보정 → 2024 테스트 백테스트에서 **모든 보정이 악화**했다(전역δ −21, isotonic −21, 월별 −52). 그리고 미학습 2024 신뢰도 분해상 **고칠 수 있는 항이 총 +13.1** 뿐이고 SLOPE 가 이미 +3 을 먹었다. 후처리 여지는 사실상 없다.
FLAG --refit-mult 2.0 | CLOSED | 4070 6시드 +0.73, t=+0.43 | 재학습 배수 1.5→2.0. 자기검증 12시드에서 +2.44(t=1.91) 로 보류였는데 **미학습 표면(RS1.5 vs RS2.0, 시드 31~36)에서 +0.73, t=0.43** 으로 사라진다. 95% 상한 +4.1 < 기각선 밖이 아니라 애매하지만 학습시간이 1.33배다.
FLAG --te-k b:500 | CLOSED | TK_b500 +0.58(t=0.19) / TK_rel +2.26(t=0.80) | 축별 신뢰도 수축. E161 적률법은 타자 축이 6~11배 덜 수축돼 있다고 했지만 효과 없음. **최적 독립 추정량과 최적 피처는 다르다** — 트리가 원본 asof 를 같이 보고 있어 수축을 맞춰도 얻을 게 없다.
FLAG --fm-context | CLOSED | FX_cnt −71.8 (A100 6시드) | 셀 × 볼카운트 맥락. 셀을 카운트별로 쪼개면 셀당 표본이 줄어 다중분류가 무너진다. 801.2 vs AB_base 873.0.
FLAG --row-filter strikes | CLOSED | FX_str −2.76, t=−1.51 (A100 6시드) | 2스트라이크 라우팅(전용 모델). 손실 지도상 0-2/1-2 카운트가 손실의 16% 라 노려봤는데, 세그먼트를 떼면 나머지에서 배울 걸 못 배운다.
FLAG --depth 9 | CLOSED | −3.38, t=−1.68 (4070) | depth 8 이 최적. depth 7(SC_d7)·9 양쪽 음수.
FLAG --lr 0.02 | CLOSED | −10.66, t=−2.68 (4070) | lr 0.01 유지. 올리면 확실히 나빠진다.
FLAG --std-k 120 | CLOSED | −4.61, t=−1.40 / k200 −19.98 | 시즌표준화 k 를 올리는 방향은 닫혔다. 내리는 방향(k40)만 OPEN.
FLAG drop-f-pre-생략 | BANNED | val2023 표면에서 best_iter 12~21, BSS 124~232 | **판정 표면(`--val-season 2023`)은 `--drop-f-pre 2022` 없이 돌릴 수 없다.** 구체제 F(성공률 0.71) 를 학습에 넣으면 신체제 2023 검증 손실이 15 iter 에서 최저를 찍고 계속 나빠진다(AB2_base 실측: best_iter 12~21, test BSS 124~232 vs AB_base 868~881). 반면 **제출 표면(`--val-season 2024`)은 이 플래그를 쓰지 않는다** — 제출 멤버 v14f·ZD5 가 그렇다. 두 표면이 서로 다른 플래그를 요구하므로 **명령을 표면 간에 복사하지 말 것.** 이걸 세 번 틀렸다(v16·v17·08-08 VB_base).
FLAG teacher-fpipe-order | CLOSED | T3 vs T2 −1.64, SE 0.86, t=−1.91 | 교사 `fpipe.fit` 이 2019~2024 를 보는 문제. 순서를 `load → season 필터 → fpipe.fit` 로 고쳐 재측정하니 이득이 +39.85 → +38.43 로 **1.6 만 줄었다**(사전 조건 |Δ|<2×SE 충족). `target_enc.build_te` 가 시즌 expanding + shift(1) 이라 행 단위로는 안 새고, 전역 prior 스칼라 하나만 흘렀다. **교사는 이제 필터-먼저 순서를 쓴다.**
FLAG 멤버-학습집합-대조 | BANNED | v16 −6.15 / v17 −53.6 / 08-08 VB_base 52점 | 블렌드 가중을 **학습 데이터가 다른 멤버들** 사이에서 고르는 것. 판정용 명령의 `--drop-f-pre 2022` 를 제출 멤버 재현에 복사해 52점 약한 모델을 v14f 재현본으로 착각했다. 눈으로는 구분 안 된다. **`python tools/member_fingerprint.py <태그들>` 로 pkl 의 `fpipe['priors']` 지문을 대조할 것** (다르면 종료코드 2). 제출 멤버 v14f·ZD5 는 `--drop-f-pre 2022` 를 **쓰지 않는다**(지문 0.5401750413).
FLAG --soft-target | CLOSED | 제출 표면 −10.19, 블렌드 가중 0.00 | **증류 종료.** 판정 표면의 `+38.43` 은 대부분 **교사만 본 F 2022 이전 행**이었다 — 교사에게도 `--drop-f-pre 2022` 를 걸면(`DT5_seq`) `+4.6~7.0` 으로 무너진다(85% 소실). 제출 표면(양쪽 다 drop 없음, val2024 8시드, 지문 일치)에서는 학생이 base 보다 **−10.19**(편향제거) 고, 3멤버 격자에서 최적 dist 가중이 **0.00** 이다. 다양성도 없다 — base 와 rms 0.0067 로 셋 중 가장 닮았다(base-cell 0.0110). 부드러운 타깃은 base 의 매끄러운 복사본이지 새 기하가 아니다.
FLAG prev-pitch-teacher | CLOSED | 위 `--soft-target` 에 흡수 | 직전 투구 결과는 평가 시점에 못 쓰므로 증류 교사로만 쓸 수 있었는데, 증류 자체가 닫혔다. 조건부 기여 +39.1 은 교사 안에서만 존재한다.
FLAG prev-pitch-teacher-old | CLOSED | 교사 A/B +39.1 | 직전 투구 결과의 **조건부** 기여. 단변량으로는 +127 로 보이지만 121개 피처를 다 넣으면 +39 다(T_self 2076.0 → T_seq 2115.1, 같은 랜덤 4-fold). 평가 시점엔 못 쓰므로 증류 교사로만 쓴다.

## 2026-08-08 Codex 단독 검문 추가

FLAG logit-blend | CLOSED | centered +0.021 | VB2_base 8시드와 ZD5 6시드, w=.55에서 probability 평균 961.628 vs logit 평균 961.649. 예측 범위가 좁아 실질 차이가 없다.
FLAG batter-experience-offset | CLOSED | 2023→2024 -47.618 | 2023 경험구간 편향을 고정해 2024에 적용하면 R -51.097, F +3.416. 상관이 높아도 크기가 전이되지 않아 전체 손실이다.
FLAG pitcher-batter-hand-residual | CLOSED | BI2023→BI2024 k100 -191.952 | 투수×타자손 공통그룹 상관 +.0108, 2024 행 커버 75.6%. 동일시즌 교차적합 +31은 연도 전이 신호가 아니다.
FLAG extreme-subgroup-correction | CLOSED | 2024 극단성 소실 | 과거 성공률 .679 수준의 후보도 2024 실제 .416~.441로 되돌아왔다. 고정 가능한 고신뢰 하위집단 없음.
FLAG --loss RMSE | CLOSED | E55 weight 0 / E93 +2.44, t=.80 | Brier 직접 최적화 안건은 이미 측정됐고 불확실성 대비 이득이 없다.
FLAG --feat-prof | CLOSED | E100 계열 -7.6 | 시즌 lag 프로필 신호가 현재 expanding/asof 피처에 중복된다.
FLAG --feat-form | CLOSED | E108 -5.4 | 단기 폼 변동이 일반화되지 않는다.
FLAG --feat-cross | CLOSED | lever H -1.7 | 추가 교차항이 기존 TE/skill과 중복된다.
FLAG --feat-count | CLOSED | E113 -1.07 | count 확장이 기존 count/TE 신호를 개선하지 못한다.
FLAG --std-excess | CLOSED | E104 -24.7 | 표준화 초과량 변환이 정보를 훼손한다.
FLAG --std-ratio | CLOSED | E110 -38.5 | 비율 변환이 불안정하다.
FLAG --std-multi-k | CLOSED | paired 약 0 | 복수 shrinkage 스케일을 함께 넣어도 추가 신호가 없다.
FLAG --std-k-mix | CLOSED | E112 계열 | 혼합 shrinkage가 현재 k80을 개선하지 못한다.
FLAG --std-k-bat | CLOSED | E112 계열 | 타자 전용 shrinkage 변경이 개선되지 않는다.
FLAG --monotone | CLOSED | E65 -28 | 단조 제약이 필요한 상호작용을 막는다.
FLAG --keep-ids | CLOSED | E88 -105 | 원시 ID가 과적합을 유발한다.
FLAG --drop-unstable | BANNED | E47 -262 및 IndexError | 성능 손실뿐 아니라 season_std 요구 열 제거로 코드 경로도 깨진다.
FLAG --fill-prev | CLOSED | -3.51 | 이전값 결측 대체가 개선되지 않는다.
FLAG --feat-v5 | CLOSED | -26 | 확장 피처 묶음이 일반화되지 않는다.
FLAG --te-halflife 2 | CLOSED | 제출 표면 +0.164, SE 1.760, t=.093 | A100의 drop-f-pre 판정 표면에서는 +6.03(t=3.32)이었으나 실제 제출 학습집합과 같은 4070 val2024 8시드에서는 사라졌다. 현행 base+cell 블렌드 대체 이득도 +0.610이고, 전반기 선택 가중은 후반기 -1.573 / 반대는 -1.112로 불안정하다. 학습집합 지문은 0.5401750413으로 일치했으므로 무효 실행이 아니라 **표면 전이 실패**다.
FLAG tabm-current-121 | CLOSED | 단독 821.81, AB 대비 margin −2.5 | 구 피처 TabM을 현재 121/std 피처와 올바른 val2023→test2024 refit 구조로 다시 만들었다. AB와 rms가 .0121뿐이고 성능 격차 61.6이 다양성 상한 59.1보다 커 최적 가중 0. **강한 TE를 같이 먹이면 NN도 Cat 기하를 재구성한다.**
FLAG tabm-cell-consistent | CLOSED | 6시드 단독 839.34, 현행 블렌드 가중 0.00 | TabM 32-head에 14 실패셀 softmax와 성공 marginal BCE를 공동 적용했다. 파일럿 1시드는 AB 대비 +16.58로 보였지만 6시드 평균은 +1.48, `AB+DW_cell` 동시 NNLS에서 가중 0. 전반→후반도 0, 반대 방향만 .122라 선택 편의였다.
FLAG mtnn-cell-consistent | CLOSED | 현행 블렌드 증분 +1.36 | 일반 MLP의 14셀 softmax+marginal BCE. 단독 701.99, rms .0256으로 다양하지만 격차가 너무 크다. `AB+DW_cell` 위 자기적합 증분 +1.36이고 후반→전반 가중 0이라 제출 비용·전이 위험을 못 넘는다.
FLAG cell-posterior-stack | CLOSED | 2023R source 최소 +101.32 → 2024R route −52.38 | 14셀 전체 확률을 성공합 스칼라 대신 ridge stack에 넣었다. 같은 시즌 전·후반에서는 거대한 이득처럼 보였지만 source-only arm/alpha를 미학습 다음 시즌에 적용하면 반전했다(전체 적용 −169.80). **실패형태 posterior의 클래스별 calibration과 성공 잔차 관계가 시즌을 넘지 않는다.**
FLAG mlp-plr | CLOSED | 미학습 최고 641.29 / sigma1 517.60 / sigma10 426.58 | TE·skill을 제외한 92개 수치+9개 범주 피처에 periodic-linear-ReLU 임베딩을 적용했다. 주파수 스케일을 0.1→1→10으로 올릴수록 악화했고 CatBoost 868.83과 격차가 너무 커 블렌드 여지가 없다.
FLAG content-two-tower | CLOSED | 미학습 675.32 / 제출 표면 699.67 | raw ID 없이 투수·타자 asof 콘텐츠 tower와 저랭크 곱/거리 상호작용을 학습해 콜드스타트는 해결했지만 두 표면 모두 CatBoost와 190점 이상 차이다. 관계 구조보다 기존 asof·TE 트리가 훨씬 강하다.
FLAG dynamic-hier-offset | CLOSED | 2022→2023R +8.03 → 2024 -19.09 | 이전 시즌 CatBoost OOF 잔차로 투수·타자 상태를 만들고 source에서 수축·계수를 고정했다. 투수 상태 연도 상관 +.115, 타자 -.089라 다음 시즌에 유지되는 잠재효과가 아니며 source 선택 이득이 반전했다.
FLAG --rank-group-size 64 | BANNED | 4070 CUDA OOM / A100 segmentation fault | PairLogitPairwise는 그룹 안 쌍을 전개한다. 4070은 추가 2748MB 요구 시 2317MB만 남아 명시적 OOM, A100도 장시간 뒤 native crash. group16 이하만 허용한다.
FLAG --baseline-col skill_pc_hat | REOPENED | 기존 중립 판정 무효 | 기존 CLOSED에는 LEDGER 실행 행이 없고 refit 모델 marker·test-season Pool·fpipe 제출 baseline이 누락돼 target/제출 예측이 오프셋 없이 계산됐다. 세 경로 수정 뒤 BC1으로 다시 측정한다.
FLAG --baseline-col skill_pc_hat | CLOSED | BC1_offset 899.31 vs VB2_base_s42 911.22 | 누락됐던 refit/test/fpipe baseline 경로를 모두 고친 같은 4070·seed42 재검정에서도 −11.91. skill 추정치를 로짓 출발점으로 강제하면 CatBoost가 이미 학습한 수준 효과와 중복된다.
FLAG F-2023-ABS-first-year-analogue | BANNED | 공식 연혁 + reverse .0497→.2988 | KBO는 퓨처스리그 ABS를 2020년부터 운영했다. F 2022→2023 성공률 .7087→.4729와 reverse 6배 급증은 ABS 최초 도입이 아니라 데이터/타깃 체제 단절이다. 이를 R 2024→2025의 1년차→2년차 아날로그로 쓰지 않는다.
FLAG --model rank full-refit | BANNED | group16 양 머신 native crash | no-refit 50iter smoke는 되지만 validation ranker를 폐기하고 full ranker를 만드는 단계에서 4070과 A100 모두 native crash했다. 별도 프로세스 2단 refit을 구현하기 전에는 재실행하지 않는다.
FLAG --boosting-type Ordered | CLOSED | A100 −38.08 / 4070 −10.73 | ordered target statistics와 별개로 boosting scheme 자체를 Plain→Ordered로 바꿨다. A100 미학습 2024와 4070 제출 표면이 모두 같은 머신 단일시드 기준선보다 크게 낮아 다중시드 가치가 없다.
FLAG --feat-window + --failmode-cells | CLOSED | A100 +0.51 / 4070 −5.85 | binary base에서 약한 양수였던 최근 1/3/5경기 산포를 depth5 실패모드 셀에 넣었으나 제출 표면에서 악화했다. ZD5와 centered RMS도 .0025라 새로운 실패형태 다양성이 없다.
FLAG frozen-career-middle-q8-k200 | CLOSED | LB 1101.802→1083.531, −18.271 | v11의 recent-5-game middle q8/k500을 career cumulative middle q8/k200으로 전량 교체하고 PB를 재적합한 v12. 2023→2024 상대 이득 +3.174는 전반 −0.292/후반 +7.704로 한쪽에 몰렸고, 2022→2023에서는 career 경로 자체가 −31.003인데 기존 경로보다 덜 나쁘다는 상대값 +23.657을 안정성으로 오판했다. 2024 고정 구간값도 비단조(+.00646, −.00365, −.00333, +.00456, +.00394, +.00062, +.00312, −.01169; 결측 −.01453)이며 k=200은 분위별 수만 행에 사실상 수축이 아니다. PB 테이블은 v11과 공통 26,355그룹·상관 .999176·차이 RMS .000130이므로 실패 원인은 PB가 아니라 career-middle 잔차 lookup의 시즌 전이 붕괴다. 재제출·부분가중·LB 기반 계수 재적합 금지.
FLAG --feat-id-cohort | CLOSED | 둘 다 전이 −3.18/+13.93/+8.13, 4070 −2.42; 투수 +1.33/−0.22, 타자 −0.10/−0.71 | 익명 선수 ID 앞 3자리는 실제 등록 코호트 구조(242/243=2022, 244=2023, 245=2024 등)를 담지만, full ID suffix를 버린 수치 prefix는 현행 asof_n·roster 신호 위에 안정적인 증분을 남기지 못했다. 둘을 함께 넣은 A100 seed3 +8.13은 단독 두 축이 모두 0이고 다른 전이·머신에서 반전하므로 트리 경로/early-stop 변동을 고른 자기선택이다.
FLAG historical-lineup-role | CLOSED | target 최고 +0.444, early +8.911 / late −10.591 | Trackman 과거 타순 lookup coverage 90%지만 role×same-hand조차 전·후반 부호가 반전했다. slot, role, count, league, 안정성·entropy 등 다른 arm은 전체 음수. 실제 2025 타순도 없으므로 확장하지 않는다.
FLAG historical-mechanics-change | CLOSED | target 최고 −0.245, early +18.801 / late −25.268 | 전년도 릴리스 위치·산포·구속·무브먼트와 변화량을 recent-middle과 함께 정직한 2023→2024 전이로 감사했다. 최고 rel-side 변화도 전체 음수이고 반분 부호가 크게 갈렸다. 과거 기계적 프로필은 현재 제구 상태로 안정 전이되지 않는다.
FLAG pb-residual-matrix-factorization | CLOSED | K4−K0 −2.639 / +0.350 / +0.551 | exact-PB residual matrix를 rank 4로 분해해 known player/new pair까지 확장했다. 2021→22는 전체 음수, 뒤 두 전이는 크기가 +1 미만이고 후반기 음수. known-new-pair도 +1.256→−0.730으로 반전해 상성을 이웃 pair로 전파할 근거가 없다.
FLAG recent-state-empirical-bayes | CLOSED | middle −3.288/+3.017/−1.247, success −2.694/+8.318/−5.633 | prev1/3/5를 1경기·2~3·4~5 블록으로 분해하고 강한 Ridge 수축으로 K0 잔차를 보정했다. middle/success 모두 세 rolling 전이에서 부호가 반복 반전했다. LB에서 성공한 고정 prev5-middle을 유지하되 복잡한 상태 확장은 하지 않는다.
FLAG workload-pace | CLOSED | +6.332 / +1.262 / −2.167 | 당해 시즌 투구수÷월 진행도, 전년도 workload, 전년도 역할×현재 이닝을 사용했다. 최신 2023→24 전체·전후반이 음수이고 F는 −25.587. 월 기반 사용강도는 역할 변화·복귀를 안정적으로 분리하지 못한다.
FLAG team-call-style | CLOSED | 연도 combined corr 중앙값 +.227 < gate .25 | 수비팀×count family×주자×타자손의 middle/ball/reverse 구성비를 k300 Dirichlet 수축했다. 최근 2023→24는 +.398이나 앞선 네 전이 중 세 개가 .205~.227이고 middle은 −.419도 있어 코칭/포수 사인 성향의 지속성이 부족하다.
FLAG intent-execution-bilinear | CLOSED | +0.390 / +3.502 / −2.919 | count-family×runner×same-hand 의도와 투수 recent/career 실패상태를 rank-4 bilinear residual로 부분공유했다. 가운데 전이만 +3을 넘고 최신 2023→24는 R −3.439, 전반 −3.144, 후반 −2.626. hard routing 반복 없이도 의도×실행 상호작용이 현행 모델에 증분을 주지 못했다.
FLAG pb-familiarity-adaptive-k | CLOSED | −4.466 / −2.123 / −0.107 | source 시즌 이전에도 만난 familiar pair만 exact-PB 수축을 k500→k250으로 완화했다. familiar 그룹은 각 source의 9,417/9,691/11,412개였으나 세 전이 모두 악화했다. 반복 대결도 PB 표본 잡음을 덜 수축할 근거가 아니며 현 k500을 유지한다.
FLAG posterior-uncertainty-residual-head | CLOSED | full −8.037/−123.482/−76.637; fixed 5% +1.462/−3.060/−0.083 | 현재시즌 pitcher success/middle 누적을 k80 binomial posterior mean·sd·precision으로 복원하고, 현행 K0 OOF 위에 intercept 없는 강수축 Ridge 잔차 head를 source 시즌에서만 학습했다. source 이득은 +66~128로 컸지만 다음 시즌 부호가 반복되지 않았고 최신 전이도 early +0.969 / late −1.458이었다. 단순 calibration이 아니라 uncertainty-조건부 resolution을 직접 검정했어도 시즌 잔차를 외웠으므로 GPU·제출 확장 금지.
FLAG trackman-link-batter-hand | INFRA PASS | 730→755명, train 행 커버 99.6376%→99.7925%, 기존 공통 730명 일치 100% | 기존 “86%”는 미매칭률이 아니라 옛 매핑 정확도 추정이었다. 실제 고신뢰 map2는 이미 99.64% 행을 덮었다. 유일하게 빠진 공통 행 단독 키 batter_hand를 추가해 저표본 25명(2,285행)을 복구했지만 전체 추가 커버는 0.1549%뿐이다. Trackman 피처 자체는 이미 CLOSED이므로 점수 레버가 아니라 향후 재현용 링키지 개선이다.
FLAG local-targeted-pruning | CLOSED | pitchmix delta -4.99, outcome delta -16.22, low-importance indicators -11.88, batter_team_id -45.21 | 같은 로컬·seed42·2023→2024 표면에서 의미 단위 삭제를 실제 재학습했다. 고정 모델 LossFunctionChange 공통 음수도 제거 인과효과가 아니었고, 낮은 gain importance 열도 다른 분할 경로를 지탱했다. 광범위 E47뿐 아니라 이 네 소규모 삭제도 반복하지 않는다.
FLAG --feat-recent-pair-bin | CLOSED | raw q8 -35.63, season-expanding TE -7.84 | 세 전이에서 안정적으로 보였던 reverse×prev3-success 관계를 fit-only 동결 2D q-bin 범주와 과거시즌 전용 k500 TE로 각각 표현했다. 둘 다 기준 876.90보다 악화해 기존 연속축이 이미 관계를 흡수한다.
FLAG --te pb | CLOSED | PBTE1 -17.56 | exact pitcher×batter의 성공률 TE를 모델 입력으로 넣으면 과적합한다. PB의 실전 +1.7은 코어 피처가 아니라 강수축한 작은 사후 residual일 때만 유지된다.
FLAG prior-pa-depth-proxy | CLOSED | 최선 +0.089/+0.565 | 완료된 과거 시즌의 투수·타자 2스트라이크/깊은 카운트/풀카운트 비율을 k500 수축해 현재 count에서 gate했다. 두 rolling 전이 최소 이득이 +1 미만이라 실제 투구번호·PA ID 없는 데이터에서는 유효한 타석 피로 proxy가 되지 못한다.
FLAG --feat-count-cat | CLOSED | FCC1 867.33, 기준 대비 -9.57 | 볼-스트라이크 정확한 12상태를 저카디널리티 범주로 추가했다. 원본 두 수치열과 pitcher×count TE/skill이 이미 같은 정보를 흡수한다.
FLAG --feat-quality-min | CLOSED | 잔차 lookup +9.75/+11.84 → 실제 재학습 -10.95 | std 투수 성공률과 std 타자 상대 성공률의 bottleneck 최소값은 source residual map에서는 두 전이 양수였지만 CatBoost 입력에 넣으면 865.95로 악화했다. 사후 잔차 해상도를 결정론적 피처 증분으로 옮길 수 없다.
