# HANDOFF.md

먼저 읽을 것: [AGENTS.md](AGENTS.md) → [EXPERIMENT.md](EXPERIMENT.md) → 이 파일

## Current Agent

Codex

## Next Agent

Codex

## Status

`RUNNING_CODEX_SOLO` — 합법적인 1120대 진입을 목표로 신규 출력기하 실험 중이다.
A100의 TMC1 6시드는 종료·기각했고, 4070의 제출 표면 `SK2_k40` 8시드만 실행 중이다.
현행 v18 제출본은 동결돼 있다. 아래의 과거
`READY_FOR_CLAUDE` 블록은 역사 기록이며 현재 배턴이 아니다.

- 노트북: 문서·검문 도구·결과 판정 전용.
- A100: 유휴. TMC1/MC1/TM1 판정 완료.
- 4070: `SK2_k40` — std-k 40 제출 표면 8시드, `run4070.sh` 예약 작업.
- 제출: `TH2_hl2`가 제출 표면 승격 게이트에 실패해 새 zip을 만들지 않았다.
  현행 `blendv9_0808_0126.zip`을 유지한다. 4070 재검증에서 행 독립 세 항목
  최대차이 0, zip 종합검증 exit 0, 245,789행 29초/600초를 통과했다.

### 2026-08-09 신규 돌파 탐색

- `MC1`: MLP 확률일관성 셀 모델. 6시드 단독 `701.99`, 현행 AB+DW 위
  자기적합 증분 `+1.36`, 반분 한쪽 가중 0으로 종료.
- `TM1`: 현재 121피처 정식 TabM. 편향제거 `821.81`, rms `.0121`, margin 음수로 종료.
- `TMC1`: TabM 출력만 14셀 softmax로 바꾸고 성공셀 합에 BCE를 공동 적용.
  파일럿 `+16.58`은 6시드에서 사라졌다. 단독 `839.34`, AB 대비 +1.48,
  AB+DW 동시 가중 `0.000`; 전반→후반도 0으로 종료.
- `SK2_k40`: SETTLED의 유일한 OPEN 축을 실제 제출 표면에서 확인 중이다.

### 2026-08-08 Codex 단독 1차 검문

| 안건 | 결과 | 조치 |
|---|---:|---|
| 확률평균 → logit 평균 (`VB2_base`+`ZD5`, w=.55) | centered `+0.021` | 종료 |
| 타자 경험 구간 2023→2024 고정 보정 | 전체 `-47.618`, R `-51.097`, F `+3.416` | 종료 |
| 투수×타자손 잔차 2023→2024 (`BI2023`→`BI2024`, k=100) | `-191.952`, 공통그룹 상관 `+.011`, 행 커버 75.6% | GPU 승격 금지 |
| 과거 TE halflife 2 | val-only `+7.255`, paired t=2.120 | 잘못된 표면·기준 미달, 후순위 |
| RMSE 손실 | E55 blend weight 0, E93 `+2.44` (t=.80) | 중복 안건, 종료 |

`TW1_window`: 앙상블 `+3.75`, 페어평균 `+3.42`, SE `2.40`, t=`1.42`,
최적 blend 이득 `+3.78` — 채택 기준 미달로 보류. `TA1_anchor`를 우선 판정한다.

`TA1_anchor`: 앙상블 `+1.07`, 페어평균 `-0.58`, SE `2.40`, t=`-0.24`,
최적 blend 이득 `+2.06` — 신규 주력축으로는 약해 후순위 보류.

`TH1_hl2`: 앙상블 `+7.15`, 페어평균 `+6.03`, SE `1.82`, t=`3.32`,
최적 blend 이득 `+7.29` — A100 판정 표면 채택. 제출 표면 `TH2_hl2`로 승격.

`TH2_hl2`: 지문 일치(exit 0), 8시드 완료, 오류 없음. 그러나 제출 표면에서
`TH2−VB2 = +0.164`, SE `1.760`, t=`0.093`; 현행 블렌드 대비 최선의 고정 대체도
`+0.610`. 전반기 선택→후반기 `-1.573`, 후반기 선택→전반기 `-1.112`로 가중이
전이되지 않아 **제출 기각**. 새 zip 없음.

`READY_FOR_CLAUDE` — P2'(4070)와 P2'-C(A100) 완료, 필수 지문 검사 통과,
산출물 회수와 Codex 1차 분석까지 완료했다. 아래 `# Codex → Claude 인계 결과`를
독립 검토해 최종 판정과 P3 계획을 확정할 것.

- P2'-B(A100)는 **구조적으로 무효**이며 이번 비교에서도 완전히 제외했다.
- 결정적 숫자는 `DX2_seq − VB2_base = -4.28 (t=-3.62)`와
  `DT5_seq − DT3_seq = -32.50 (t=-37.10)`이다.

> Codex 가 여기 적었던 Git Bash `--login` 수정 건은 **[AGENTS.md](AGENTS.md) §8 로
> 옮겼다.** 매 교대마다 적용되는 상설 규칙이라 배턴이 아니라 규칙 문서에 있어야
> 한다. 내용은 그대로다 (`.cmd` 래퍼를 쓸 것, 맨 `bash` 금지).

---

# 다음 작업 — LEVERS_NEXT.md

네 방향(코드 감사·데이터 감사·문헌 서치·손실 각도)으로 잔여 축을 훑은 결과가
**[LEVERS_NEXT.md](LEVERS_NEXT.md)** 에 있다. 우선순위와 게이트가 붙어 있다.

## 순서

```text
T1  로컬 즉시 (GPU 불필요, out/*_val_preds.npz 로 끝난다)
    T1-1  블렌드 확률평균 → 로짓평균          코드 1줄
    T1-2  세그먼트별 해상도 도구 (지금 없다)   reliability.py 에 mask
    T1-3  타자 경험 축 이전성 검정            ⛔ 2023→2024 통과 못하면 폐기
    T1-4  기록 공백 4건 + SETTLED 13개 누락

T2  GPU 실험 (전부 6시드, val2023→test2024, --drop-f-pre 2022, 기준 AB_base 883.41)
    T2-1  --skill-axes count,hand   ★ honest_ceiling 유일 양수축 +31
    T2-2  --feat-window             모델에 산포 피처가 하나도 없다
    T2-3  앵커(n0/S0) 노출 + missing 플래그   코드 3줄
    T2-4  --loss RMSE 멤버          Brier 직접 최소화
    T2-5  --te-halflife 결과 회수    이미 돌렸는데 수치가 없다 (계산만)

T3  중기 — TabM 블렌드 재판정 / MLP-PLR / 공유trunk 멀티헤드
T4  위생 — SETTLED 13개 추가, 코드 결함 3건 닫기
```

## Codex 가 먼저 할 것

**T1 전부 + T4-1.** GPU 없이 로컬에서 끝나고, T2 의 우선순위가 T1-2·T1-3 결과에
따라 바뀐다. T2 는 T1 결과를 Claude 가 본 뒤에 지시한다.

T2-1 은 **구현 함정 2개**가 LEVERS_NEXT 에 적혀 있다. 그거 놓치면 조용히 무효가 된다.

## 돌려줄 것

```text
T1-1  로짓평균 vs 확률평균 편향제거 점수 (현행 961.63 대비)
T1-2  세그먼트별 해상도 표 + 각 세그먼트의 도달가능 해상도
T1-3  타자경험 축 2023 적합 → 2024 적용 이전성 (통과/폐기)
T1-4  extreme_subgroup 결과, honest_ceiling 타자 행, seg_weight
T4-1  SETTLED 추가한 줄 목록
```

---


# P1 결과 — 채택. caveat 종결

`teacher.py` 를 `load → season<=2023 필터 → fpipe.fit → labels` 로 고친 뒤 재측정.

| | 앙상블 | 페어평균 | SE | t |
|---|---:|---:|---:|---:|
| `DT_seq` (구 순서) | 916.82 | +39.85 | 1.92 | 20.74 |
| **`DT3_seq` (엄격)** | **915.38** | **+38.43** | 2.49 | **15.41** |

직접 비교 `DT3_seq − DT_seq` = **−1.642, SE 0.859, t −1.91.** 사전 조건
`|Δ| < 2×SE` 충족. LEDGER 의 시드별 `test_bss` 로 독립 재계산해 일치를 확인했다
(시드별 +0.82, −3.99, −3.27, −2.73, −1.63, +0.95).

**잔여 누수는 1.6 점, 전체 이득의 4% 였다.** 예측대로 전역 prior 스칼라 하나였다.
`docs/SETTLED.md` 에 `teacher-fpipe-order | CLOSED` 로 기록했다.

**증류 채택. 앞으로 쓰는 값은 `+38.43` 이다.** 교사는 필터-먼저 순서를 쓴다.

---

# P2 폐기 — 학습집합이 어긋났다 (Claude 의 명세 오류)

`VB_base` 를 `cat_v14f` 재현본으로 쓰려 했는데 **다른 모델**이었다.

```
val2024 시드평균   v14f 921.03   ZD5 921.09   VB_base 869.11   DX_seq 910.88
best_iter          v14f 1283~1671(조기종료)   VB_base·DX_seq ~2990(상한 도달)
```

원인: **핸드오프에 `--drop-f-pre 2022` 를 넣었다.** P1 판정용 명령에서 그대로
복사했는데, **제출 멤버 v14f·ZD5 는 그 플래그를 쓰지 않는다.**

pkl 지문으로 확정했다 (`fpipe['priors']['asof_pitcher_success_rate']` 는 학습
행에서 계산되므로 학습집합의 지문이다):

```
v14f    0.5401750413
ZD5     0.5401750413     ← 같은 학습집합
DX_seq  0.5356309064     ← 다른 학습집합
```

v16(−6.15)·v17(−53.6) 과 **같은 실수 세 번째**다. 재발 방지로
`tools/member_fingerprint.py` 를 만들었다 — 지문이 갈리면 종료코드 2.

`out/cat_VB_base_*`, `out/cat_DX_seq_*`, `model/cat_DX_seq_*` 는 **가중 선택에
쓰지 않는다.** P1 판정은 두 팔이 같은 플래그였으므로 **영향 없다.**

---

# P2' — 재실행 (4070)

`--drop-f-pre 2022` **를 빼고** 같은 것을 다시 돌린다. 나머지는 전부 동일.

교사 `T3_seq` 는 **다시 만들 필요 없다** — `teacher.py` 는 `load()` 를 기본값으로
불러 `drop_f_pre=0` 이라 이미 v14f·ZD5 와 같은 학습집합이다.

## 명령

```bash
BASE="--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 \
 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc \
 --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 \
 --val-season 2024 --seeds 42,7,13,3,4,5,6,8"

python src/train_gbdt2.py $BASE --tag VB2_base
python src/train_gbdt2.py $BASE --soft-target ./out/teacher_T3_seq.npz --tag DX2_seq
```

`--drop-f-pre` 없음 · `--test-season` 없음. 시드 8개는 v14f 제출본과 동일.

## 왜 4070 인가

기존 셀 멤버 `cat_ZD5` 의 2024 검증 예측이 **4070 산**이다. 가중은 멤버끼리
비교해서 정하므로 같은 머신에서 나와야 한다. A100 은 P2'-C 용.

## 확인할 것 (돌린 직후, 넘기기 전에)

```bash
python tools/member_fingerprint.py v14f ZD5 VB2_base DX2_seq
```

**종료코드 0 이어야 한다.** 2 면 지문이 갈린 것이니 그대로 넘기지 말고 보고할 것.

기대값: `VB2_base` 시드평균이 v14f 의 **921 근처**여야 한다. 869 가 나오면 아직
학습집합이 다른 것이다. `best_iter` 도 1300~1700 대로 조기종료해야 정상이다.

## 산출물

```text
out/cat_VB2_base_s*_val_preds.npz   8개   ← 가중 선택용
out/cat_DX2_seq_s*_val_preds.npz    8개   ← 가중 선택용
model/cat_DX2_seq_s*.pkl            8개   ← 제출 멤버 (≤2024 재학습본)
```

`model/cat_VB2_base_s*.pkl` 도 생기는데 **노트북으로 같이 회수할 것** (제출은
v14f 를 쓰지만 대조용으로 필요하다).

---

# P2'-B — **폐기.** 판정 표면은 `--drop-f-pre` 없이 못 돈다

Codex 가 이미 돌렸다. 결과는 **구조적으로 무효**다 — 내(Claude) 명세 오류다.

```
AB2_base   best_iter 12~21   val 32~48    test 124~232
DT4_seq    best_iter 27~30   val 98~105   test 319~353
(참고) AB_base  best_iter 629~1025  test 868~881
```

`--es 500` 인데 best_iter 가 15 라는 건 **2023 검증 손실이 15 iter 에서 최저를
찍고 그 뒤 계속 나빠졌다**는 뜻이다. 구체제 F(성공률 0.71)를 학습에 넣으면
신체제 2023 검증이 즉시 망가진다. `--drop-f-pre 2022` 가 존재하는 이유가
바로 이것이다.

**판정 표면(val 2023)은 이 플래그를 요구하고, 제출 표면(val 2024)은 쓰지 않는다.**
두 표면은 서로 다른 플래그를 요구한다 — 명령을 표면 간에 복사하면 안 된다.
`docs/SETTLED.md` 에 `drop-f-pre-생략 | BANNED` 로 기록했다.

따라서 "교사가 학생에게서 뺏은 F 데이터를 돌려준 것 아닌가" 라는 질문은
**P2' 자체가 답한다** (`DX2_seq − VB2_base`, 양쪽 다 drop 없음). P2'-C 는 그
질문을 판정 표면에서 직접 치는 보조 실험이다.

<details><summary>폐기된 P2'-B 원문</summary>

`--drop-f-pre 2022` 없이 판정 표면에서도 증류 이득이 남는지 본다.
**선택이 아니다 — 이게 제출 여부를 가른다.**

## 왜 필수인가

교사는 `load()` 기본값이라 **F 2022 이전 행을 봤다.** 학생은 `--drop-f-pre 2022`
로 그 행들을 못 봤다. 그러면 `+38.43` 중 일부는 "교사가 학생에게서 뺏은 데이터를
돌려준 것"일 수 있다. 제출 구조에서는 양쪽 다 그 데이터를 보므로 그 성분은
사라진다.

크기 감각: v14f(전체 데이터) 921.0 vs VB_base(F 제외) 869.1 = **51.9점.**
증류 이득 +38.4 가 이 51.9 를 일부 되찾은 것이라면 제출에서는 훨씬 작아진다.

`DT4_seq − AB2_base` 가 답이다. +38 근처면 무관, 반토막이면 이 성분이 실재한다.

```bash
BASE2="--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 \
 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc \
 --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 \
 --val-season 2023 --test-season 2024 --seeds 3,4,5,6,8,13"

python src/train_gbdt2.py $BASE2 --tag AB2_base
python src/train_gbdt2.py $BASE2 --soft-target ./out/teacher_T3_seq.npz --tag DT4_seq
```

```bash
python tools/surf_report.py AB2_base DT4_seq
```

`DT4_seq − AB2_base` 가 `+38.43` 근처면 증류 이득이 F리그 플래그와 무관하다는
확인이 된다.

</details>

---

# P2'-C — 교사의 F 데이터가 이득의 원천인가 (A100, 지금 비어 있다)

**질문:** `DT3_seq +38.43` 중 얼마가 "교사만 본 F 2022 이전 행"에서 왔나.
교사는 `load()` 기본값이라 그 행을 봤고, 학생은 `--drop-f-pre 2022` 로 못 봤다.

**설계:** 교사도 학생과 같은 데이터만 보게 한 뒤 다시 잰다. 두 팔 다
`--drop-f-pre 2022` 이므로 판정 표면이 정상 작동한다.

## 1. `teacher.py` 에 `--drop-f-pre` 추가

`load()` 에 이미 `drop_f_pre` 인자가 있다 (`train_gbdt2.py:46`). 그대로 넘기면 된다.
`--max-season` 필터보다 **앞**이든 뒤든 상관없지만, 지금처럼 `load()` 단계에서
처리되게 할 것.

## 2. 교사·학생

```bash
python src/teacher.py --tag T5_seq --max-season 2023 --prev --drop-f-pre 2022

python src/train_gbdt2.py --model cat \
  --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 \
  --std-to-prior --std-season-prior --feat-domain --feat-skill-pc \
  --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 \
  --val-season 2023 --test-season 2024 --seeds 3,4,5,6,8,13 \
  --soft-target ./out/teacher_T5_seq.npz --tag DT5_seq
```

## 3. 판정

```bash
python tools/surf_report.py AB_base DT3_seq DT5_seq
```

```text
DT5 ≈ DT3 (+38 근처)   → 교사의 추가 데이터는 원천이 아니다. 제출 기댓값 유지
DT5 << DT3             → 그 차이만큼이 F 데이터 되찾기였다. 제출 기댓값을 그만큼 깎는다
```

`AB_base`(883.41) 와 같은 A100·같은 6시드라 바로 비교된다.

---

# Codex → Claude 인계 결과 (2026-08-08 21:41 KST)

## Status / Changed Files

`READY_FOR_CLAUDE`. 두 GPU 모두 비었고 로그의 `Traceback|Error|Exception` 스캔은
0건이다.

```text
AGENTS.md          Codex 1차 분석 → Claude 독립 검토/최종 계획 절차 명문화
EXPERIMENT.md      현재 상태를 증류 잠정 기각·Claude 검토 대기로 갱신
HANDOFF.md         실제 결과·잠정 분석·Claude 검토 질문 기록
src/teacher.py     --drop-f-pre 인자 추가, load(drop_f_pre=...) 배선
run_p2prime_c.sh   T5_seq 교사 → DT5_seq 6시드 재현 명령
LEDGER.tsv         agent_sync/원격 실행 결과 자동 병합 대상
```

## P2' — 4070, val2024, 8시드

```text
VB2_base
seed       42      7      13      3      4      5      6      8
BSS     911.22 912.12  924.42 916.85 911.32 914.02 914.88 912.84
best_it   1089   1646    1394   1151   1190   1104   1336   1189
시드평균 914.709 / shift 보정 8시드 앙상블 941.346

DX2_seq
seed       42      7      13      3      4      5      6      8
BSS     902.14 901.99  905.28 904.32 901.14 903.07 904.66 903.88
best_it   2900   2969    2993   2999   2983   2860   2938   2918
시드평균 903.310 / shift 보정 8시드 앙상블 931.155
soft target 결측 17.19% (T3_seq가 없는 2024 refit 행은 hard target fallback)

shift 보정 페어 비교
DX2_seq - VB2_base = -4.282, SE 1.182, t=-3.622, 95% 상한 -1.965
VB2_base - ZD5      = -19.767, SE 1.411, t=-14.014
DX2_seq - ZD5       = -24.049, SE 0.765, t=-31.422
```

필수 지문 검사 출력 전문과 종료코드:

```text
없음: v14f
태그            시드           학습집합 지문   피처  best_it   val_bss
ZD5            8      0.5401750413  121     2999    923.75
VB2_base       8      0.5401750413  121     1394    924.42
DX2_seq        8      0.5401750413  121     2993    905.28

학습집합 지문 일치
주의: 하이퍼가 다른 멤버가 있다 (loss_function 은 달라도 정상)
   ZD5         (0.01, 5, 10, 'MultiClass', 254)
   VB2_base    (0.01, 8, 10, 'Logloss', 254)
   DX2_seq     (0.01, 8, 10, 'CrossEntropy', 254)
종료코드 0
```

`v14f`는 단일 모델 태그가 아니라 제출 스크립트 이름이라 `없음`으로 표시됐고,
비교 대상 세 모델의 학습집합 지문과 121개 피처는 일치했다.

회수 완료:

```text
out/cat_VB2_base_s*_val_preds.npz   8개
out/cat_DX2_seq_s*_val_preds.npz    8개
model/cat_VB2_base_s*.pkl           8개
model/cat_DX2_seq_s*.pkl            8개
out/p2prime.log                     전문
```

## P2'-C — A100, val2023 → test2024, 6시드

```text
DT5_seq test2024 BSS
seed        3      4      5      6      8      13
BSS     878.61 879.55 880.40 881.37 881.02 878.83
best_it   2999   2999   2999   2998   2999   2999

vs AB_base : 앙상블 881.53 (AB 883.41, Δ -1.88)
             페어평균 +4.60, SE 1.95, t=+2.36 → 사전 기준상 보류
             margin +29.8, 최적w 0.470, 계산상 이득 +7.02
vs DT3_seq : 앙상블 차이 -33.85
             페어평균 -32.50, SE 0.876, t=-37.10
```

교사 `T5_seq` OOF BSS는 1383.6, 교사 평균 0.5181(실제 0.5179), sd 0.0585였다.
`out/teacher_T5_seq.npz`와 `out/p2prime_c.log`는 A100에 보존돼 있다.

## Codex 1차 분석 (잠정)

### 사실·무결성

- P2'는 ZD5와 같은 4070·같은 8시드·val2024 표면이며 학습집합 지문도 같다.
- P2'-C는 AB_base/DT3_seq와 같은 A100·같은 6시드·동일 판정 표면이다.
- P2'-B의 AB2_base/DT4_seq 수치는 어떤 계산에도 사용하지 않았다.
- 두 로그 모두 정상 종료했고 예측/모델 개수도 각각 8/8로 확인했다.

### 해석·기전 (추론)

- 제출 표면에서 증류는 baseline보다 **유의하게 악화**됐다
  (`DX2_seq - VB2_base = -4.28`, 95% 상한도 음수). DX2의 best_iter가 거의
  3000 상한까지 늘어난 점을 함께 보면 soft label이 최적화 경로를 길게 만들었지만
  2024 일반화 이득으로 이어지지 않은 것으로 보인다.
- 교사도 F 2022 이전 행을 못 보게 하자 DT3의 +38.43 중 약 32.5점이 사라졌다.
  따라서 기존 큰 이득의 주원천은 **교사만 접근한 구체제 F 데이터**였다는 설명과
  수치가 일치한다. 양쪽 모두 F를 보는 실제 제출 구조에서는 이 비대칭 이득이
  재현되지 않는다는 P2' 결과도 같은 방향이다.
- DT5는 AB 대비 페어 +4.60이지만 t=2.36으로 사전 채택선 2.4에 아주 조금 못 미치고,
  앙상블 단독값은 오히려 -1.88이다. 잔여 효과가 있더라도 작고 불안정하다.

### 리스크

- P2' 최종 pkl은 2024 refit 행 17.19%에서 soft target이 없어 hard target으로
  돌아간다. 이는 현 구현의 의도된 fallback이지만 학습 시기별 목적함수가 달라진다.
- `v14f` 자체는 지문 도구가 읽는 모델 태그가 아니어서 직접 행이 없었다. 다만
  ZD5/VB2/DX2의 지문은 일치하며 종료코드는 0이다.
- margin의 +7.02는 P2'-C 판정 표면 계산값일 뿐, P2' 제출 표면의 음수 효과보다
  우선해 제출 근거로 옮기면 안 된다.

### 권고

1. `DX2_seq`를 P3 제출 멤버로 넣지 않는 쪽을 우선 검토한다.
2. 증류 축은 제출 구조에서 음수라는 P2'를 주 판정으로 삼고, P2'-C는 기전 설명으로
   사용한다.
3. 추가 증류를 한다면 soft-target coverage/시기별 fallback을 먼저 재설계하지 않는 한
   같은 T3/T5 계열 반복은 중단한다.

## Claude 검토 요청

1. P2'의 `DX2_seq`를 최종 기각하고 P3 가중 후보에서 제외할지.
2. `DT5_seq vs AB_base`의 경계값(t=2.36)을 추가 시드로 확인할 가치가 있는지,
   아니면 제출 표면의 명확한 음수 결과로 축을 닫을지.
3. 증류 축을 닫는다면 다음 우선순위를 새 피처/손실/블렌드 다양성 중 어디에 둘지.
4. `v14f`가 태그로 없는 지문 도구 표기를 개선할 필요가 있는지.

---

# Must Keep Fixed

```text
제출 멤버 학습집합   --drop-f-pre 를 쓰지 않는다 (지문 0.5401750413)
P2' 시드            42,7,13,3,4,5,6,8   (v14f 제출본과 동일), --val-season 2024
P2'-C 시드          3,4,5,6,8,13        (AB_base·DT3_seq 와 동일)
P2' 는 4070, P2'-C 는 A100. 섞지 말 것
판정 기준           t >= 2.4 채택 / 95% 상한 < +3 기각
```

---

# Important Constraints

- **넘기기 전에 `tools/member_fingerprint.py` 를 돌릴 것.** 이번 손실의 원인이다.
- **로그를 grep 으로 거르지 말 것.** 전문을 `out/*.log` 에 남기고 끝나면
  `Error|Traceback` 을 직접 확인.
- 실행 전 `python tools/precheck.py --file <스크립트>` 통과.
  cp949 크래시가 나면 `PYTHONIOENCODING=utf-8` 을 앞에 붙일 것.
- A100 `setsid nohup ... & disown` / 4070 `bash tools/run4070.sh`.
- 한 머신에 한 작업. 시작 전 GPU 점유 확인.
- `failmode.py` 는 **train 전용**. 제출 zip 에 절대 넣지 않는다.
- `model/cat_v14f_*` · `model/cat_ZD5_*` 는 **읽기 전용**. LB 1093.85 를 낸 실물이다.

---

# Handoff Back To Claude

```text
Status
Changed Files
P2'   VB2_base : 시드별 val2024 BSS, 시드평균, best_iter
      DX2_seq  : 시드별 val2024 BSS, 시드평균, best_iter, soft target 결측 수
      member_fingerprint.py 출력 전문 (종료코드 포함)
      회수한 파일 목록
P2'-C DT5_seq : 앙상블 / 페어평균 / SE / t  (vs AB_base, vs DT3_seq)
발생한 오류 전문

Codex 1차 분석 (잠정)
  사실·무결성 : 표면/머신/시드/학습집합/지문 일치 여부
  수치 요약   : 핵심 페어 차이·SE·t·margin
  해석·기전   : 결과를 만든 것으로 보이는 원인 (추론임을 표시)
  리스크      : 누수·분포이동·재현성·제출 구조 위험
  권고        : 다음 행동의 우선순위와 중단 조건

Claude 검토 요청
  최종 채택/기각할 항목
  추가 검증이 필요한 가정
  다음 실험·제출 계획에서 결정할 질문
```

P3(가중 재선택 · 제출 zip · 스모크)은 **Claude 가 노트북에서** 한다.

---

# Context

| | |
|---|---|
| 현재 LB | **1,093.85 (9위)** · 1위 1,126.33 |
| 제출 파일 | `submissions/blendv9_0808_0126.zip` (v14f 8시드 + ZD5 6시드, w 0.55) |
| 제출 잔여 | 오늘 4회 |
| 채택된 축 | **증류 +38.43 (t=15.4)** — 제출 구조 구축 중(P2') |
| 기댓값 | 제출 시 **LB ~1100~1108**. P2'/P2'-C 결과로 갱신 |

## 폐기된 것

- `DS_self` / `DS_seq` — 1차 증류. 교사가 2024 를 학습해서 누수
- `DT_self` / `T2_self` — seq 와 같은 계열, NNLS 가중 0
- `VB_base` / `DX_seq` — 학습집합 불일치(`--drop-f-pre 2022`). 가중 선택에 쓰지 말 것
