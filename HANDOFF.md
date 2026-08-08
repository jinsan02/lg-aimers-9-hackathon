# HANDOFF.md

먼저 읽을 것: [AGENTS.md](AGENTS.md) → [EXPERIMENT.md](EXPERIMENT.md) → 이 파일

## Current Agent

Claude

## Next Agent

Codex

## Status

`RUNNING` — 4070 에서 **P2' `VB2_base` 진행 중**(2/8, 20:29 기준). Codex 가 이미 걸었다.
A100 은 비었다 → `# P2'-C` 를 걸 것.

- P2'-B(A100)는 **구조적으로 무효**로 판명났다. 아래 참조.
- 오늘의 결정적 숫자는 이제 **`DX2_seq − VB2_base`**(P2') 와 **`DT5_seq − DT3_seq`**(P2'-C) 둘이다.

> Codex 가 여기 적었던 Git Bash `--login` 수정 건은 **[AGENTS.md](AGENTS.md) §8 로
> 옮겼다.** 매 교대마다 적용되는 상설 규칙이라 배턴이 아니라 규칙 문서에 있어야
> 한다. 내용은 그대로다 (`.cmd` 래퍼를 쓸 것, 맨 `bash` 금지).

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
