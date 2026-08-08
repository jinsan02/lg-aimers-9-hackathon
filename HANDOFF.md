# HANDOFF.md

먼저 읽을 것: [AGENTS.md](AGENTS.md) → [EXPERIMENT.md](EXPERIMENT.md) → 이 파일

## Current Agent

Codex

## Next Agent

Claude

## Status

`READY_FOR_CODEX` — E165b 검증 완료(아래 판정). **P1·P2 두 단계를 이어서 진행한다.**
두 GPU 모두 비어 있다. P3(가중 선택·제출 패키징)은 Claude 가 노트북에서 한다.

---

# E165b 판정 (Claude, 2026-08-08)

**채택 방향.** LEDGER 의 시드별 `test_bss` 로 독립 재계산했고 Codex 수치와 일치한다.

| | 페어평균(LEDGER 재계산) | SE | t |
|---|---:|---:|---:|
| `DT_self` | +36.20 | 2.34 | 15.5 |
| `DT_seq` | **+41.14** | 2.00 | **20.5** |

시드 6개 전부 같은 부호(+33.6 ~ +47.8). `DT_self` 는 폐기 — seq 와 rms .0018 로
같은 계열이고 NNLS 가중 0 이다. **`DT_seq` 하나만 간다.**

## 남은 caveat — 확인했고, 좁다

`teacher.py:74-75` 가 `is_val = 전부 False` 라 `fpipe.fit` 이 2019~2024 전부를 봤다.
학생·기준선은 `train_gbdt2.py:800` 에서 `is_fit = ~is_val & ~_is_test` 로 2023·2024
를 **둘 다** 뺀다. 비대칭이 맞다.

다만 실제로 흐르는 양은 좁다. `target_enc.build_te` 는 시즌 expanding + `shift(1)`
(`target_enc.py:87-88`) 이라 **2023 행의 키별 rate 는 ≤2022 만 쓴다 — 2024 타깃은
행 단위로 안 들어간다.** 유일한 경로는 `prior = df[TARGET].mean()`
(`target_enc.py:56`) 이라는 **전역 스칼라 하나**다. 이걸로 +41 이 나올 수 없다.

그래도 P1 로 닫는다. 1차 누수도 사전엔 "설마" 였다.

---

# P1 — T3 재검정 (A100)

## 1. `teacher.py` 순서 수정

지금:  `load → fpipe.fit(전체) → season 필터 → labels`
바꿀 것: `load → season 필터 → fpipe.fit → labels`

`--max-season` 필터를 `fpipe.fit` **앞으로** 옮긴다. 필터 뒤 `sort_index()` 유지
(라벨 복원이 투수별 연속 투구 차분이라 경계가 어긋나면 안 된다).

## 2. 교사 재생성 — `T3_seq` 하나만

```bash
~/venv451/bin/python src/teacher.py --tag T3_seq --max-season 2023 --prev
```

`T3_self` 는 만들지 않는다 (self 계열 폐기).

## 3. 학생 6시드 — `DT3_seq`

```bash
~/venv451/bin/python src/train_gbdt2.py --model cat \
  --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 \
  --std-to-prior --std-season-prior --feat-domain --feat-skill-pc \
  --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 \
  --val-season 2023 --test-season 2024 --seeds 3,4,5,6,8,13 \
  --soft-target ./out/teacher_T3_seq.npz --tag DT3_seq
```

## 4. 판정

```bash
python tools/surf_report.py AB_base DT_seq DT3_seq
python tools/margin_surf.py AB_base DT_seq DT3_seq
```

**핵심 비교는 `DT3_seq` vs `DT_seq`** 다 (둘 다 A100, 같은 6시드).

```text
차이가 SE 안(|Δ| < 2×SE ≈ 4)   → caveat 종결. SETTLED 에 CLOSED 한 줄 추가
DT3 가 유의하게 낮음            → 남은 이득 = DT3 값. 그 값으로 다시 판단
DT3 가 AB_base +3 미만          → 증류 전체 폐기. P2 중단하고 Claude 에 즉시 보고
```

---

# P2 — 제출 구조 산출물 (4070, T3_seq 나온 직후 시작)

**P1 의 판정을 기다리지 않는다.** 교사 npz 만 나오면 바로 건다. P1 이 폐기 판정이면
그때 죽이면 된다.

## 왜 4070 인가

블렌드 멤버끼리는 **같은 머신**에서 나와야 가중이 안 치우친다. 기존 셀 멤버
`cat_ZD5` 의 2024 검증 예측이 `out/cat_ZD5_s*_val_preds.npz` 로 **4070 산**이다.
그래서 새 멤버와 base 재생성도 4070 에서 한다. A100 은 P1 전용.

```bash
scp hsu-server:~/aimers/out/teacher_T3_seq.npz ./out/     # 노트북 경유로 4070 에 전달
```

## 1. base 멤버 2024 검증 예측 재생성 — `VB_base`

`out/` 에 `cat_v14f_*_val_preds.npz` 가 **없다.** 가중을 다시 고르려면 필요하다.

```
--val-season 2024 (test-season 없음)  --seeds 42,7,13,3,4,5,6,8  --tag VB_base
```

플래그는 Best Configuration 그대로, `--soft-target` 없이.

> ⚠️ **`--tag v14f` 로 돌리지 말 것.** `model/cat_v14f_s*.pkl` 은 LB 1093.85 를 낸
> 실물이다. 덮어쓰면 되돌릴 수 없다. 반드시 `VB_base` 로.

## 2. 증류 멤버 — `DX_seq`

같은 명령에 `--soft-target ./out/teacher_T3_seq.npz --tag DX_seq` 만 추가.
시드도 같은 8개(42,7,13,3,4,5,6,8).

`--val-season 2024` 면 학습 구간이 ≤2023 이라 `T3_seq` 가 **전 행을 덮는다** —
soft target 결측 0 이어야 한다. 출력의 결측 경고 수치를 반드시 보고할 것.

## 3. 산출물

```text
out/cat_VB_base_s*_val_preds.npz   ← 가중 선택용
out/cat_DX_seq_s*_val_preds.npz    ← 가중 선택용
model/cat_DX_seq_s*.pkl            ← 제출 멤버 (≤2024 재학습본)
```

`.pkl` 은 `train_gbdt2.py:1159` 가 `refit-mult 1.5` 재학습본을 저장한다 =
제출 그대로다. 2024 행은 soft target 이 없어 실제 라벨로 되돌아가는데
(`train_gbdt2.py:1043`) 그게 맞다 — 보수적인 쪽이다.

---

# Must Keep Fixed

```text
피처·학습 플래그   Best Configuration 그대로. 바꾸는 건 --soft-target 하나뿐
P1 시드           3,4,5,6,8,13   (AB_base·DT_seq 와 동일)
P2 시드           42,7,13,3,4,5,6,8   (cat_v14f 제출본과 동일)
P1 은 A100, P2 는 4070. 섞지 말 것
판정 기준         t >= 2.4 채택 / 95% 상한 < +3 기각
```

---

# Important Constraints

- **로그를 grep 으로 거르지 말 것.** 1차 증류가 `CatBoostError: Target with classes
  must contain only 2 unique values` 로 죽었는데 grep 이 트레이스백을 삼켜서
  "완료"로 보였다. 전문을 `out/*.log` 에 남기고 끝나면 `Error|Traceback` 직접 확인.
- 실행 전 `python tools/precheck.py --file <스크립트>` 통과.
- A100 `setsid nohup ... & disown` / 4070 `bash tools/run4070.sh`.
- 한 머신에 한 작업. 시작 전 GPU 점유 확인.
- `failmode.py` 는 **train 전용**. 제출 zip 에 절대 넣지 않는다.
- `model/cat_v14f_*` · `model/cat_ZD5_*` 는 **읽기 전용으로 취급**. 제출 실물이다.

---

# Handoff Back To Claude

```text
Status
Changed Files
P1  DT3_seq : 앙상블 / 페어평균 / SE / t   (vs AB_base, vs DT_seq)
    margin  : D / rms / Dmax / margin / 최적w / 이득
    caveat 판정 (종결 / 축소 / 폐기)
P2  VB_base : 완료 시드 수, 2024 검증 BSS
    DX_seq  : 완료 시드 수, 2024 검증 BSS, **soft target 결측 수**
    저장된 pkl 목록
발생한 오류 전문
```

P3(가중 재선택 · 제출 zip · 스모크)은 **Claude 가 노트북에서** 한다. 제출 판단도
Claude 영역이다 — P2 산출물만 넘기고 제출은 건드리지 말 것.

---

# Context — 지금 어디까지 왔나

| | |
|---|---|
| 현재 LB | **1,093.85 (9위)** · 1위 1,126.33 |
| 제출 파일 | `submissions/blendv9_0808_0126.zip` (v14f 8시드 + ZD5 6시드, w 0.55) |
| 제출 잔여 | 오늘 4회 |
| 열린 축 | **증류(DT_seq +41.1, t=20.5)** 하나. std-k 40 은 제출 구조 검증 미실시 |

닫힌 축은 전부 `docs/SETTLED.md` 에 기전과 함께 있다. `precheck.py` 가 읽는다.

## 폐기된 것

- `DS_self` / `DS_seq` — 1차 증류. 교사가 2024 를 학습해서 누수. LEDGER 에 남은
  ≈1017~1020 행을 **비교군으로 쓰지 말 것**
- `DT_self` / `T2_self` — seq 와 같은 계열, NNLS 가중 0
