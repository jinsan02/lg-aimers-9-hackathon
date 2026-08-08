# HANDOFF.md

먼저 읽을 것: [AGENTS.md](AGENTS.md) → [EXPERIMENT.md](EXPERIMENT.md) → 이 파일

## Current Agent

Claude

## Next Agent

Codex

## Status

`WAITING_FOR_CODEX` — **두 GPU 모두 비어 있다.** 1차 증류(`DS_self`/`DS_seq`)는
15:44 에 6시드 다 끝났지만 **교사 누수라 수치를 쓰지 않는다.** LEDGER 에 남아 있는
`DS_*` 행(미학습 2024 ≈1017~1020)은 비교군으로 쓰지 말 것 — 아래 E165b 로 다시 잰다.

---

# Current Task

**E165b — 누수 없는 증류 재검정**

1차 증류가 미학습 2024 에서 **+142.87 (t=76.11)** 로 나왔는데 **누수다.**
교사 `T_self` 를 랜덤 4-fold CV 로 **전체 train(2024 포함)** 에 돌렸다. 학생은
≤2023 만 학습하지만 그 부드러운 타깃이 2024 를 본 교사에서 나왔고, 평가 시즌이
바로 2024 다. 실제 제출에는 2025 데이터가 아예 없으므로 이 이득은 존재하지 않는다.

확인 근거 — 교사가 2024 를 시즌 수준까지 맞히고 있다:

```
season   교사 예측평균   실제       차이
2023       0.498628   0.499957   -0.001330
2024       0.487632   0.486105   +0.001527
```

---

# What To Do

## 1. `src/teacher.py` 에 `--max-season` 추가

지정 시즌 **초과 행을 CV 이전에 제거**한다. 위치는 `load()` + `fpipe.fit()` 직후,
`_pitch_labels` 호출 **전**. (라벨 복원은 투수별 연속 투구 차분이라 행을 지운 뒤에
해야 경계가 어긋나지 않는다 — 지우고 나서 `sort_index()` 유지할 것.)

저장 npz 는 그대로 `row_id`, `prob` 두 배열. 학생이 못 찾는 row_id 는
`train_gbdt2.py` 가 이미 원래 라벨로 되돌린다(결측 경고를 출력하므로 값을 볼 것).

## 2. 교사 두 개 재생성

```bash
python src/teacher.py --tag T2_self --max-season 2023
python src/teacher.py --tag T2_seq  --max-season 2023 --prev
```

## 3. 학생 재검정 (A100, 6시드)

```bash
python src/train_gbdt2.py --model cat \
  --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 \
  --std-to-prior --std-season-prior --feat-domain --feat-skill-pc \
  --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 \
  --val-season 2023 --test-season 2024 --seeds 3,4,5,6,8,13 \
  --soft-target ./out/teacher_T2_self.npz --tag DT_self
```

`T2_seq` 로 같은 것을 `--tag DT_seq` 로 한 번 더.

## 4. 판정

```bash
python tools/surf_report.py AB_base DT_self DT_seq DW_cell
python tools/margin_surf.py AB_base DT_self DT_seq DW_cell
```

---

# Must Keep Fixed

```text
피처·학습 플래그   위 명령 그대로 (Best Configuration 과 동일)
기준선            AB_base 883.41  ← A100 산. 4070 결과와 섞지 말 것
시드              3,4,5,6,8,13
판정 기준         t >= 2.4 채택 / 95% 상한 < +3 기각
```

---

# Important Constraints

- **로그를 grep 으로 거르지 말 것.** 1차 증류가 재학습 단계에서
  `CatBoostError: Target with classes must contain only 2 unique values` 로 죽었는데,
  `2>&1 | grep -E "^\[cat|..."` 가 트레이스백을 삼켜서 "완료"로 보였다.
  전문을 `out/*.log` 에 남기고 끝나면 `Error|Traceback` 을 직접 확인할 것.
- 실행 전 `python tools/precheck.py --file <스크립트>` 통과.
- A100 은 `setsid nohup ... & disown`, 4070 은 `bash tools/run4070.sh`.
- 한 머신에 한 작업. GPU 점유 먼저 확인.
- `failmode.py` 는 **train 전용**. 제출 zip 에 절대 넣지 않는다.

---

# Expected Output

```text
1. 변경된 파일
2. 실행한 명령 (전문)
3. DT_self / DT_seq 의 미학습 2024 앙상블·페어평균·SE·t
4. margin (D, rms, Dmax, margin, 최적w, 이득)
5. AB_base 883.41 및 DW_cell(+17.76) 대비 위치
6. 발생한 오류 전문
```

---

# Handoff Back To Claude

아래를 채워서 돌려준다.

```text
Status
Changed Files
DT_self  : 앙상블 / 페어평균 / t / margin
DT_seq   : 앙상블 / 페어평균 / t / margin
교사 OOF : T2_self / T2_seq (그리고 1차 T_self 2076.0 / T_seq 2115.1 과 비교)
Conclusion
Next Recommendation
```

---

# Context — 지금 어디까지 왔나

| | |
|---|---|
| 현재 LB | **1,093.85 (9위)** · 1위 1,126.33 · 8위와 +2.8 |
| 제출 잔여 | 오늘 4회 |
| 후처리 여지 | **총 +13.1** (SLOPE 가 이미 +3) — 상수 튜닝은 끝났다 |
| 열린 축 | 증류 하나. std-k 40 은 제출 구조 검증 미실시 |

08-07~08 에 기각된 것: Optuna · GPBoost · trackman · 세그먼트 드리프트 · LightGBM ·
피처 부분공간 · refit 2.0(18시드) · te-k 축별 · 셀×카운트 맥락 · 2스트라이크 라우팅 ·
F리그 분리 · isotonic · 시드 확장. 전부 `docs/SETTLED.md` 에 기전과 함께 있다.
