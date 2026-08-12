# 현행 챔피언 v11 — 파트별 레시피

- **LB `1101.8020672065`** · 제출 파일 `submissions/v11_pb_posix_0809.zip` (102.8 MB, 23항목)
- 작성 2026-08-13. **출처는 제출 zip 안의 실물**이지 작업 트리 파일이 아니다.
  다만 이번에 대조한 결과 `script.py` 와 `src/script_blend_v11.py` 는
  sha256 `da13e45361be9894…` 로 **동일**했다.

> 읽는 법: 파트 1~5 가 점수를 만드는 실체다. 파트 6 은 **이 레시피에서 근거가
> 약한 곳**을 표시한 것이다 — 재현하거나 손볼 때 6부터 보는 게 빠르다.

---

## 파트 1 — 멤버 (모델 14개, 2계열)

두 계열은 **출력 기하가 다르다.** 그래서 섞인다. 같은 121피처, 같은 데이터다.

### 1-A. `cat_v14f` — 이진 (8시드)

```text
loss_function  Logloss
depth          8
learning_rate  0.01
l2_leaf_reg    10
border_count   254
features       121
시드            42, 7, 13, 3, 4, 5, 6, 8
```

| 시드 | best_iter | 최종 트리 | val BSS |
|---|---:|---:|---:|
| 3 | 1616 | 2424 | 923.83 |
| 4 | 1563 | 2344 | 918.25 |
| 5 | 1433 | 2149 | 929.59 |
| 6 | 1283 | 1924 | 920.47 |
| 7 | 1479 | 2218 | 921.98 |
| 8 | 1621 | 2431 | 914.25 |
| 13 | 1671 | 2506 | 923.62 |
| 42 | 1439 | 2158 | 916.24 |

조기종료가 정상 작동했다(1283~1671, 상한 3000 에 안 닿음).

### 1-B. `cat_ZD5` — 실패모드 셀 다중분류 (6시드 탑재)

타깃을 셀 안에 넣어 `P(성공) = Σ(성공비트=1인 셀)` 이 **정확히** 성립한다.

```text
loss_function  MultiClass  (셀 14개)
depth          5
learning_rate  0.01
l2_leaf_reg    10
border_count   254
features       121
fm_success     [9, 10, 11, 12, 13]      ← 이 인덱스들의 확률을 더해 P(성공)
시드            42, 7, 13, 3, 4, 5       (pkl 은 6, 8 도 있으나 **탑재 안 됨**)
```

| 시드 | best_iter | 최종 트리 | val BSS |
|---|---:|---:|---:|
| 3 | 2989 | 4483 | 917.88 |
| 4 | 2995 | 4492 | 922.62 |
| 5 | 2997 | 4495 | 921.92 |
| 13 | 2999 | 4498 | 923.75 |
| 42 | 2999 | 4498 | 922.14 |
| 7 | 2998 | 4497 | 921.05 |

⚠ **best_iter 가 전부 3000 상한에 붙어 있다.** 조기종료가 아니라 iteration 이
바닥난 것이다. 상한을 5000 으로 올려본 결과는 `failmode-cell-iters-5000`
(미학습 셀 −8.67) 로 닫혔으니 **상한이 전이 정규화 역할**을 하고 있다.

---

## 파트 2 — 피처 파이프라인 (121개)

`fpipe` 단계 순서: `tm → v2 → std → te → skill`. 두 멤버가 동일하다.

```text
--feat-v2                                  원본 45열 + v2/v3 파생 28개
--feat-std --std-k 80
  --std-to-prior --std-season-prior        시즌내 성적 복원 23개
--te p,pc,ph,b,pi --te-k 50 --te-dev       타깃 인코딩 18개
--feat-domain                              도메인 교차 5개
--feat-skill-pc                            선형 실력 추정 2개 (skill pack 1)
--lr 0.01 --es 500 --depth 8|5 --l2 10 --refit-mult 1.5
```

**꺼져 있는 것**: `count` · `form` · `window` · `prof` · `tm-feats` · `cross`
(전부 `docs/SETTLED.md` 에서 닫힌 축)

### TE 축 5개 (실물에서 추출)

| 코드 | 키 |
|---|---|
| `p` | `pitcher_id` |
| `pc` | `pitcher_id × balls_before × strikes_before` |
| `ph` | `pitcher_id × batter_hand` |
| `b` | `batter_id` |
| `pi` | `pitcher_id × inning_bucket` |

`te.dev = True` (편차 3개 추가). 시즌 expanding + `shift(1)` 이라 행 단위 누수 없음.

### `std_season_prior` — 시즌별 사전확률 (pkl 에 저장된 실물)

```text
2020 .559821   2021 .547245   2022 .535459
2023 .535402   2024 .523730   2025 .511397   2026 .511397(= 2025 반복)
```

**2025 값 .511397 이 제출 시점에 이미 박혀 있다.** 드리프트 외삽이며 이게
`season` 피처를 못 빼는 이유와 같은 계열이다.

### 학습 데이터 지문

```text
fpipe['priors']['asof_pitcher_success_rate'] = 0.5401750413   (두 멤버 동일)
```

→ **`--drop-f-pre` 를 쓰지 않는다.** 1,475,092행 전체로 refit 됐다.
판정 표면은 이 플래그를 *요구*하므로 두 표면의 명령을 섞으면 안 된다
(`docs/SETTLED.md` `drop-f-pre-omitted`).

---

## 파트 3 — 블렌드 가중

```python
_W_CELL = 0.55
_F = [42, 7, 13, 3, 4, 5, 6, 8]     # base 8시드
_C = [42, 7, 13, 3, 4, 5]           # cell 6시드
```

| 계열 | 총가중 | 시드당 |
|---|---:|---:|
| base `cat_v14f` | 0.45 | 0.05625 |
| cell `cat_ZD5` | 0.55 | 0.09166667 |

**확률 공간 평균**이다(로짓 평균 아님 — `logit-blend` 는 centered +0.021 로 닫힘).

---

## 파트 4 — 후처리 (순서가 곧 정의다)

```text
① 14멤버 확률 가중평균
② SLOPE  p ← sigmoid( 1.0416 × logit(clip(p, 1e-6, 1-1e-6)) )
③ SHIFT  p ← clip(p − 0.0052, 0, 1)
④ + recent-middle 오프셋      (행 자신의 컬럼 룩업)
⑤ + pitcher×batter 오프셋     (행 자신의 두 ID 룩업)
⑥ clip(p, 0, 1)
```

### ④ recent-middle — `asof_pitcher_prev5_game_middle_rate` 8분위

스크립트에 **인라인 상수**로 박혀 있다(파일 참조 아님).

| 구간 | 경계 | 오프셋 |
|---|---|---:|
| 1 | ~ .114286 | **+0.00912860** |
| 2 | ~ .141026 | +0.00145823 |
| 3 | ~ .159091 | +0.00166458 |
| 4 | ~ .174312 | +0.00313052 |
| 5 | ~ .188889 | +0.00028883 |
| 6 | ~ .203837 | −0.00547501 |
| 7 | ~ .227273 | −0.00413855 |
| 8 | .227273 ~ | −0.00507436 |
| 결측 | — | **−0.00810269** |

⚠ **단조가 아니다** (구간 1→2 에서 +0.0091 → +0.0015 로 급락, 5→6 부호 전환).
career-middle 로 바꾼 v12 가 LB −18.271 로 무너진 축이 정확히 이 룩업이다.

### ⑤ pitcher×batter — `model/matchup_constants_2024.npz`

```text
사용하는 배열   pb0_pitcher / pb0_batter / pb0_offset
쌍 개수        26,355
오프셋 범위     −0.018069 ~ +0.021082   (평균 +0.0000120)
미등록 쌍       0.0
```

⚠ npz 에는 **쓰이지 않는 배열이 더 있다** — `pb_*`(26,355) 와 `bo_*`(2,461, 타순
추정). 스크립트는 `pb0_*` 만 읽는다. 남은 둘은 사표다.

### 행 독립성

④⑤ 모두 **그 행 자신의 컬럼**과 train 에서 동결한 표만 쓴다. 다른 test 행을
보지 않으므로 규칙 위반이 아니다(`tools/audit_rowindep.py` 통과 이력 있음).

---

## 파트 5 — 제출 패키지

```text
v11_pb_posix_0809.zip  (102.8 MB, 23항목)
├── script.py                      ← 서버가 실행. zip 루트여야 한다
├── requirements.txt               "catboost==1.2.10\n"  (이 한 줄만)
├── fpipe.py  features.py  season_std.py  skill.py  target_enc.py
└── model/
    ├── cat_v14f_s{42,7,13,3,4,5,6,8}.pkl      8개
    ├── cat_ZD5_s{42,7,13,3,4,5}.pkl           6개
    └── matchup_constants_2024.npz
```

- **`failmode.py` 없음** — 확인했다. test 에 쓰면 2025 타깃이 96.79% 복원되므로
  train 전용이고 절대 탑재하지 않는다.
- 추론 실측 **33.7초 / 600초** (6 vCPU, 245,789행). 병목 아님.
- 출력 `./output/submission.csv`, `sample_submission` 의 row_id 순서에 맞춰 매핑.

---

## 파트 6 — 이 레시피에서 근거가 약한 곳

재현하거나 손보기 전에 알아야 할 것. **점수를 부정하는 게 아니라 어디를 믿으면
안 되는지**의 목록이다.

| 항목 | 상태 |
|---|---|
| `v14f`·`ZD5` 학습 명령 전문 | **기록 없음.** LEDGER 이전에 만들어졌다. 위 하이퍼는 pkl 에서 역추출한 것 |
| `SHIFT .0052` · `SLOPE 1.0416` | 당시 설명은 지금 독립 근거로 쓰기 어렵다. 제거가 개선이라는 뜻도 아니다 |
| `_W_CELL = .55` | source-only 선택값이 아니다. 재유도 전까지 legacy 비교 상수로만 취급 |
| ④ recent-middle 룩업 | 비단조. v12 가 이 축을 바꿔 LB −18.271 |
| npz 경로 | 생성기는 `out/` 에 쓰고 스크립트는 `model/` 에서 읽는다 — 패키징 때 수동 이동이 끼어 있다 |
| refit 트리 수 | 이 pkl 들은 `int(best_iter × 1.5)` 로 만들어졌다. `best_iter` 는 0-based 이므로 정확히는 `int((best_iter+1) × 1.5)` — 1~2그루 적다. 2026-08-13 에 코드는 고쳤으나 **이 모델들은 옛 공식 산물**이다 |
| 셀 감독 라벨 | 이 pkl 들은 **전역 shift 버그가 있던 `failmode.py`** 로 학습됐다(셀이 달라지는 행 3.87%). 수정 후 재학습본이 아니다 → `LEGACY_BUGGED_LABELS` |

**재현 목표는 byte 일치가 아니다.** 같은 소스·같은 데이터로 다시 구우면
run-to-run 변동 범위 안에서 재현되는 것으로 본다. 새 lineage 는
`out/lineage_<tag>.json` 에 commit·host·행수·피처 지문·셀 taxonomy 지문을 남긴다.

---

## 한 장 요약

```text
14 모델
  = CatBoost 이진 depth8  × 8시드  (가중 0.45)
  + CatBoost 셀 다중분류 depth5 × 6시드  (가중 0.55)
공통 121피처 = v2파생 + 시즌내복원(k80) + TE 5축(dev) + domain + skill-pc
학습        lr .01 / l2 10 / border 254 / es 500 / refit ×1.5 / drop-f-pre 없음
후처리      로짓기울기 ×1.0416 → −0.0052 → recent-middle 8분위 → PB 26,355쌍
결과        LB 1101.802
```
