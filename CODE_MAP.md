# 코드 지도 — 어디부터 읽고 무엇을 무시할 것인가

파일이 90개지만 **살아있는 경로는 13개**다. 나머지는 시도했다가 접은 트랙이고,
지우지 않은 이유는 "이미 해봤다"는 기록이기 때문이다. **archive/ 는 읽지 말 것.**

---

## core/ — 이것만 이해하면 파이프라인 전체를 안다 (13개)

### 학습

| 파일 | 역할 |
|---|---|
| **train_gbdt2.py** (828줄) | 중앙 학습기. 모든 피처 플래그가 여기 모인다. 크지만 **읽어야 한다** |
| **season_std.py** (394줄) | **E99 핵심.** asof 통산 누적을 차분해 당해 시즌 성적 복원 + 야구 기전 교차항 |
| **target_enc.py** (166줄) | 시즌 expanding 타깃 인코딩 + `_dev`(순수 상호작용 성분) |
| **features.py** (304줄) | 피처 v2 — 카운트 상호작용, 실패 모드 구성비 등 |

읽는 순서: `season_std.add_std` → `target_enc.build_te` / `apply_dev` →
`train_gbdt2.main()` 의 피처 조립부(480~560줄 부근) → `run_cat`.

### 추론 · 제출

| 파일 | 역할 |
|---|---|
| **script_blend_v6.py** | **제출 zip 의 script.py 가 되는 파일.** 여기 WEIGHTS/SHIFT 를 바꾼다 |
| **make_submission.py** | script + model/*.pkl + requirements 를 zip 으로 묶는다 |
| pyproject.toml / requirements.txt | requirements.txt 는 **비어 있어야 한다** (서버 기본 패키지만 사용) |

⚠️ `script_blend_v6.py` 의 `predict_one` 은 학습 때 쓴 피처 생성을 **그대로 다시**
해야 한다. 학습에 플래그를 추가하면 **여기에도 반드시 추가**할 것. 안 하면
`ModuleNotFoundError` 나 컬럼 부재로 제출이 죽고 **제출 횟수가 차감된다.**

### 판정 · 검증 도구 (tools/)

| 파일 | 역할 |
|---|---|
| **arm_stats.py** | 스윕 결과를 시드 페어드로 판정. **원 BSS 와 편향제거 후를 둘 다** 낸다 |
| **group_blend.py** | 시드는 균등평균, 설정끼리만 greedy → WEIGHTS 출력 |
| **decide_shift.py** | 블렌드 편향 측정 → SHIFT 권고 |
| **verify_submission.py** | **제출 전 필수 게이트.** zip 구조·5행 완주·컬럼·추론시간 21항목 |
| **preflight.py** | 원격에 보내기 전 문법 검사 |
| **blend_value.py** | 새 모델을 **짓기 전에** 블렌드 기여를 계산 |
| **signal_audit.py** | 그룹별 오라클 천장 (투수 990.8 / 투수×카운트 2740.9) |

---

## wip/ — 진행 중, 아직 채택 안 됨 (3개)

| 파일 | 상태 |
|---|---|
| **train_nn.py** | NN 다양성. 현재 711점, 손익분기(826) 미달로 기여 0. 스윕 진행 중 |
| **link_pitchers.py** | 상황 튜플 공기 매칭으로 링키지 재구축. **실행 완료 — 기존과 99.9% 일치** |
| **build_tm_pitchmix.py** | `P(구종 \| 투수, 볼카운트)`. **작성만 하고 미실행.** 아래 §다음 할 일 |

---

## analysis/ — 판정 근거를 만든 스크립트 (5개)

결론은 문서에 다 적혀 있으니 **다시 돌릴 필요는 없다.** 근거를 의심할 때만 본다.

| 파일 | 무엇을 밝혔나 |
|---|---|
| domain_probe.py | 상황별 BSS 상한. 동일손 40.3 이 1위, 나머지는 <6 |
| count_intent.py | 카운트가 난이도가 아니라 **의도**를 바꾼다 (3볼에서 볼 성향 무력화) |
| abs_hetero.py | ABS 충격은 **균일**하다 → 세그먼트 SHIFT 근거 없음. 드리프트 −0.008~−0.012/시즌 |
| error_analysis.py | 캘리브레이션 십분위 + 세그먼트 편향 |
| column_relations.py | 시즌마다 부호가 뒤집히는 = 이전 안 되는 컬럼 목록 |

---

## archive/ — 읽지 말 것 (69개)

접은 트랙이다. 같은 걸 다시 시도하지 않도록 남겨둔 것뿐이다.

- `script_blend*.py`, `script_cat*.py`, `script_lgbm.py`, `script_tabm.py` — 제출
  스크립트 이전 버전 12개. **v6 만 유효**
- `tabm_reference.py`, `rtdl_num_embeddings.py`, `train_tabm.py`, `train_dtt.py`,
  `tokenize_v1.py`, `llm_probe.py` — LLM/DL 트랙. TabM v1 이 544 점에서 멈춤
- `pitcher_role.py`, `manager_feat.py`, `rules.py`, `career_gap.py`,
  `pitcher_cluster.py` — 역할/감독/규칙 모델. **LB 에서 유해했다** (838 로 하락)
- `build_tm_features.py`, `build_tm_consistency.py`, `build_tm_command.py`,
  `tm_context.py`, `tm_pitcher_ctx.py` — 트랙맨 집계 피처. 전부 기여 0
- 나머지 eval_*/eda*/bench_* — 일회성 조사

---

## 실행 순서 (처음부터)

```bash
uv sync
```

데이터는 Dacon 에서 받아 `data/` 에 둔다 (`train.csv` 368MB, `trackman_history.csv`
354MB, `test.csv` 5행 샘플, `sample_submission.csv`). **저작권상 이 묶음에는 없다.**

```bash
python src/train_gbdt2.py --model cat --tag v11f_s42 --seed 42 --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --lr 0.01 --es 500 --depth 8 --l2 10
```

시드만 바꿔 8개(42 7 13 3 4 5 6 8) 돌리면 현재 제출 구성이 재현된다.
A100 기준 모델당 약 3분. 그다음:

```bash
python tools/group_blend.py
```

```bash
PYTHONPATH=src python tools/decide_shift.py
```

출력의 `WEIGHTS`/`SHIFT` 를 `src/script_blend_v6.py` 에 반영하고:

```bash
python make_submission.py blendv6 && python tools/verify_submission.py submissions/<최신>.zip
```

**게이트를 통과하기 전에는 업로드하지 말 것.**

---

## 다음 할 일 (우선순위)

1. **NN 을 826점 이상으로** — 그러면 블렌드에 +14. 현재 711.
   추세가 "lr 낮을수록·배치 작을수록 좋다"로 명확하다 (622 → 711).
2. **`build_tm_pitchmix.py` 실행 + 병합 경로 확장** — 현재 `--tm-feats` 는
   (pitcher_id, season) 2키 병합이라 (pitcher_id, season, balls, strikes) 4키로
   확장해야 한다. train 에 없는 유일한 새 정보다.
3. `_delta` / 타자측 std 절제 실험, R/F 분리 모델

상세는 `HANDOFF.md` §6.
