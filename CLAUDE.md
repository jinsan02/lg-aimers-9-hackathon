# LG Aimers 9기 — 투구 제구 성공 확률 예측 AI 온라인 해커톤

> **에이전트 공용 규칙은 [AGENTS.md](AGENTS.md) 에 있다.** Claude 와 Codex 가 같이
> 쓰므로 운영 규칙(precheck·판정 기준·머신 규칙·제출 절차·과거에 치른 대가)은
> 그쪽 한 곳에서만 관리한다. 이 파일은 **대회 스펙**만 담는다.
>
> 현재 상태 → [EXPERIMENT.md](EXPERIMENT.md) · 인수인계 → [HANDOFF.md](HANDOFF.md)
> 닫힌 질문 → [docs/SETTLED.md](docs/SETTLED.md) · 실행 기록 → `LEDGER.tsv`

- 대회 페이지: https://dacon.io/competitions/official/236743
- 기간: 2026-08-05 시작 → **2026-09-01 제출 마감** (Phase 2) → 09-07 코드/PPT 제출 → 09-14 오프라인(Phase 3) 진출자 발표
- 주제: KBO 투구 1구 단위로, 투구가 이루어지기 **전까지 확인 가능한 정보만으로** 제구 성공 확률(0~1)을 예측

## 문제 정의

- Target: `control_success` (1 = 제구 성공, 0 = 실패)
- 제구 실패 정의: ① 스트라이크존 중앙, ② 스트라이크존 밖, ③ 포수 요구 반대 방향
- 학습 데이터: train.csv 1,475,092행 × 49컬럼 (2019~2024) / 평가 데이터: 2025 시즌 245,789행 (평가 서버에만, 배포본 test.csv는 5행 샘플 × 48컬럼)
- 입력 피처는 **test.csv 컬럼이 기준** (`row_id` 제외 47개; 범주형: `top_bottom`, `game_type`, `base_state`, `pitcher_hand`, `batter_hand` 등 코드형 포함). train.csv에만 있는 컬럼을 피처로 쓰면 추론 실패.
- 보조 데이터: `trackman_history.csv` — 2019~2024 과거 투구 로그 1,793,078행 × 30컬럼 (구종, 구속, 회전수, 무브먼트, 릴리스 정보 등). 베이스라인 미사용. train/test와 1:1 결합 불가 — 투수/타자 단위 요약 피처용.
- 전체 데이터 명세: [data/data_description.md](data/data_description.md)

### 데이터 주요 컬럼 그룹 (train/test)

- 경기 정보: `season`, `game_month`, `game_dayofweek`, `inning`, `top_bottom`, `game_type`
- 카운트/점수: `balls_before`, `strikes_before`, `outs_before`, `run_*`, `score_diff_*`
- 주자/중요도: `runner_on_1b/2b/3b`, `num_runners_on`, `base_state`, `home/away_win_expectancy`, `li`
- 선수/팀: `pitcher_id`, `batter_id`, `pitcher_hand`, `batter_hand`, `pitcher_team_id`, `batter_team_id`
- `asof_*` 이력 피처(19개): 투수/타자의 투구 직전까지 누적 성공률, 최근 1/3/5경기 성공률, 구종 비율 등 — 공식 제공이므로 사용 가능. 표본 0이면 결측(cold-start) → smoothing/fallback 설계 자유

### 사용 금지 정보 (실격 사유)

- 현재 투구 이후 확정 정보 (실제 코스/판정/구종/Trackman 측정값)
- 2025년 Trackman 데이터
- test.csv 내부 다른 행을 이용한 모든 피처 (누적/빈도/분포/rolling/target encoding/사후 보정)

## 평가 지표 — Brier Skill Score

```
brier = mean((pred - y)^2)
baseline_brier = r * (1 - r)          # r = 실제 성공률 (비공개, 상수 예측의 Brier)
score = max(0, 100000 * (1 - brier / baseline_brier))   # 높을수록 좋음
```

- 확률 캘리브레이션이 점수에 직결됨 (Brier는 calibration에 민감). 0/1로 반올림 금지.
- **추론 원칙: 평가 데이터의 각 행을 독립적으로 예측해야 함. test 전체 분포를 이용한 후처리/보정 금지.**
- **Public Score = 전체 테스트 데이터 100%. Private Score = 대회 종료 시점의 Public Score** (별도 hidden split 없음 → Public 과적합 이슈보다 순수 일반화 성능이 중요).
- 1차 평가: Private Score 100% → 코드/PPT 검증 통과한 상위 ~100명이 Phase 3 진출.
- **LG Aimers 수료 조건: Public Score ≥ 549.51** (운영진이 베이스라인을 평가 환경에서 실행한 점수). 로컬 2024 홀드아웃 415.57과 갭이 있음 — LB의 r과 데이터 난이도가 다르기 때문. 로컬 점수와 LB 점수는 절대값 비교 불가, 상대 개선만 신뢰할 것.

## 규칙 / 제한조건

| 항목 | 내용 |
|---|---|
| 언어 | Python만 허용 |
| 외부 데이터 | **금지** (공식 제공 데이터만) |
| 사전학습 모델 | 공개 가중치 + 비상업 이상 라이선스(MIT, Apache 2.0 등)만 허용 |
| 원격 API | 금지 (OpenAI, Gemini 등) |
| 제출 횟수 | 1일 최대 5회 |
| 팀 인원 | 최대 5명 |
| 재현성 | 로컬 환경에서 코드로 재현 가능해야 함 (random_state 고정 필수) |

### 평가 서버 사양 (공식 공지 확인됨)

| 항목 | 사양 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU | **NVIDIA L4 (VRAM 22.4GiB), CUDA 12.8** — GPU 추론 가능 |
| CPU / RAM | 6 vCPU / 28GB |
| Python | **3.11.15** (로컬 WSL 환경과 동일) |
| 인터넷 | ❌ 차단 (패키지 설치 외 외부 다운로드 불가 — HuggingFace 등 런타임 다운로드 코드 작동 안 함) |

### 평가 서버 제약

| 항목 | 제한 |
|---|---|
| 추론 시간 | ≤ 10분 (245,789행) |
| 패키지 설치 시간 | ≤ 10분 |
| 제출 zip 용량 | ≤ 10GB (압축 해제 후 32GB) |

- 서버는 zip을 풀고 `script.py`를 실행, `./output/submission.csv`를 채점. `data/`는 **읽기 전용**.
- 서버가 `./data/`에 실제 평가 데이터를 주입 (로컬 배포본 test.csv는 5행 샘플).
- **오류 구분**: ① 설치 오류(zip 구조 불일치, 패키지 설치 실패) → 제출 횟수 차감 없음 / ② 제출 오류(script.py 실행 중 오류) → **제출 횟수 차감됨**. script.py는 제출 전 반드시 로컬 완주 확인.
- zip 최상위에 추가 폴더가 있으면 설치 오류 — `script.py`가 zip 루트에 바로 있어야 함.

### 평가 서버 기본 설치 패키지 (버전 고정 — 이 버전과 다르게 지정하면 설치 오류 위험)

```
torch==2.7.1+cu128  pandas==2.0.3  numpy==1.26.4  scipy==1.15.3
scikit-learn==1.8.0  joblib==1.5.3  threadpoolctl==3.6.0  narwhals==2.21.2
transformers==4.46.3  accelerate==1.9.0  sentencepiece  regex  tqdm  loguru  pyyaml  rich
```

- **전략: 기본 설치 패키지만 사용하고 requirements.txt는 비워둔다** (설치 시간 0, 설치 오류 리스크 0).
- 로컬 학습 환경은 서버와 동일 버전으로 고정: `pandas==2.0.3`, `numpy==1.26.4`, `scikit-learn==1.8.0`, `joblib==1.5.3` (pkl 호환성 때문에 sklearn 버전 일치가 특히 중요).
- LightGBM/XGBoost/CatBoost는 기본 목록에 없음 → 사용 시 requirements.txt에 추가해야 하며 설치 10분 내 완료 확인 필요.

## 제출 zip 구조

```
submit.zip
├── model/            # 학습된 모델 파일 (예: rf.pkl)
├── script.py         # 추론 코드 — 서버가 실행
└── requirements.txt  # 버전 고정 필수 (설치 10분 제한)
```

- 제출 파일은 `sample_submission.csv`와 같은 `row_id` 순서/컬럼이어야 함.
- 파이프라인 밖에서 만든 피처가 있다면 그 생성 코드가 `script.py`에도 반드시 포함되어야 함.

## 디렉토리 구조

```
C:\aimers\
├── CLAUDE.md
├── data\                  # ★ Dacon에서 다운로드한 데이터 배치 (git 제외 대상)
│   ├── train.csv          # 2019~2024, target 포함
│   ├── test.csv           # 5행 샘플 (실제 평가 데이터는 서버에)
│   ├── sample_submission.csv
│   └── trackman_history.csv   # 179만 행 과거 로그 (보조)
├── notebooks\             # 베이스라인 노트북 (참고용)
├── src\
│   ├── train.py           # 학습 → model/rf.pkl 저장 + 2024 홀드아웃 검증 점수 출력
│   └── script.py          # 추론 (제출 대상, 로컬에서도 동일 경로로 실행 가능)
├── model\                 # 학습된 모델 (train.py 산출물)
├── output\                # submission.csv (script.py 산출물)
├── submissions\           # 제출용 zip 보관 (버전별)
├── make_submission.py     # model/ + script.py + requirements.txt → submissions/*.zip
└── requirements.txt
```

## 컴퓨팅 자원

| 머신 | 접속 | 사양 | 용도 |
|---|---|---|---|
| 로컬 (이 PC) | - | RTX 5060 Laptop 8GB, Windows | 개발/실험 |
| desktop-4070 | `ssh desktop-4070` | RTX 4070 Ti SUPER 16GB | GPU 학습 |
| hsu-server | `ssh hsu-server` (desktop-4070 경유 ProxyJump) | **A100 40GB**, RAM 503GB, 80코어 | 대규모 학습/튜닝 |

- 평가 서버에도 L4 GPU(22.4GB)가 있어 GPU 추론 가능. 단 torch 기반일 때 얘기고, sklearn/GBDT 계열은 CPU(6 vCPU) 추론 10분 내 완주 확인 필요.

## 실행 방법 — Windows 네이티브 권장 (2026-08-05 이후)

**Smart App Control이 꺼졌다** (`VerifiedAndReputablePolicyState = 0`). 이전에는 서명되지 않은
.pyd DLL이 차단되어 WSL만 가능했으나, 이제 **Windows 네이티브 실행이 정상 작동**한다.

```bash
uv sync
```

```bash
uv run python src/train_gbdt2.py --model cat --tag x --l2 3 --feat-v2
```

- 네이티브 venv: `C:\aimers\.venv` (uv 관리). 인용부호 문제가 없어 WSL보다 편하다.
- PYTHONPATH가 필요한 스크립트는 `$env:PYTHONPATH="src"` 를 먼저 설정.

### WSL 경로 (여전히 사용 가능, 레거시)

- venv: WSL 내부 `~/.venvs/aimers` / 프로젝트: `/mnt/c/aimers`

```bash
wsl -d Ubuntu-22.04 -e bash -lc 'cd /mnt/c/aimers && export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/aimers && ~/.local/bin/uv sync'
```

```bash
wsl -d Ubuntu-22.04 -e bash -lc 'cd /mnt/c/aimers && ~/.venvs/aimers/bin/python src/train.py'
```

```bash
wsl -d Ubuntu-22.04 -e bash -lc 'cd /mnt/c/aimers && ~/.venvs/aimers/bin/python src/script.py && ~/.venvs/aimers/bin/python make_submission.py'
```

※ PowerShell에서 위 명령을 만들 때 큰따옴표를 쓰면 `$HOME`이 Windows 경로로 확장되어
경로가 깨진다 — 반드시 작은따옴표로 감쌀 것.

### 패키지 버전 — 평가 서버 기본 설치 버전과 동일하게 고정

로컬(pyproject.toml): `pandas==2.0.3`, `numpy==1.26.4`, `scikit-learn==1.8.0`, `joblib==1.5.3`
— 평가 서버 기본 설치 버전과 완전 일치. **requirements.txt는 비워둠** (기본 패키지만 사용 시 설치
단계 자체를 스킵 → 설치 오류 리스크 0). 새 라이브러리 도입 시에만 requirements.txt에 추가하되,
서버 기본 패키지의 버전은 절대 requirements.txt에 다시 적지 말 것 (버전 충돌 시 설치 오류).
모델 pkl은 학습 환경과 서버의 sklearn 버전이 일치해야 안전하게 로드됨.

## EDA 주요 발견 (2026-08-05, src/eda.py)

1. **강한 시즌 drift**: 제구 성공률이 2019 0.5647 → 2024 0.4861로 단조 하락. 평가(2025)는 더 낮을 가능성. `season`을 피처로 쓰면 2025는 외삽 — 트리는 2024 리프로 처리하므로 동작은 하지만, 최근 시즌 가중치/드리프트 보정 실험 가치 높음. asof 피처들도 같은 방향으로 drift.
2. **trackman ID는 train과 직접 조인 불가**: `pitcher_trackman_id`(50008~)와 `pitcher_id`(20700~) 교집합 0. 팀/날짜/상황 기반 레코드 링키지를 만들어야 구속·무브먼트 피처 활용 가능 (고난도, 별도 트랙).
3. **cold-start**: 2024 투수 391명 중 신규 81명(행 기준 약 20%). prev1/3/5경기 피처 결측 ~2%.
4. asof 피처 상관 최대 0.084 (`asof_pitcher_success_rate`) — 신호가 약한 문제. Brier 개선 폭은 작은 단위로 쌓아야 함.
5. 클래스 균형 양호(52:48), 결측 거의 없음, row_id 중복 없음. `game_type` R(정규)/F(퓨처스?) 2종 — F가 11%.

## 모델 실현 가능성 (제출 조건 기준, src/bench_inference.py 실측)

**추론 시간은 병목이 아님**: RF(100트리) 기준 245,789행 예측 0.4초 (6코어 제한 실측). 10분
한도의 0.1%. 병목은 ① 추가 패키지 설치 시간 ② 인터넷 차단(런타임 다운로드 불가) ③ sklearn 등
기본 패키지 버전 충돌.

| 모델 | 판정 | 근거 |
|---|---|---|
| sklearn 계열 (RF, HistGB) | ✅ 즉시 가능 | 기본 설치, 추론 초 단위 |
| LightGBM / XGBoost / CatBoost | ✅ 가능 | requirements에 추가 (wheel 설치 1~2분). 대형 앙상블(수십 모델)도 추론 수 분 |
| Torch NN (MLP, TabM, FT-Transformer) | ✅ 가능 | torch 2.7.1+cu128 기본 설치, L4 22.4GB. 가중치는 zip에 동봉 (런타임 다운로드 금지) |
| 시드×폴드 대형 앙상블 (GBDT+NN 혼합) | ✅ 가능 | 추론 예산 충분. zip 10GB 한도만 주의 |
| AutoGluon | ⚠️ 비추천 | 의존성 대량 설치 → 기본 패키지 버전 충돌/설치 10분 초과 위험 |
| TabPFN | ⚠️ 조건부 | 147만 행은 컨텍스트 초과 → 서브샘플 앙상블만 가능. 가중치 동봉 + 라이선스 확인 필요 |
| 원격 API (OpenAI 등) | ❌ 규칙 금지 | |

## 컴퓨팅 역할 분담

| 머신 | 담당 작업 |
|---|---|
| **노트북** (RTX 5060 8GB, WSL RAM 16GB) | EDA, 피처 엔지니어링 프로토타입, LightGBM 소규모 실험, 제출 패키징/스모크 테스트 |
| **desktop-4070** (4070 Ti SUPER 16GB) | 중형: XGBoost/CatBoost GPU 학습, MLP/TabM 중형 NN, Optuna 수십 trial 튜닝 |
| **hsu-server** (A100 40GB, 80코어, RAM 503GB) | 대형: 대규모 Optuna 스윕(수백 trial), FT-Transformer/TabM 대형, 다중 시드×폴드 앙상블 일괄 학습, trackman 레코드 링키지(80코어 CPU 병렬) |

## 로드맵

1. ✅ 베이스라인 파이프라인 + 첫 제출 zip (415.57 로컬)
2. ✅ EDA + 추론 벤치마크
2-1. ✅ TabM v1 (2026-08-05): A100 학습, **2024 검증 BSS 544.24** (RF +31%).
   파이프라인: `src/prep_v1.py`(노트북, sklearn 1.8.0 전처리) → `src/train_tabm.py`(A100
   venv451) → `src/script_tabm.py`(제출 추론) → `make_submission.py tabm`.
   에폭당 10초, early stop. 개선 여지: lr 2e-3가 커서 검증 점수 진폭 큼 → v2에서 lr 낮추고
   스케줄러/앙상블(시드 평균). 체크포인트에 numpy 스칼라 넣지 말 것(weights_only=True 로드 실패).
3. [노트북] 검증 체계 강화: 2024 홀드아웃 유지 + 2023 보조 홀드아웃으로 drift 민감도 확인
4. [노트북] 피처 v1: 카운트 상황 상호작용, 매치업(투수손×타자손), cold-start smoothing, season 처리 실험
5. [노트북→4070] LightGBM 전환 + 튜닝 → 두 번째 제출
6. [4070] XGBoost/CatBoost GPU + 확률 캘리브레이션 실험
7. [A100] 대규모 튜닝 스윕 + 시드 앙상블
8. [A100] NN 실험 — 선별 완료: TabM 주력, RealMLP/FT-Transformer 보조, TabICL 조건부, TabPFN 제외. 상세: [docs/dl_models.md](docs/dl_models.md). 추론은 state_dict + vendoring 코드로 순수 torch 이식(requirements 빈 상태 유지)
9. [리서치, 병렬] trackman 레코드 링키지 → 성공 시 구속/무브먼트 요약 피처
10. [노트북] 최종 앙상블 + 제출 관리 (마감 09-01)

## 작업 원칙

- 검증은 시즌 기반 홀드아웃(2024 검증)이 기본. 시간 순서가 있는 데이터이므로 무작위 KFold보다 시즌/시간 분할을 우선.
- 제출 전 체크리스트: ① script.py가 `./data`, `./model`, `./output` 상대경로만 사용하는지 ② requirements.txt 버전 고정 ③ 로컬에서 5행 샘플로 script.py 완주 확인 ④ 추론 10분 제한 고려 (모델 크기/복잡도).
- 1일 5회 제출 제한 → 로컬 검증 점수로 충분히 걸러낸 뒤 제출.
- 개선 방향 후보: trackman_history 기반 투수/타자/매치업 누적 피처, GBDT(LightGBM/XGBoost/CatBoost), 확률 캘리브레이션(isotonic/Platt — 단, train 데이터로만), 시즌 가중치.
