# 실험 운영 가이드 — 멀티 머신 병렬 실험 기준

> 목적: 노트북·4070·A100에서 실험을 병렬로 돌려도 결과가 비교 가능하고 기록이 안 새게 하는 규칙.
> 결과 기록은 전부 [docs/EXPERIMENTS_LOG.md](../docs/EXPERIMENTS_LOG.md) 한 곳에.

## 1. 머신별 역할과 실행 환경

> ⚠️ **이 절은 [AGENTS.md](../AGENTS.md) §3-1 로 옮겼다** (2026-08-08 실측으로 갱신).
> 여기 있던 표는 낡았다 — 노트북 WSL 강제·4070 `python -m uv run`·4070 WSL venv211 은
> 전부 지금 사실이 아니다. 접속·셸·경로·지속 실행·환경 차이는 AGENTS.md 를 볼 것.

요약만 남긴다:

| 머신 | 역할 |
|---|---|
| 노트북 | 분석·판정·제출 패키징 (발열 때문에 학습 안 함) |
| desktop-4070 | CatBoost GPU 학습 (cmd.exe 셸) |
| hsu-server (A100) | 대규모 학습·스윕 (bash, 4070 경유 ProxyJump) |

- 원본은 노트북 `C:\aimers`. 원격에서 편집하지 않는다.
- 전처리 산출물(`data/processed/*.npz`, `model/prep_*.pkl`)은 **노트북(sklearn 1.8.0)에서 생성** — 평가 서버 pkl 호환성.

## 0. 실험 전 필수 — precheck (2026-08-08 신설)

```bash
python tools/precheck.py <train_gbdt2.py 에 줄 플래그 전부>
python tools/precheck.py --file some.sh      # 스크립트 통째로
```

종료코드 2 면 **금지된 실험**이다. 그냥 돌리지 말 것.

- 08-07 에 30개 실험을 돌리고 이 문서의 기록을 하나도 안 갱신했다. 그래서
  E08(season 제거 −580)을 "안 해본 축"이라 부르며 다시 큐에 걸었다.
- 이제 기록은 `train_gbdt2.py` 가 **자동으로** `LEDGER.tsv` 에 쌓는다(시드마다 1행).
  precheck 이 그걸 읽어 같은 플래그 조합의 과거 결과를 보여준다.
- 닫힌 질문의 판정은 [SETTLED.md](SETTLED.md) 에 모았다. 새로 닫으면 여기에 한 줄 추가.
- precheck 은 **제출 설정과 다른 플래그**도 알려준다. 다르면 그 실행에서 잰
  후처리 상수를 제출에 쓰면 안 된다 (v16 −6.15 가 이 실수였다).

## 2. 실험 프로토콜 (모든 머신 공통)

1. **판정 표면**: `--val-season S-1 --test-season S` (배치 구조 = ≤S-1 재학습 → 미학습 S).
   자기검증 2024 홀드아웃은 **참고용**이다 — 셀 멤버의 D 가 두 표면에서 +2.8 vs −18.0
   으로 갈렸고, refit 배수 이득(+16.3)은 자기검증 표면에 **원리적으로 안 나타난다.**
   - 로컬 원점수 → LB 환산: **LB ≈ 원점수 + 155** (v14 +153.33 / v15 +155.16).
     오프셋은 SHIFT 이득이 아니라 **구조적 이득**이다 — RF 는 시프트 없이도 +134.1.
2. **모델 선택/조기종료는 raw BSS** (클리핑 없는 값). `max(0,·)` 점수로 하면 음수 구간에서 오판.
3. **시드 명시**: 명령에 `--seed` 반드시 지정. NN은 시드 분산이 큼(±100점) → 3시드 이상 돌려 평균/앙상블로 판단.
4. **실험 번호**: docs/EXPERIMENTS_LOG.md의 다음 E## 번호를 선점하고 시작. 태그(`--tag`)는 `e##` 또는 모델별 버전(v#) — 점수판 비고에 매핑 기재.
5. **로그**: stdout을 `out/<실험명>.log`로 리다이렉트. 한글 grep은 인코딩 이슈 → 파일로 남기고 파일을 읽을 것.
6. **기록**: 실험 끝나면 즉시 docs/EXPERIMENTS_LOG.md 점수판 + 관찰/교훈 갱신. "부진한 실험"도 반드시 기록 (같은 실수 반복 방지).

## 3. 데이터 규칙 (실격 방지 — 어기면 실험 자체가 무효)

- test.csv의 다른 행/전체 분포를 쓰는 피처·보정 금지 (rolling, target encoding, 사후 스케일링 등).
- season 피처 **제거 금지** (drift 캘리브레이터 — E08에서 -580점 확인).
- 전처리기(스케일러/인코더)는 train.csv로만 fit.
- trackman 2025 없음 / 외부 데이터 금지.

## 4. 병렬 실행 규칙

### GPU 점유 확인 (실험 시작 전 필수)

```bash
ssh hsu-server "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
```

```bash
ssh desktop-4070 "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
```

- A100은 공유 서버 — 다른 사용자 프로세스가 있으면 대형 작업 자제. 우리 TabM은 ~6GB라 병렬 2~3개까지 가능.
- 4070은 16GB — XGBoost/CatBoost GPU 학습은 1개씩, LGBM CPU 시드 스윕은 코어 수만큼.

### A100 장기 실행 패턴 (ssh 끊겨도 유지)

```bash
ssh hsu-server "cd ~/aimers && setsid nohup ./run_X.sh > out/X.log 2>&1 < /dev/null & echo launched"
```

- 러너 스크립트(`run_*.sh`)는 노트북에서 작성 → scp → `sed -i 's/\r$//'`(CRLF 제거) → chmod +x.
- 죽일 때: `pkill -f run_X.sh; pkill -f train_tabm`.

### 4070 장기 실행 패턴

```
ssh desktop-4070
cd C:\aimers
Start-Process -NoNewWindow python -ArgumentList '-m','uv','run','python','src\train_xxx.py','--tag','e##' -RedirectStandardOutput out\e##.log
```

## 5. 제출 규칙

- 1일 5회. **로컬 검증에서 기존 최고 대비 +15점 이상일 때만 제출** (환산 신뢰도 유지 목적의 예외적 탐색 제출은 하루 1회 이내).
- 제출 전 체크: ① 로컬 스모크 완주(5행) ② zip 구조(script.py가 루트) ③ 서버 기본 패키지 버전 재명시 금지 ④ 새 라이브러리는 requirements에 추가(설치 시간 확인).
- 제출 후 docs/EXPERIMENTS_LOG.md 제출 이력에 LB 점수 즉시 기록.

## 6. 현재 베스트 스택 (2026-08-05)

- 단일: LGBM E06 (정규화 + ID 제외) — 로컬 632 / LB 764
- 블렌드: 0.65·LGBM + 0.35·TabM v4 3시드 — 로컬 669 (LB 예상 ~802)
- 다음 지렛대: 피처 v2(카운트 상호작용, cold-start 플래그), trackman 링키지, TabM 안정화(시즌 오프셋), CatBoost 추가
