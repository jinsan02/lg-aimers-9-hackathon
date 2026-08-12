# scripts/ — 일회성 실행 스크립트

## 규칙

| 어디에 | 무엇을 |
|---|---|
| `tools/*.py`, `tools/run4070.sh` | **재사용하는 것.** 도구·검증기·런처 |
| `scripts/` | **이번에 한 번 돌릴 것.** 실험 러너 |
| `scripts/archive/` | 끝난 러너. 결과는 `LEDGER.tsv` / `docs/SETTLED.md` 에 있다 |

프로젝트 루트에 `.sh` / `.bat` 를 두지 않는다. 08-08 에 30개, 08-12 에 다시 51개가
쌓여서 어떤 게 살아 있는 건지 구분이 안 됐다. **규칙만으로는 두 번 다 안 지켜졌으므로
`tools/agent_sync.sh end` 가 루트 러너를 발견하면 종료코드 2 로 막는다.**

## 지금 `scripts/` 에 남아 있는 것 (2026-08-12 정리)

| 파일 | 왜 살아 있나 |
|---|---|
| `make_final_constants.bat` | **제출 챔피언 v11 이 읽는 `model/final_constants_2024.npz` 를 만든다.** 없으면 제출을 재현 못 한다 |
| `make_matchup_constants.bat` | 위와 같은 계열 (투수×타자 잔차 테이블) |
| `verify_v11pb.bat` · `audit_v11pb.bat` | 현행 챔피언 검증·감사 |
| `run_tdec1_5070.bat` | 진행 예정 (열-토큰 decoder, desktop-5070) |

나머지 126개는 전부 `scripts/archive/` 로 옮겼다. 재현은 `LEDGER.tsv` 의
명령 전문으로 하고, 파일 자체는 git 이력에 남아 있다.

## 이 스크립트들이 왜 남아 있나

`LEDGER.tsv` 가 **실행된 명령 전문**을 시드마다 기록하므로 재현에는 원장이면 된다.
다만 원장은 08-08 01:30 부터 쌓이기 시작했다 — 그 이전 실험은 스크립트가 유일한
기록이라 지우지 않는다.

## archive 목록 (2026-08-07 ~ 08)

| 스크립트 | 실험 | 판정 |
|---|---|---|
| `o1.bat` `o1.sh` | Optuna 최적 설정을 홀드아웃 시드로 재검정 | 기각 −7.44 (t=−2.65) |
| `zd5.bat` | 셀 멤버 depth5 재빌드 (refit 포함) | **채택** — v15 에 들어감 |
| `tm.bat` | trackman 투수×시즌 요약 13개 | 기각 −6.04 (t=−2.14) |
| `r23base.bat` | 2023R 기준선 (블렌드 가중 결정용) | 자료 생성 |
| `rm.bat` `rm2.bat` `rm3.sh` | refit 배수 1.5/1.7/2.0 | 기각 (18시드 합산 +0.60) |
| `bias.sh` | 1시즌 앞 편향 시계열 (BI2021~2024) | SHIFT 판정 근거 |
| `div.sh` `div2.sh` | 다양성 멤버 재검정 (LightGBM·부분공간·셀) | LightGBM·부분공간 기각 |
| `fleague.bat` | 구체제 F 유지 vs 제외 | 구조 인공물 — 결론 못 씀 |
| `v17.bat` | 구체제 F 제외 재빌드 | 기각 −53.6 |
| `surf_a.bat` `surf_b.sh` | 미학습 표면 축 재판정 A/B조 | 중단 (season 제거가 BANNED 였음) |
| `chain.sh` `chain2.sh` `chain3.sh` | A100 큐 연결 (rm3 → 표면 → 셀 동물원) | chain3 만 유효 |
| `surf4070.bat` | 표면 4축 (lr·depth·te-k·std-k) | 전부 기각 |
| `stdk.bat` | std-k 10/20/30/60 정밀 스윕 | 보류 (t<2.4) |
| `k40rep.bat` | **std-k 40 을 같은 머신에서 재현** | ★ 머신 간 비교 사고를 잡은 실험 |
| `tek.bat` | 축별 te-k (신뢰도 기반) | 기각 +0.58 / +2.26 |
| `a100base.sh` | A100 전용 기준선 (AB_base 883.41) | 자료 생성 — A100 결과 판정의 기준 |
| `cellctx.sh` | 셀 × 볼카운트 맥락 (32셀/100셀) | 기각 −5.13 / −74.71 |
| `night2.bat` | F 전용 모델 + refit 시드 추가 | F 전용은 크래시 (데이터 부족) |
| `night3.bat` | 셀 시드 6→8, te-k 100 재현 | 이득 0 / 기각 |
| `twostrike.bat` | 2스트라이크 세그먼트 모델 + 라우팅 | 기각 −13.61 |
| `distill.sh` | 증류 1차 | **누수** — 교사가 2024 를 봤다 |
| `distill2.sh` | 증류 재실행 (재학습 loss 수정 후) | 누수는 그대로 — 폐기 |
| `v19.sh` | std-k 40 으로 제출 모델 빌드 | 큐에서 제거 (근거가 머신 인공물이었음) |
| `pre-0807/` | 08-07 이전 러너 19개 | `docs/EXPERIMENTS_LOG.md` 참조 |
