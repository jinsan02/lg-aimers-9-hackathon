# 딥러닝 모델 선별 (A100 학습용) — 2026-08-05 조사

대회 규칙 기준 필터: ① 가중치 공개 + 최소 비상업 허용 라이선스(사전학습 모델인 경우)
② 평가 서버 인터넷 차단 → 가중치/코드 전부 zip 동봉 ③ 설치 ≤10분 ④ 추론 ≤10분(L4 22.4GB)
⑤ Python 3.11 / torch 2.7.1+cu128 기본 설치.

**핵심 판단: from-scratch 학습 모델은 사전학습 가중치 규칙의 적용 대상이 아님** —
코드 라이선스(Apache/MIT)만 확인하면 됨. 파운데이션 모델(TabPFN 등)만 가중치 라이선스 검토 필요.

## 선별 결과

| 모델 | 유형 | 라이선스 | 판정 | 비고 |
|---|---|---|---|---|
| **TabM** (Yandex, ICLR 2025) | from-scratch MLP + BatchEnsemble | Apache 2.0, PyPI `tabm` | ✅ **주력 추천** | GBDT급 성능, MLP급 추론 속도. 코드가 단일 파일(`tabm_reference.py`) → zip에 vendoring하면 설치 0 |
| **MLP-PLR** (rtdl_num_embeddings) | from-scratch MLP + 수치 임베딩 | MIT | ✅ 추천 | PLR 임베딩은 TabM에도 결합 가능. 소형 패키지, vendoring 가능 |
| **FT-Transformer / ResNet** (rtdl_revisiting_models) | from-scratch | MIT | ✅ 가능 | 147만 행 학습은 A100에서 OK. TabM보다 무겁고 성능 이점 불확실 → 앙상블 다양성용 |
| **RealMLP** (pytabkit) | from-scratch MLP | Apache 2.0 | ✅ 가능 (학습만) | TabArena 상위권. 단 pytabkit 의존성(lightning 등)이 무거움 → **A100 학습 전용**, 추론은 state_dict + 순수 torch 코드로 이식 |
| **TabICL v2** (soda-inria) | 파운데이션 (ICL) | 코드 permissive, 가중치 HF 공개 | ⚠️ 조건부 실험 | ~50만 행 컨텍스트까지 — 147만 행은 서브샘플 필요. 가중치 동봉 + L4에서 245k 추론 시간 실측 필요 |
| **TabPFN 2.5/2.6** (Prior Labs) | 파운데이션 | tabpfn-2.5-license-v1.1 | ❌ 제외 | 상업/프로덕션 및 competitive benchmarking 금지 조항 → 대회 사용 규칙상 회색지대 + 50k 행 성능 한계로 실익도 없음 |
| TabNet, NODE, SAINT | from-scratch | 허용적 | ❌ 제외 | 최신 벤치마크에서 TabM/RealMLP/GBDT에 일관되게 열세 |
| AutoGluon (NN 포함) | 프레임워크 | Apache 2.0 | ❌ 제외 | 서버 기본 패키지와 버전 충돌/설치 10분 초과 위험 |

## 제출 전략 — "학습은 자유, 추론은 순수 torch"

평가 서버에는 torch 2.7.1이 이미 있으므로, **어떤 프레임워크로 학습하든 추론은
state_dict(.pt) + vendoring한 모델 정의 코드**로 수행하면 requirements.txt가 계속 빈 상태 유지:

```
submit.zip
├── model/
│   ├── tabm.pt            # state_dict (from-scratch 학습 가중치 — 규칙 제약 없음)
│   └── preprocess.pkl     # 스케일러/인코더 (sklearn 기본 설치)
├── tabm_reference.py      # Apache 2.0 단일 파일 vendoring (출처 주석 명기)
├── script.py
└── requirements.txt       # 빈 파일 유지
```

- TorchScript/ONNX export도 대안이지만 state_dict + 코드 동봉이 디버깅 쉬움.
- L4 추론 예산: 245,789행 × TabM(k=32 앙상블) ≈ 수 초. 시간 제약 없음.
- 앙상블 멤버 여러 개(.pt 여러 개)도 zip 10GB 내 여유.

## 빅테크 대형모델(LLM) 가중치 사용 검토 (2026-08-05)

**규칙상으로는 가능** — Qwen(Apache 2.0), DeepSeek(MIT), Mistral(Apache 2.0), Llama/Gemma(비상업
허용 커뮤니티 라이선스) 모두 "가중치 공개 + 최소 비상업 허용" 조건을 충족. 로컬 실행이므로
원격 API 금지 조항에도 걸리지 않음.

**실전에서는 부적합** — 세 가지 정량 근거:

1. **추론 10분 한도**: 245,789행을 텍스트로 인코딩하면 행당 ~200토큰 × 245k행 ≈ 4,900만 토큰.
   L4에서 0.5B 모델 prefill 처리량(~15k tok/s)으로도 약 54분 → 초과. 7B는 ~10배 더 느림.
   프롬프트를 ~50토큰으로 줄여도 0.5B가 마지노선(10~20분, 아슬아슬).
2. **zip 10GB 한도**: fp16 기준 7B = 14GB로 초과(가중치는 압축이 안 됨). 양자화(bitsandbytes 등)는
   추가 설치 필요 → 설치 리스크. ≤5B만 fp16 동봉 가능.
3. **성능 근거 없음**: 이 데이터는 전부 수치/범주형 — 텍스트가 없어 LLM의 사전학습 이점이 작동할
   여지가 없음. 소형 LLM의 tabular 성능은 GBDT/TabM에 일관되게 열세 (문헌·벤치마크 공통).
   "LLM 사전학습 지식을 tabular에 이식" 역할은 이미 TabICL 같은 tabular 전용 파운데이션 모델이 담당.

**결론: LLM 트랙은 제외. 파운데이션 모델 실험은 TabICL로 한정.**

## A100 서버 현황 (탐색 완료 2026-08-05 — 있는 것 재사용, 신규 다운로드 금지)

- 작업 디렉토리: `/home/token1234` (홈). 기존 파일(aadp, auto_farm*, A0/A1 zip 등)은 이전 프로젝트 — 건드리지 말 것. 프로젝트는 `~/aimers` 새 디렉토리에서.
- **재사용할 것**: `~/venv451` (Python 3.10.12, **torch 2.5.1+cu121 CUDA 작동 확인**, transformers 4.51.3, pandas 2.3.3, sklearn 1.7.2, accelerate, 5.3GB). venv52는 transformers 5.x 실험용 — 사용 안 함.
- HF 캐시: HyperCLOVAX-SEED 0.5B (1.1GB) 있음 — 우리 용도엔 불필요.
- 디스크 여유 49GB — torch 재설치(~5GB) 불필요해짐. uv/conda 없음.
- **주의: venv451의 sklearn은 1.7.2 (평가 서버는 1.8.0)** → sklearn 전처리 pkl은 A100에서 만들지 말 것. 전처리는 노트북(1.8.0)에서 수행·저장하고, A100은 전처리 완료된 배열(.npy/.parquet)로 torch 학습만 담당. state_dict는 torch 버전 간 호환되므로 2.5.1로 학습 → 서버 2.7.1 로드 OK.
- venv451은 공용 계정의 기존 환경이므로 **패키지 설치/업그레이드 금지**. 추가 패키지가 꼭 필요하면 별도 venv를 새로 만들 것.

## A100 학습 계획

1. hsu-server 환경: Ubuntu 22.04, Python 3.10 시스템 → **uv로 3.11 + torch cu12x 별도 구성**
   (디스크 여유 49GB — torch 설치 ~5GB OK. 드라이버 535 = CUDA 12.x minor compatibility로 cu128 wheel 구동 가능)
2. 1순위 실험: **TabM-mini + PLR 임베딩** (파라미터 효율 앙상블 k=32)
   - 범주형: pitcher_id(792), batter_id(830) → 임베딩 레이어
   - 수치형: QuantileTransform/PLR
   - 시즌 drift 대응: 최근 시즌 오버샘플링/가중 실험
3. 2순위: RealMLP(pytabkit), FT-Transformer — 앙상블 다양성 확보
4. 최종: GBDT(4070) + NN(A100) 확률 평균/스태킹, Brier 직접 최적화(BCE와 근접하므로 logloss 학습 후 캘리브레이션 점검)

## 참고 링크

- TabM: https://github.com/yandex-research/tabm (PyPI: https://pypi.org/project/tabm/)
- rtdl: https://github.com/yandex-research/rtdl_num_embeddings
- pytabkit/RealMLP: https://github.com/dholzmueller/pytabkit
- TabICL: https://github.com/soda-inria/tabicl
- TabPFN 2.5 라이선스: https://huggingface.co/Prior-Labs/tabpfn_2_5
