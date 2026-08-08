# 저장 전용(storage-only) 양자화 코덱 — 핸드오프

> 출처: 2026 AI·SW 디지털경진대회 AI부문(DACON Agent Action) 팀 토큰강도 제출물.
> 원본 코드: `C:\dacon\finals_code_submission\quantize_checkpoint.py` (int8) · `quantize_int4.py` (int4)
> 작성 근거: 위 두 파일 + `pack/script.py` 로더부(2384~2521행) + `junhyun_repo/tests/test_int4_mixed_codec.py` 실물 코드 정독.

---

## 0. 결론부터 — 이 대회엔 **아마 필요 없습니다**

| 항목 | DACON(원 출처) | aimers(현재) |
|---|---|---|
| 제출 zip 한도 | **1GB** | **10GB** (해제 후 32GB) |
| 모델 | 0.5B LLM × 3개 (fp16 3.4GB) | TabM/MLP 등 소형 tabular NN |
| 압축 없이 되나? | ❌ 한도 초과 → 코덱이 필수였음 | ✅ 여유 100배 이상 |

TabM 1~2M 파라미터면 fp32로 **4~8MB**입니다. 10GB 한도 안에서 **시드 100개를 fp32로 그냥 넣어도 1GB 미만**입니다.
→ **기본 판정: 쓰지 마세요.** 무손실 fp32/fp16 저장이 정답이고, 코덱은 정확도를 깎으면서 필요 없는 공간을 버는 순손실입니다.

### 그럼에도 꺼내 쓸 조건 (셋 중 하나라도 해당될 때)

1. 텍스트/범주형 임베딩용 **사전학습 트랜스포머를 zip에 동봉**하게 되는 경우 (0.5B만 돼도 fp16 1.1GB)
2. NN 시드 앙상블이 **수백 개 규모**로 커지는 경우
3. zip 한도가 아니라 **해제 후 32GB** 쪽에 먼저 걸리는 경우 (대형 아티팩트 동봉 시)

> ⚠️ zip 압축은 도움이 안 됩니다. safetensors 바이너리는 이미 엔트로피가 높아 zip으로 거의 안 줄어듭니다. 용량을 줄이려면 **바이트 자체를 줄이는** 이 코덱 같은 방식이어야 합니다.

---

## 1. 핵심 설계 원칙 — "추론 방식이 아니라 저장 방식"

이 코덱의 정체성은 한 줄입니다: **가중치를 디스크에 작게 저장하고, 로드하는 순간 fp16/fp32로 완전히 복원한다.**

- GPU 추론은 **plain fp16 그대로** — int8/int4 커널을 쓰지 않습니다.
- 따라서 **속도-정확도 트레이드오프가 없습니다.** 느려지지도, 빨라지지도 않습니다. 오직 파일 크기만 줄어듭니다.
- 손실은 **가중치 반올림 오차 한 번**뿐이고, 그 크기를 검증 하네스로 실측합니다.

심사·QnA에서 "양자화로 성능 떨어진 거 아니냐"는 질문에 이 한 줄로 답이 끝납니다. 우리 대회에서 실제로 그렇게 방어했습니다.

---

## 2. int8-rowwise-v1 (기본 코덱)

### 사양

| 항목 | 내용 |
|---|---|
| 대상 | `is_floating_point() and ndim >= 2`인 텐서만 |
| 방식 | **행별(per-row) 대칭 양자화** — 첫 축 기준, 나머지 축 전체에서 amax |
| 스케일 | `amax / 127.0`, fp32 저장. amax=0이면 1.0으로 치환(0 나눗셈 방지) |
| 값 범위 | `clamp(round(w/scale), -127, 127)` → int8 |
| 1D float | 양자화 안 함 → **fp16으로 다운캐스트** ⚠️(§6 주의) |
| 비-float | 그대로 통과 (int64 인덱스 등) |
| 크기 | fp32 대비 ~25%, fp16 대비 ~50% |

### 코어 (원본 그대로, torch + safetensors만 필요)

```python
SCALE_SUFFIX = ".__scale__"

def quantize_state_dict(state):
    packed, quantized, dtypes = {}, [], {}
    for name, tensor in state.items():
        dtypes[name] = str(tensor.dtype).replace("torch.", "")
        if tensor.is_floating_point() and tensor.ndim >= 2:
            w = tensor.float()
            amax = w.abs().amax(dim=tuple(range(1, w.ndim)))     # 첫 축만 남김
            scale = amax / 127.0
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            shaped = scale.view(-1, *([1] * (w.ndim - 1)))
            packed[name] = torch.clamp((w / shaped).round(), -127, 127).to(torch.int8)
            packed[name + SCALE_SUFFIX] = scale
            quantized.append(name)
        elif tensor.is_floating_point():
            packed[name] = tensor.to(torch.float16)              # 1D → fp16
        else:
            packed[name] = tensor                                # 비-float 통과
    return packed, {"format": "int8-rowwise-v1", "quantized": quantized, "dtypes": dtypes}
```

복원은 정확히 역순입니다 — `state[name] = (tensor.float() * shaped).to(dtype)`.
전체 로더는 원본 `quantize_checkpoint.py:52` `load_int8_state_dict()` 참고.

---

## 3. int4-group128-v1 (공격적 압축)

int8로 부족할 때만. fp16 대비 ~26%까지 내려갑니다.

### 사양

| 항목 | 내용 |
|---|---|
| 형태 변환 | `reshape(shape[0], -1)` — **2D로 평탄화** |
| 그룹 | 열 축을 **128개씩** 분할 (그룹 크기 조절 가능) |
| 스케일 | `amax/7.0`, **(행, 그룹)마다 하나**, fp16 저장 → int8보다 훨씬 촘촘함 |
| 값 범위 | `clamp(round, -7, 7)` → +8 오프셋 → `[0,15]` 니블 |
| 패킹 | **2개 값을 1바이트에**: `nib[:, 0::2] \| (nib[:, 1::2] << 4)` |
| 패딩 | 열이 128 배수가 아니면 우측 zero-pad, 복원 시 잘라냄 |
| 홀수 니블 | 행 끝이 홀수면 중립값 8 하나 추가(패킹 가능하게), 복원 시 제거 |

### 언패킹 (복원 핵심 3줄)

```python
lo = (tensor & 0x0F).to(torch.int8) - 8
hi = (tensor >> 4).to(torch.int8) - 8
q  = torch.stack((lo, hi), dim=2).reshape(rows, -1)      # 원래 순서로 인터리브
```

이후 `(q * scale)` → 패딩 제거 → 원 shape 복원. 전체는 `quantize_int4.py:154`.

---

## 4. mixed 모드 — 민감한 층만 정밀도 올리기

`int4-mixed-v1`. 실무에서 가장 쓸모 있는 부분입니다. 전부 fnmatch 패턴이라 층 이름으로 지정합니다.

| 옵션 | 효과 | 쓸 곳 |
|---|---|---|
| `--keep-fp16 PATTERN` | 해당 텐서는 fp16 원본 유지 | 출력 헤드(`score.weight` 등) — 가장 민감 |
| `--keep-int8 PATTERN` | int4 대신 행별 int8 | 중간 민감도 층 |
| `--group-override PAT=SIZE` | 그룹 크기 개별 조정(작을수록 정밀) | 오차 큰 특정 층 |
| `--int8-rows PAT=rows.json` | **한 텐서 안에서 지정 행만** int8, 나머지는 int4 | 고빈도 임베딩 행만 보호 |

예시(원본 테스트 케이스에서):
```bash
python quantize_int4.py quantize --input model.safetensors --output model.int4.safetensors \
  --keep-fp16 "score.weight" \
  --keep-int8 "*.q_proj.weight" \
  --group-override "*.mlp.*=64"
```

---

## 5. 파일 레이아웃과 로더 연결

```
model.int8.safetensors            # 패킹된 텐서 + .__scale__ 사이드카 텐서
model.int8.safetensors.meta.json  # ← 반드시 같이 이동. 없으면 복원 불가
```

`meta.json`에는 `format` / `quantized`(양자화된 텐서 이름 목록) / `dtypes` / int4는 추가로 `shapes`·`group_size`·`kept_fp16` 등이 들어갑니다. **로더가 이 목록을 보고 어떤 텐서를 어떻게 되돌릴지 결정하므로, 짝을 잃으면 파일이 죽습니다.**

추론 스크립트 연결(원본 `pack/script.py:2490~2521`)은 **파일명 자동 감지** 방식입니다:

```python
if os.path.exists(f"{hf_dir}/model.int4.safetensors"):      # int4 우선
    state = load_int4_state_dict(...)
elif os.path.exists(f"{hf_dir}/model.int8.safetensors"):    # 없으면 int8
    state = load_int8_state_dict(...)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    raise RuntimeError(...)                                  # 조용한 실패 금지
```

`strict=False` + **명시적 예외**가 중요합니다. 텐서 이름이 하나라도 어긋나면 조용히 랜덤 초기화 상태로 추론해버리는 사고를 막아줍니다.

---

## 6. ⚠️ tabular NN에 포팅할 때 주의점 4가지

원본은 HF 트랜스포머 전용으로 검증됐습니다. TabM 같은 소형 tabular NN엔 **다음 네 가지가 실제 위험**입니다.

### (1) BatchNorm 통계의 fp16 오버플로 — 가장 위험
1D float는 무조건 `to(torch.float16)`으로 다운캐스트됩니다. 그런데 tabular 모델은 입력 스케일이 정규화 안 된 경우가 많아 **`running_var`가 fp16 최대값 65504를 넘을 수 있습니다.** 넘으면 `inf` → 추론 전체가 NaN입니다. 반대로 아주 작은 분산은 언더플로(fp16 최소 정규값 ~6e-5)로 0이 되어 `1/sqrt(var+eps)` 폭발.
→ **대응**: `running_var`·`running_mean`을 `keep_fp16` 대상에서도 제외하고 **fp32 통과**로 예외 처리하거나, 포팅 전 `max(|1D float|)`를 찍어 65504 근처인지 확인.

### (2) ndim ≥ 3 앙상블 가중치는 int8이 오히려 불리
TabM의 efficient ensembling은 `(k, in, out)` 형태 3D 가중치를 씁니다. int8-rowwise는 **첫 축 기준 amax**라 → **앙상블 멤버 하나당 스케일 1개**라는 극단적으로 거친 양자화가 됩니다.
→ **대응**: 3D 텐서엔 **int4-group 경로가 더 정확합니다**(2D 평탄화 후 128열마다 스케일 = 훨씬 촘촘). 직관과 반대이니 주의. 또는 int8을 쓰려면 `(k*in, out)`으로 reshape 후 양자화.

### (3) 소형 모델은 int4 여유가 없음
LLM은 파라미터 중복이 커서 int4를 견딥니다. 1~2M 파라미터 모델은 중복이 적어 **같은 int4라도 훨씬 크게 망가집니다.** 우리도 0.5B 주 모델에 int4를 걸었을 때 argmax 일치가 94.5%로 무너져 보조 멤버에만 썼습니다.
→ **대응**: 소형 모델은 **int8부터**, 반드시 §7 검증 통과 후에만.

### (4) GBDT는 적용 불가
CatBoost/XGBoost/LightGBM은 torch state_dict가 아닙니다. 이 코덱과 무관하며, 각 라이브러리 자체 저장 포맷을 쓰세요.

---

## 7. 검증 — 이게 코덱의 절반입니다

압축률보다 **"내 모델에서 얼마나 손실됐는가"를 실측하는 하네스**가 본체입니다. 두 단계입니다.

### 1단계: roundtrip (합성 데이터, 10초)
```bash
python quantize_int4.py roundtrip --group-size 128
```
랜덤 텐서로 인코드→디코드 후 shape·오차를 확인합니다. 원본은 **패딩 경로(300열)·홀수 그룹(7×5)·1D 통과·비-float 통과**를 전부 커버합니다. 코드 수정 시 여기부터 깨집니다.

### 2단계: verify (실제 데이터로 예측 일치율)
원본 `cmd_verify`는 HF 전용이라 **그대로는 못 씁니다.** 이식할 뼈대는 이렇습니다:

```python
# 1) 원본 모델로 검증셋 추론 → logits_a  (메모리 해제!)
# 2) 코덱 복원 state_dict 로드한 모델로 추론 → logits_b
# 3) 세 가지 지표
weight_err = 상대 L1 오차           # 가중치 자체
logit_err  = (logits_a - logits_b).abs()
agreement  = (a.argmax(1) == b.argmax(1)).float().mean()   # ← 최종 판정 기준
```

원본은 여기에 확률 total variation과 불일치 행 목록까지 JSON 리포트로 남깁니다(`quantize_int4.py:405~457`). 회귀 문제라면 argmax 대신 **예측값 상관계수·RMSE 차이**로 바꾸면 됩니다.

**판정선 (우리 팀 운영 기준)**: argmax 일치 **99.5% 이상 → 채택**, 그 미만 → 해당 층 정밀도 상향 또는 코덱 포기.

> 💡 이 verify 패턴 자체는 코덱과 무관하게 재사용 가치가 있습니다. **어떤 손실 변환이든(양자화·프루닝·fp16 캐스팅·근사) "지표 대신 예측 일치율로 검증한다"**는 규율이 핵심입니다.

---

## 8. 우리 대회 실측값 (참고용 — 그대로 믿지 말고 각자 측정)

| 항목 | 값 |
|---|---|
| int8: 주 모델 | 1,132MB → **568MB** (fp16 대비 50%) |
| int8: argmax 일치 | **511/512** (README 기재 검증 런) · 발표자료 표기 99.6% |
| int4-group128: 보조 멤버 | **292MB** (fp16 대비 ~26%) |
| int4를 주 모델에 적용 시 | argmax **94.5%** → **기각**, 보조 멤버에만 사용 |
| 최종 패키지 | 0.5B × 3 = **1,005.6MB** (1GB 한도 통과, 1,031MB는 거부 실측) |
| 추론 속도 영향 | **없음** (fp16 추론 그대로) |

---

## 9. 가져다 쓸 때 (필요해졌다면)

```bash
# 원본 두 파일만 복사하면 끝 — 의존성은 torch + safetensors뿐
cp C:/dacon/finals_code_submission/quantize_checkpoint.py C:/aimers/tools/
cp C:/dacon/finals_code_submission/quantize_int4.py       C:/aimers/tools/

# 인코드
python tools/quantize_checkpoint.py quantize --input model/tabm.safetensors \
                                             --output model/tabm.int8.safetensors
# → tabm.int8.safetensors + tabm.int8.safetensors.meta.json 두 개 생성. 반드시 함께 zip에.
```

`cmd_verify`는 HF 의존이라 **삭제하고 §7 뼈대로 다시 쓰는 것**을 권합니다. `quantize_state_dict` / `load_*_state_dict` 네 함수만 순수 torch라 그대로 재사용 가능합니다.

---

## 10. 요약 판단표

| 상황 | 판단 |
|---|---|
| TabM/MLP 소형 NN 몇 개 | **코덱 불필요** — fp32/fp16 그대로 |
| GBDT 계열 | **적용 불가** |
| 시드 수백 개 앙상블 | int8 검토 (§7 검증 후) |
| 사전학습 트랜스포머 동봉 | int8 권장, 부족하면 int4-mixed |
| 어떤 경우든 | **압축률이 아니라 예측 일치율로 판정** |
