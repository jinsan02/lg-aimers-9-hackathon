"""LLM 확장 가능성 실측 — 직렬화 비용 + 실제 추론 시간.

핵심 설계 판단:
  이 과제는 분류/확률예측이므로 LLM을 쓴다면 **AutoModelForSequenceClassification
  방식의 단일 forward(prefill only)** 다. 자기회귀 생성이 없으므로 **KV-cache 자체가
  발생하지 않고, 따라서 KV-cache 양자화는 적용 대상이 아니다.**
  속도는 가중치 값과 무관하므로 랜덤 초기화 모델로 측정해도 결과가 동일하다
  (다운로드 불필요 = 평가 서버 인터넷 차단 조건과도 부합).

측정: (모델 크기) × (시퀀스 길이) → 초당 행 처리량 → 245,789행 소요시간 환산.
실행(A100): ~/venv451/bin/python tools/llm_feasibility.py
"""

import time

import torch
from transformers import LlamaConfig, LlamaForSequenceClassification

N_ROWS = 245_789
LIMIT_S = 600.0
# A100(PCIe, bf16 dense ~312TF) → L4(fp16 dense ~121TF, 실효 ~1/3.5)
L4_FACTOR = 3.5

CONFIGS = {  # (hidden, layers, heads, intermediate) ≈ 파라미터 수
    "0.1B": dict(hidden_size=576, num_hidden_layers=12, num_attention_heads=9,
                 num_key_value_heads=3, intermediate_size=1536),
    "0.5B": dict(hidden_size=896, num_hidden_layers=24, num_attention_heads=14,
                 num_key_value_heads=2, intermediate_size=4864),
    "1.5B": dict(hidden_size=1536, num_hidden_layers=28, num_attention_heads=12,
                 num_key_value_heads=2, intermediate_size=8960),
}


def serialize_example():
    """투구 1행 → 텍스트 직렬화 예시와 길이 산출."""
    row = {
        "season": 2025, "month": 7, "inning": 6, "half": "T", "type": "R",
        "b": 2, "s": 1, "o": 1, "base": "1_3", "score_diff": -2, "li": 1.42,
        "p_hand": "R", "b_hand": "L",
        "p_n": 1842, "p_succ": 0.512, "p_rev": 0.083, "p_mid": 0.147,
        "p_prev1": 0.488, "p_prev3": 0.501, "p_prev5": 0.509,
        "bat_n": 2310, "bat_succ": 0.523, "fb": 0.54, "br": 0.29, "os": 0.17,
    }
    compact = " ".join(f"{k}={v}" for k, v in row.items())
    verbose = ("2025년 7월 6회초 정규시즌 경기. 볼카운트 2-1, 1아웃, 1·3루 주자, "
               "점수차 -2, 상황중요도 1.42. 우완 투수(누적 1842구, 제구성공률 0.512, "
               "반대방향 0.083, 몰린공 0.147, 최근1경기 0.488, 3경기 0.501, "
               "5경기 0.509)가 좌타자(2310구 상대, 0.523)를 상대한다. "
               "구종비율 속구 0.54 변화구 0.29 체인지업 0.17.")
    return compact, verbose


def bench(name, cfg, seq_len, batch, dtype=torch.bfloat16, device="cuda"):
    config = LlamaConfig(vocab_size=32000, num_labels=2, max_position_embeddings=2048,
                         pad_token_id=0, **cfg)
    model = LlamaForSequenceClassification(config).to(device=device, dtype=dtype).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    ids = torch.randint(0, 32000, (batch, seq_len), device=device)
    with torch.no_grad():
        for _ in range(2):  # warmup
            model(ids)
        torch.cuda.synchronize()
        t = time.perf_counter()
        reps = 5
        for _ in range(reps):
            model(ids)
        torch.cuda.synchronize()
        el = (time.perf_counter() - t) / reps
    rows_s = batch / el
    total_a100 = N_ROWS / rows_s
    total_l4 = total_a100 * L4_FACTOR
    verdict = "✅" if total_l4 < LIMIT_S else "❌"
    print(f"{name:5s} ({n_params:.2f}B) seq={seq_len:4d} b={batch:3d} | "
          f"{rows_s:8.1f} row/s | A100 {total_a100 / 60:6.1f}분 | "
          f"L4추정 {total_l4 / 60:6.1f}분 {verdict}")
    del model
    torch.cuda.empty_cache()
    return total_l4


def main():
    compact, verbose = serialize_example()
    print("=== 1. 직렬화 방식 ===")
    print(f"[compact] {len(compact)}자: {compact[:110]}...")
    print(f"[verbose] {len(verbose)}자: {verbose[:60]}...")
    print(f"토큰 추정(영숫자 ~3.5자/토큰): compact ≈ {len(compact) // 3.5:.0f}토큰 | "
          f"verbose(한글 ~1.5자/토큰) ≈ {len(verbose) / 1.5:.0f}토큰\n")

    print(f"=== 2. 추론 실측 ({N_ROWS:,}행, prefill only, 한도 {LIMIT_S / 60:.0f}분) ===")
    print(f"    L4 환산계수 {L4_FACTOR}× (A100 대비 보수적 추정)\n")
    for name, cfg in CONFIGS.items():
        for seq_len, batch in [(32, 512), (48, 384), (64, 256),
                               (128, 128), (256, 64)]:
            try:
                bench(name, cfg, seq_len, batch)
            except torch.cuda.OutOfMemoryError:
                print(f"{name:5s} seq={seq_len} b={batch} | OOM")
                torch.cuda.empty_cache()

    print("\n=== 3. 참고: 현 제출 모델 ===")
    print("    CatBoost 245,789행 추론 실측 ≈ 3초 (한도의 0.5%)")


if __name__ == "__main__":
    main()
