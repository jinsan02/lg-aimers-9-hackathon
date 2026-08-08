"""저장 전용(storage-only) 양자화 코덱 — TabM(tabular NN) 포팅판.

출처: docs/storage_codec_handoff.md (DACON 팀 토큰강도 int8-rowwise-v1 / int4-group128-v1)
목적: 가중치를 디스크에 작게 저장 → 로드 시 fp32로 복원. 추론 경로는 그대로(속도 영향 없음).

handoff §6의 tabular 위험 4가지에 대한 대응을 실제로 구현·검증한다:
 (1) 1D float(BatchNorm 통계) fp16 오버/언더플로 → 위험 시 fp32 통과
 (2) 3D 앙상블 가중치(k,in,out)는 int8-rowwise가 거침 → (k*in, out) reshape 후 행별 양자화
 (3) 소형 모델은 int4 여유 없음 → int8 기본
 (4) GBDT는 적용 불가

사용:
  python tools/quantize_tabm.py encode  model/tabm_v4s0_best.pt  model/tabm_v4s0.int8.pt
  python tools/quantize_tabm.py verify  model/tabm_v4s0_best.pt  model/tabm_v4s0.int8.pt
"""

import os
import sys

import numpy as np
import torch

SCALE_SUFFIX = ".__scale__"
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.1e-5


def quantize_state_dict(state):
    """int8 행별 대칭 양자화. 3D+ 텐서는 마지막 축만 남기고 평탄화(위험 2 대응)."""
    packed, meta = {}, {"format": "int8-rowwise-tabm-v1", "quantized": [],
                        "dtypes": {}, "shapes": {}, "fp32_passthrough": []}
    for name, t in state.items():
        meta["dtypes"][name] = str(t.dtype).replace("torch.", "")
        if t.is_floating_point() and t.ndim >= 2:
            w = t.float()
            meta["shapes"][name] = list(w.shape)
            # 위험 2: (k, in, out) → (k*in, out)로 펴서 행 수를 늘림 = 스케일 촘촘하게
            w2 = w.reshape(-1, w.shape[-1]) if w.ndim >= 3 else w
            amax = w2.abs().amax(dim=1)
            scale = amax / 127.0
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            q = torch.clamp((w2 / scale.view(-1, 1)).round(), -127, 127).to(torch.int8)
            packed[name] = q
            packed[name + SCALE_SUFFIX] = scale
            meta["quantized"].append(name)
        elif t.is_floating_point():
            # 위험 1: 1D float은 fp16 표현 범위를 벗어나면 fp32로 통과
            a = t.abs()
            nz = a[a > 0]
            unsafe = bool(a.max() > FP16_MAX) or (
                nz.numel() > 0 and bool(nz.min() < FP16_MIN_NORMAL))
            if unsafe:
                packed[name] = t.float()
                meta["fp32_passthrough"].append(name)
            else:
                packed[name] = t.to(torch.float16)
        else:
            packed[name] = t
    return packed, meta


def dequantize_state_dict(packed, meta):
    state = {}
    qset = set(meta["quantized"])
    for name, t in packed.items():
        if name.endswith(SCALE_SUFFIX):
            continue
        if name in qset:
            scale = packed[name + SCALE_SUFFIX]
            w = t.float() * scale.view(-1, 1)
            state[name] = w.reshape(meta["shapes"][name]).to(
                getattr(torch, meta["dtypes"][name]))
        else:
            state[name] = t.to(getattr(torch, meta["dtypes"][name]))
    return state


def cmd_encode(src, dst):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    packed, meta = quantize_state_dict(ckpt["state_dict"])
    out = {k: v for k, v in ckpt.items() if k != "state_dict"}
    out["packed"] = packed
    out["codec_meta"] = meta
    torch.save(out, dst)
    a, b = os.path.getsize(src) / 1e6, os.path.getsize(dst) / 1e6
    print(f"{os.path.basename(src)} {a:.2f}MB → {os.path.basename(dst)} {b:.2f}MB "
          f"({b / a * 100:.1f}%)")
    print(f"  양자화 텐서 {len(meta['quantized'])}개 | "
          f"fp32 통과(1D 위험) {len(meta['fp32_passthrough'])}개: "
          f"{meta['fp32_passthrough'][:3]}")


def load_quantized(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["state_dict"] = dequantize_state_dict(ckpt["packed"], ckpt["codec_meta"])
    del ckpt["packed"]
    return ckpt


def cmd_verify(orig, quant, data="data/processed/train_v1.npz", n=50000):
    """handoff §7 2단계 — 실제 데이터 예측 일치율로 판정 (회귀형 지표)."""
    sys.path.insert(0, "src")
    from eval_tabm_ensemble import CAT_NAMES, predict  # noqa: E402
    from tabm_reference import Model  # noqa: E402

    def build(ckpt):
        cfg = ckpt["config"]
        bins = ckpt["bins"]
        m = Model(n_num_features=cfg["n_num_features"],
                  cat_cardinalities=cfg["cat_cardinalities"], n_classes=None,
                  backbone=cfg["backbone"], bins=bins,
                  num_embeddings=cfg["num_embeddings"],
                  arch_type=cfg["arch_type"], k=cfg["k"])
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        kept = cfg.get("cat_cols_kept") or CAT_NAMES
        return m, [CAT_NAMES.index(c) for c in kept]

    d = np.load(data)
    is_val = d["season"] == 2024
    Xn = d["X_num"][is_val][:n]
    Xc = d["X_cat"][is_val][:n].astype(np.int64)
    y = d["y"][is_val][:n].astype(np.float64)

    ck_a = torch.load(orig, map_location="cpu", weights_only=False)
    ma, idx = build(ck_a)
    pa = predict(ma, Xn, Xc[:, idx])
    del ma

    ck_b = load_quantized(quant)
    mb, idx = build(ck_b)
    pb = predict(mb, Xn, Xc[:, idx])

    def bss(p):
        r = y.mean()
        return float(max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))))

    w_err = []
    for k_, v in ck_a["state_dict"].items():
        if v.is_floating_point():
            num = (v.float() - ck_b["state_dict"][k_].float()).abs().sum()
            den = v.float().abs().sum().clamp_min(1e-12)
            w_err.append((num / den).item())
    print(f"가중치 상대 L1 오차: 평균 {np.mean(w_err):.5f} 최대 {np.max(w_err):.5f}")
    print(f"예측 절대차: 평균 {np.abs(pa - pb).mean():.6f} "
          f"최대 {np.abs(pa - pb).max():.6f}")
    print(f"예측 상관: {np.corrcoef(pa, pb)[0, 1]:.6f}")
    print(f"BSS  원본 {bss(pa):.2f} → 코덱 {bss(pb):.2f} "
          f"(차이 {bss(pb) - bss(pa):+.2f})")
    ok = np.corrcoef(pa, pb)[0, 1] > 0.9995 and abs(bss(pb) - bss(pa)) < 5
    print("판정:", "✅ 채택 가능" if ok else "❌ 손실 과다")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "encode":
        cmd_encode(sys.argv[2], sys.argv[3])
    elif cmd == "verify":
        cmd_verify(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: encode SRC DST | verify ORIG QUANT")
