"""블렌드 멤버들이 **같은 학습 데이터·같은 하이퍼**에서 나왔는지 확인한다.

2026-08-08 에 `VB_base` 를 `cat_v14f` 재현본이라고 믿고 그 검증 예측으로 가중을
고르려 했다. 실제로는 52점 약한 다른 모델이었다 — 핸드오프 명령에 `--drop-f-pre
2022` 가 섞여 학습 데이터가 달랐다. pkl 안의 `fpipe['priors']` 는 학습 행에서
계산되므로 **학습 집합의 지문**이다. 같으면 같은 데이터, 다르면 다른 데이터.

같은 실수를 v16(−6.15) 에서 한 번, v17(−53.6) 에서 한 번, 여기서 또 했다.

실행: python tools/member_fingerprint.py v14f ZD5 DX_seq
"""

import glob
import sys

import joblib

try:                                   # Windows 콘솔(cp949)에서 안 죽게
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

KEYS = ("learning_rate", "depth", "l2_leaf_reg", "loss_function", "border_count")


def main(tags):
    if not tags:
        tags = sorted({f.split("cat_")[1].rsplit("_s", 1)[0]
                       for f in glob.glob("./model/cat_*_s*.pkl")})
    rows = []
    for t in tags:
        fs = sorted(glob.glob(f"./model/cat_{t}_s*.pkl"))
        if not fs:
            print(f"없음: {t}")
            continue
        d = joblib.load(fs[0])
        p = d["model"].get_all_params()
        rows.append((t, len(fs),
                     d["fpipe"]["priors"]["asof_pitcher_success_rate"],
                     len(d["features"]),
                     tuple(round(p[k], 6) if isinstance(p.get(k), float)
                           else p.get(k) for k in KEYS),
                     d["best_iteration"], d["val_bss"]))

    print(f"{'태그':<12}{'시드':>4}{'학습집합 지문':>18}{'피처':>5}"
          f"{'best_it':>9}{'val_bss':>10}")
    for t, n, pr, nf, hp, bi, vb in rows:
        print(f"{t:<12}{n:>4}{pr:>18.10f}{nf:>5}{bi:>9}{vb:>10.2f}")

    prs = {r[2] for r in rows}
    hps = {r[4] for r in rows}
    print()
    if len(prs) > 1:
        print("!! 학습집합 지문이 다르다 — 이 멤버들은 서로 다른 데이터로 학습됐다.")
        print("   같은 표면에서 가중을 고르면 안 된다. 플래그를 대조할 것.")
    else:
        print("학습집합 지문 일치")
    if len(hps) > 1:
        print("주의: 하이퍼가 다른 멤버가 있다 (loss_function 은 달라도 정상)")
        for t, _, _, _, hp, _, _ in rows:
            print(f"   {t:<12}{hp}")
    return 2 if len(prs) > 1 else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
