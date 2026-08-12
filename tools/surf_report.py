"""미학습 표면 축 재판정 결과표. 기준선과 페어 비교해서 t 까지 낸다.

test_preds npz 만 읽으므로 로그 인코딩과 무관하다 (한글 grep 이 머신마다 깨져서
로그로 읽으려다 두 번 헛발질했다).

실행: python tools/surf_report.py RN1.5 SC_k40 SC_k120 SD_lr02 ...
"""

import glob
import sys

import numpy as np

from invalidated import guard as _guard_invalidated


def machine_of(tag):
    """LEDGER 에서 그 태그를 만든 머신을 찾는다.

    2026-08-08: std-k 40 이 A100 에서 +15.05, 4070 에서 +4.27 이었다. 같은 설정·
    같은 시드인데 11점이 **머신 차이**다. 기준선이 4070 산인 줄 모르고 A100 결과와
    비교해서 하마터면 근거 없는 설정으로 제출할 뻔했다. 이제 비교 전에 대조한다.
    """
    try:
        with open("./LEDGER.tsv", encoding="utf-8") as f:
            hosts = {c[1] for c in (l.split("\t") for l in f)
                     if len(c) > 2 and c[2].startswith(tag + "_s")}
        return "/".join(sorted(hosts)) if hosts else "?"
    except OSError:
        return "?"


def load(tag):
    fs = sorted(glob.glob(f"./out/*_{tag}_s*_test_preds.npz"))
    if not fs:
        return None, None, []
    seeds, preds, y = [], [], None
    for f in fs:
        z = np.load(f, allow_pickle=True)
        seeds.append(f.split("_s")[-1].split("_")[0])
        preds.append(z["pred"].astype(np.float64))
        y = z["y"].astype(np.float64)
    return dict(zip(seeds, preds)), y, seeds


def main():
    _guard_invalidated(sys.argv[1:])
    tags = sys.argv[1:]
    if len(tags) < 2:
        print(__doc__)
        return 1
    B, y, bs = load(tags[0])
    if B is None:
        print(f"기준 없음: {tags[0]}")
        return 1
    r = float(y.mean())
    base = r * (1 - r)

    def sc(p):
        p = p - (p.mean() - r)          # 수준은 SHIFT 가 따로 맡는다
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)

    b_ens = sc(np.mean(list(B.values()), 0))
    b_host = machine_of(tags[0])
    print(f"기준 {tags[0]}  {len(bs)}시드 앙상블 {b_ens:.2f}  "
          f"[{b_host}]  (미학습 {len(y):,}행)\n")
    print(f"{'축':<12}{'시드':>5}{'앙상블':>10}{'Δ':>8}{'페어평균':>10}"
          f"{'SE':>7}{'t':>7}  판정  머신")
    for t in tags[1:]:
        A, _, _ = load(t)
        if A is None:
            print(f"{t:<12}  (없음)")
            continue
        common = sorted(set(A) & set(B))
        if not common:
            print(f"{t:<12}  (공통 시드 없음)")
            continue
        d = np.array([sc(A[s]) - sc(B[s]) for s in common])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        ens = sc(np.mean([A[s] for s in common], 0))
        tv = d.mean() / se if se else float("nan")
        hi = d.mean() + 1.96 * se
        verdict = ("채택" if tv >= 2.4 else ("기각" if hi < 3 else "보류"))
        host = machine_of(t)
        # 머신이 다르면 판정을 지운다. 11점짜리 머신 효과가 실측됐으므로
        # 이 비교는 t 값이 아무리 커도 의미가 없다.
        if host != "?" and b_host != "?" and host != b_host:
            verdict = "**무효(머신다름)**"
        print(f"{t:<12}{len(common):>5}{ens:>10.2f}{ens - b_ens:>+8.2f}"
              f"{d.mean():>+10.2f}{se:>7.2f}{tv:>+7.2f}  {verdict}  {host}")
    print("\n※ 채택 t>=2.4 / 기각 95%상한<+3 / 그 외 보류(시드 추가 필요)")
    print("※ 기준과 머신이 다르면 판정 무효 — std-k 40 이 A100 +15.05 / 4070 +4.27 이었다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
