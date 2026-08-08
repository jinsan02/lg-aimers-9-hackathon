"""레버 영향성 평가표 — 두 홀드아웃에서의 leave-one-out 절제 (중첩/중복 검사).

지금까지의 판정은 전부 **한계 효과**(현재 설정 위에 얹었을 때)였다. 그건
'새 레버가 기존 것과 중복인가'는 잡지만, **이미 채택된 레버끼리의 중첩**은 못 본다.
2026-08-06 하루에 교호작용이 5번 나왔으므로 최종 스택에서 하나씩 빼봐야 한다.

두 홀드아웃을 쓰는 이유:
  2024 = 제출과 같은 '한 시즌 앞' 구조. 단 R리그 ABS 도입을 가로지른다.
  2023(R리그 전용) = 체제 변화가 없는 구간. **2024->2025 도 체제 변화가 없으므로**
                     오히려 이쪽이 제출 상황과 구조가 같다.
  둘 다에서 부호가 일치해야 그 레버가 체제와 무관한 진짜다.

사용: python tools/lever_impact.py
  ./out/*_A24_<arm>_s*_val_preds.npz 와 *_A23_* 를 읽는다.
"""

import glob
import re
import sys

import numpy as np

ARMS = [
    ("nov2", "피처 v2 (features.py)"),
    ("note", "타깃 인코딩 전체"),
    ("nodev", "TE _dev (E95)"),
    ("nostd", "시즌내 복원 std (E99)"),
    ("k30", "std-k 80 -> 30 (E106)"),
    ("career", "수축목표 리그평균 -> 자기통산"),
    ("nosp", "수축목표 시즌정렬 (E102)"),
    ("nodom", "야구 기전 교차항 (E111)"),
    ("lr02", "lr 0.01 -> 0.02"),
    ("w3", "+ 최신시즌 가중 W3 (E85, 지금은 제거 상태)"),
]


def load(prefix, arm):
    out = {}
    for f in glob.glob(f"./out/*_{prefix}_{arm}_s*_val_preds.npz"):
        m = re.search(rf"_{prefix}_{arm}_s(\d+)_val_preds\.npz$",
                      f.replace("\\", "/"))
        if m:
            z = np.load(f)
            out[int(m.group(1))] = (z["y"], z["pred"])
    return out


def stat(prefix, arm, full):
    d = load(prefix, arm)
    seeds = sorted(set(d) & set(full))
    if len(seeds) < 2:
        return None
    diffs = []
    for s in seeds:
        y, p = d[s]
        y2, p2 = full[s]
        r = y.mean()
        base = r * (1 - r)

        def b(q, yy):
            q = q - (q.mean() - yy.mean())        # 편향 제거 후 비교
            return 1e5 * (1 - ((q - yy) ** 2).mean() / base)
        diffs.append(b(p2, y2) - b(p, y))          # full - arm = 그 레버의 기여
    a = np.array(diffs)
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan
    return a.mean(), (a.mean() / se if se and se > 0 else 0.0), len(a)


def main():
    f24, f23 = load("A24", "full"), load("A23", "full")
    print(f"2024 full 시드 {sorted(f24)} | 2023R full 시드 {sorted(f23)}\n")
    print(f"{'레버 (빼면 잃는 값)':<38}{'2024':>9}{'t':>7}"
          f"{'2023R':>10}{'t':>7}   판정")
    for arm, name in ARMS:
        a = stat("A24", arm, f24) if f24 else None
        b = stat("A23", arm, f23) if f23 else None
        if a is None and b is None:
            continue
        c24 = f"{a[0]:>9.1f}{a[1]:>7.2f}" if a else f"{'-':>9}{'-':>7}"
        c23 = f"{b[0]:>10.1f}{b[1]:>7.2f}" if b else f"{'-':>10}{'-':>7}"
        v = "?"
        if a and b:
            if a[0] > 0 and b[0] > 0:
                v = "확증 (둘 다 기여)"
            elif a[0] < 0 and b[0] < 0:
                v = "**제거 검토** (둘 다 해로움)"
            else:
                v = "체제 의존 (부호 불일치)"
        print(f"{name:<38}{c24}{c23}   {v}")
    print("\n※ 값은 '그 레버를 빼면 잃는 BSS'. 양수면 기여, 음수면 빼는 게 낫다.")
    print("※ 전부 **편향 제거 후** 비교다 (수준은 SHIFT 가 따로 잡는다).")
    print("※ w3 팔은 '현재 스택 + W3' 다. 따라서 양수 = **넣지 않는 지금이 낫다**.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
