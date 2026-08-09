"""제출 경로가 **행 독립**인가 — 기계적으로 확인한다 (E167).

데이콘 2026-08-06 공지: "평가 데이터의 다른 행이나 전체 평가 데이터의 분포를
이용해 특정 행의 예측값을 보정하거나 생성하는 방식은 정상적인 추론 절차로
인정되지 않습니다." 위반 시 실격.

우리는 이 대회에서 **행 의존 신호가 얼마나 큰지 직접 재봤다**:
  - asof 한 투구 차분으로 직전 투구 결과 99.9% 복원 → BSS 상당 **+127**
  - 같은 대수로 다음 행을 보면 **그 행 자신의 타깃이 96.79% 복원**
즉 유혹의 크기를 알고도 안 쓴 것이므로, 안 썼다는 것도 증명해 둔다.

## 검사 방법 (주장이 아니라 실측)

같은 test 행을 **① 전체 배치로** 예측한 값과 **② 한 행씩 따로** 예측한 값이
비트 단위로 같은지 본다. 다른 행을 하나라도 참조하면 값이 달라진다.
행 순서를 뒤섞어도 같은지 함께 본다(정렬·이웃 의존 탐지).

실행: python tools/audit_rowindep.py [script_blend_v9.py]
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "script_blend_v9.py"
    path = name if os.path.isfile(name) else os.path.join("src", name)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location("_sub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    n = len(test)
    print(f"{name} | test {n}행 | 모델 {len(mod.WEIGHTS)}개 "
          f"| SHIFT {mod.SHIFT} SLOPE {mod.SLOPE}")

    batch = np.asarray(mod.blend(test), dtype=np.float64)

    # ① 한 행씩 따로
    one = np.array([float(np.asarray(mod.blend(test.iloc[[i]]))[0])
                    for i in range(n)])
    d1 = float(np.abs(batch - one).max())

    # ② 순서를 뒤집어 예측한 뒤 원래 순서로 되돌리기
    rev = test.iloc[::-1].reset_index(drop=True)
    pr = np.asarray(mod.blend(rev), dtype=np.float64)[::-1]
    d2 = float(np.abs(batch - pr).max())

    # ③ 절반만 넣고 예측 (배치 구성이 값을 바꾸는가)
    half = test.iloc[: max(1, n // 2)]
    ph = np.asarray(mod.blend(half), dtype=np.float64)
    d3 = float(np.abs(batch[: len(ph)] - ph).max())

    print(f"\n{'검사':<34}{'최대 |차이|':>14}  판정")
    for nm, d in (("① 전체배치 vs 한 행씩", d1),
                  ("② 행 순서 역순", d2),
                  ("③ 절반 배치", d3)):
        print(f"{nm:<34}{d:>14.3e}  {'통과' if d == 0.0 else '★ 행 의존 의심'}")
    ok = max(d1, d2, d3) == 0.0
    print("\n" + ("행 독립 확인 — 다른 행이나 배치 분포를 전혀 안 본다."
                  if ok else "★ 값이 배치 구성에 따라 변한다. 추론 경로를 점검할 것."))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
