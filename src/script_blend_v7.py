"""블렌드 v7 — 제출 zip 의 script.py.

피처 생성은 전부 `fpipe.transform()` 이 한다. 학습(train_gbdt2)이 쓰는
`fpipe.fit()` 과 같은 파일에 나란히 있어서 순서가 어긋날 수 없다.
(예전엔 여기에 순서를 손으로 다시 적어서 두 번 깨졌다 — skill 위치,
 std_anchors 위치 튜플 인덱스.)

여기 남는 것은 **제출에만 있는 결정** 두 가지뿐이다.

WEIGHTS  설정끼리 greedy, 시드는 균등평균 (tools/group_blend.py).

SHIFT (E96) 한 시즌 앞을 예측하면 모델이 드리프트를 덜 반영해 전 구간에서
  균일하게 과대예측한다. 상수 시프트 c 의 이득은 (2bc - c^2)/(r(1-r))x1e5
  라 0 < c < 2b 면 항상 이득이다. 2024 홀드아웃 실측 편향 b_hat 에
  2년차 복귀 계수 0.65 를 곱해 쓴다 (E105: F리그 2년차 -0.0068 = 평시 수준).
  train 만으로 정한 상수를 모든 행에 동일 적용하므로 후처리 금지에 저촉되지
  않는다 — 평가 데이터의 분포를 전혀 보지 않는다.
"""

import os

import joblib
import numpy as np
import pandas as pd

import fpipe

ID_COL = "row_id"
TARGET_COL = "control_success"
SHIFT = 0.0052

# 로짓 기울기 보정. SHIFT 는 0차(절편)만 고치는데, 19모델 평균은 구조적으로
# **과소분산**이라 1차(기울기)도 어긋나 있다. 2023 홀드아웃에서 적합한 기울기가
# 1.0416, 2024 자기적합이 1.0494 로 두 시즌이 거의 같다 — 시즌에 안 의존하는
# 구조적 성질이라는 뜻이다. 실제로 2023 적합값을 2024 에 적용하면 +3.51 이다.
# (같은 회귀의 **절편**은 −48.34 로 전혀 안 넘어간다. 절편은 시즌 드리프트라
#  해마다 다르고, 그건 SHIFT 가 따로 맡는다.)
# train 만으로 정한 상수를 각 행에 동일 적용하므로 행 독립 원칙을 지킨다.
SLOPE = 1.0416

# v15 = v14 의 셀 멤버를 **depth 8 -> 5** 로 바꾸고 가중을 다시 정한 것.
# depth 5 가 나은 이유는 margin 이다: 단독 점수는 비슷한데 base 와의 성능격차 D 가
# 25.5 -> 2.8 로 줄어 margin = 400,320 x rms^2 - D 가 커진다 (tools/margin.py).
#
# 가중은 **2023R 에서 정하고 2024 에서 확인**했다 (오늘 세운 시즌 이전 원칙).
#   2023R 최적 0.30 (+3.46)  ->  그대로 2024 에 적용하면 +8.30
#   2024 자기적합 최적 0.45 (+9.37, 낙관치)
# 두 시즌 최적의 중점 0.38 을 쓴다. 0.30~0.45 구간에서 2024 이득이 +8.3~+9.4 로
# 평평하므로 어디를 골라도 큰 차이가 없다 — 최적화가 아니라 강건성 선택이다.
_W_CELL = 0.38
_F = [42, 7, 13, 3, 4, 5, 6, 8]
_C = [42, 7, 13, 3, 4, 5]
WEIGHTS = ([(f"./model/cat_v14f_s{s}.pkl", (1 - _W_CELL) / len(_F)) for s in _F]
           + [(f"./model/cat_ZD5_s{s}.pkl", _W_CELL / len(_C)) for s in _C])


def blend(test):
    """가중 평균 후 시프트. 각 행은 자기 자신만 보고 예측된다."""
    total = sum(w for _, w in WEIGHTS)
    preds = np.zeros(len(test))
    for path, w in WEIGHTS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 없음: {path}")
        p = fpipe.predict(joblib.load(path), test)
        preds += (w / total) * p
        print(f"  {os.path.basename(path)} (w={w:.4f}): mean={p.mean():.4f}")
    raw = preds.mean()
    if SLOPE != 1.0:
        z = np.log(np.clip(preds, 1e-6, 1 - 1e-6) / (1 - np.clip(preds, 1e-6, 1 - 1e-6)))
        preds = 1.0 / (1.0 + np.exp(-SLOPE * z))
    print(f"BLEND v7 mean={raw:.4f} -> 기울기 x{SLOPE} -> {preds.mean():.4f} "
          f"-> SHIFT -{SHIFT:.4f}")
    return np.clip(preds - SHIFT, 0.0, 1.0)


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if list(sub.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼 불일치: {list(sub.columns)}")
    print(f"test={len(test)} submission={len(sub)}")

    preds = blend(test)
    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur
                       in zip(sub[ID_COL], sub[TARGET_COL])]
    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)}) "
          f"mean={preds.mean():.4f}")


if __name__ == "__main__":
    main()
