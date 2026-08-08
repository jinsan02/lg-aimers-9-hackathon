"""블렌드 v9 — 제출 zip 의 script.py.

피처 생성은 전부 `fpipe.transform()` 이 한다. 학습(train_gbdt2)이 쓰는
`fpipe.fit()` 과 같은 파일에 나란히 있어서 순서가 어긋날 수 없다.
(예전엔 여기에 순서를 손으로 다시 적어서 두 번 깨졌다 — skill 위치,
 std_anchors 위치 튜플 인덱스.)

여기 남는 것은 **제출에만 있는 결정** 두 가지뿐이다.

WEIGHTS  설정끼리 greedy, 시드는 균등평균 (tools/group_blend.py).

SHIFT (E96) 한 시즌 앞을 예측하면 모델이 드리프트를 덜 반영해 전 구간에서
  균일하게 과대예측한다. 상수 시프트 c 의 이득은 (2bc - c^2)/(r(1-r))x1e5
  라 0 < c < 2b 면 항상 이득이다. 2024 홀드아웃 실측 편향 b_hat 에
  2년차 복귀 계수 0.65 를 곱해 쓴다. 단, 2026-08-09 공식 연혁 감사에서
  F리그 ABS는 2020년부터 운영된 것으로 확인되어 'F 2023=ABS 1년차'라는 인과 설명은
  폐기됐다. 0.65와 SHIFT 값은 별도 단일변경 검증 전까지 현행 제출 재현을 위해 유지한다.
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
# v16(=v15에서 0.38) 대비 **가중만** 올린다. 이유 두 가지:
#
# ① 지표를 잘못 썼다. decide_v15 는 **원점수**로 w 를 골랐는데, 파이프라인은
#    블렌드 뒤에 전역 SHIFT 를 적용하므로 수준은 이미 따로 처리된다.
#    편향제거 후로 다시 고르면 2024 최적이 0.45 -> 0.55 로 올라간다.
#      w=0.38 +12.64 / w=0.55 +14.14  (2024 자기검증)
#
# ② 셀 멤버를 **미학습 표면**(= 배치 구조)에서 재보니 base 보다 강했다.
#    자기검증 D +2.8 -> 미학습 D **-18.0**, margin +63.9, 최적 w 0.70.
#      gain(w) = w*63.9 - w^2*45.9  →  w=0.38 +17.7 / w=0.55 +21.2
#    (이 측정은 predict_proba[:,1] 버그로 어제 BSS -1367 로 버려졌던 것이다)
#
# 0.70 까지 안 가는 이유: 올바른 지표로 다시 그린 2023R 곡선은 최적이 0.20 이고
# w>0.35 에서 음수다. 세 측정 중 하나가 반대라 두 2024 계열의 합의점에 선다.
# (2023R 은 --league R 로 F 를 뺀 실험이고 평가 데이터엔 F 가 있어 신뢰도가 낮다)
_W_CELL = 0.55
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
    print(f"BLEND v9 mean={raw:.4f} -> 기울기 x{SLOPE} -> {preds.mean():.4f} "
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
