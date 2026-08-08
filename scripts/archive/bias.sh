#!/bin/bash
# 1시즌 앞 편향의 **시계열**을 만든다 (E152).
#
# SHIFT 는 c = 0.65 x (2024 홀드아웃 편향) 으로 정했는데, 0.65 는 판단이었다.
# 이득식 1e5(2bc - c^2)/base 는 b 에 선형이라 **b 를 잘못 잡으면 손해가 크다**:
#   b=0.0066 이면 c=0.0052 로 이미 최적 근처 (상방 +0.7)
#   b=0.015  이면 c=0.0052 는 51.6 인데 최적은 90 (상방 +38)
# 그래서 b 를 시즌별로 재서 2025 를 외삽한다. train 자료만 쓰므로 합법이다.
#
# 구조: --val-season S-1 --test-season S 는 <=S-1 재학습 모델을 미학습 S 로 평가한다.
# 제출(<=2024 재학습 -> 2025)과 정확히 같은 구조다.
cd ~/aimers || exit 1
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022"
for S in 2021 2022 2023 2024; do
  V=$((S-1))
  echo "### 미학습 $S (검증 $V)"
  $P src/train_gbdt2.py --model cat $B --val-season $V --test-season $S \
     --tag BI$S --seeds 3,4,5 2>&1 | grep -E "^\[cat|미학습"
done
echo BIAS_DONE
