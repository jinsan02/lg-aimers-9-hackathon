#!/bin/bash
# E142 검정 — Optuna 최적 설정을 **탐색에 안 쓴 시드**로 재현한다.
#
# tune.py 는 시드 42,7 로만 점수를 매겼다. 40 trial 중 최고값이므로 승자의 저주가
# 끼어 있다(2시드 SE 3.2 → 선택 편의만으로 +3~6 가능). 기준선 v14f 는 시드
# 3,4,5,6,8,13 이 이미 있고 그 6시드 평균이 945.25 다. 같은 6시드로 재면
# 선택 편의가 빠진 순수 차이가 나온다.
#
# tune.py 와 정확히 맞춘 것: iterations 4000 / es 400 / no-refit
#   (검증 예측은 refit 이전 모델에서 나오므로 refit 은 이 비교에 무관하고 시간만 먹는다)
cd /mnt/c/aimers || exit 1
P=./.venv/Scripts/python.exe
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc"
O="--lr 0.0081 --depth 8 --l2 41.9097 --border-count 128 --bagging-temp 0.5508 --random-strength 0.6526 --cat-min-leaf 85"
$P src/train_gbdt2.py --model cat $B $O --iters 4000 --es 400 --no-refit \
   --tag O1 --seeds 3,4,5,6,8,13
echo O1_DONE
