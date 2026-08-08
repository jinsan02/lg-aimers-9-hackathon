#!/bin/bash
# refit 배수 1.5 vs 2.0 을 **시드 6개 추가**로 판정한다.
# 현재 +2.44 (t=1.91, 95%CI [-0.06,+4.94]) — 채택 기준 t>=2.4 에 미달.
# 6시드를 더 붙이면 SE 가 1.28 -> 0.9 로 줄어 효과가 실재하면 t~2.7 이 된다.
# 시드는 기존(3,4,5,6,8,13)과 겹치지 않게 고른다.
cd ~/aimers || exit 1
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --drop-f-pre 2022"
V="--val-season 2023 --test-season 2024"
S="--seeds 21,22,23,24,25,26"
for M in 1.5 2.0; do
  $P src/train_gbdt2.py --model cat $B $V $S --refit-mult $M --tag RQ$M 2>&1 \
    | grep -E "^\[cat|미학습|!!"
done
echo RM3_DONE
