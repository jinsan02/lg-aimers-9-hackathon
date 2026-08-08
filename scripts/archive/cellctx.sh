#!/bin/bash
# E164 — 셀 코드에 **볼카운트 맥락**을 붙인다.
#
# 손실 지도: 0-2 카운트 BSS 277.6 / 1-2 629.7 (전체 878.8). 2스트라이크에서
# 포수가 존 밖을 요구하므로 '제구 성공'의 의미가 뒤집히는데, 모델은 나머지 84%
# 에서 배운 규칙을 그대로 쓴다. 셀에 맥락을 붙이면 다중분류가 그 구간을
# **다른 문제로** 다룬다. P(성공)=sum(성공셀) 합산식은 그대로 성립한다.
#
# 맥락을 붙이면 셀이 배로 늘어 기본 임계 0.5% 에서 대부분 뭉개지므로 낮춰 준다.
# 기준선은 A100 의 AB_base(883.41) / 비교 대상은 DW_cell(+17.76).
cd ~/aimers || exit 1
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 5 --l2 10 --iters 3000 --refit-mult 1.5 --drop-f-pre 2022 --failmode-cells"
V="--val-season 2023 --test-season 2024"
S="--seeds 3,4,5,6,8,13"
$P src/train_gbdt2.py --model cat $B $V $S --fm-context strikes_before \
   --fm-min-share 0.002 --tag FX_str 2>&1 | grep -E "^\[cat|미학습|셀|!!"
echo FXSTR_DONE
$P src/train_gbdt2.py --model cat $B $V $S --fm-context balls_before,strikes_before \
   --fm-min-share 0.001 --tag FX_cnt 2>&1 | grep -E "^\[cat|미학습|셀|!!"
echo CELLCTX_DONE
