#!/bin/bash
# E155c — div.sh 재실행. 두 버그를 고친 뒤:
#   ① 셀 다중분류의 미학습 시즌 예측이 predict_proba[:,1] 이었다 (14열 중 1열).
#      성공 셀 합산으로 고침. DV_cell 이 BSS -1367 / rms 0.37 로 나온 원인.
#   ② LightGBM 은 test 프레임의 범주를 **학습 프레임 기준**으로 맞춰야 한다.
#      안 맞추면 코드가 밀려 조용히 틀린 예측이 나온다 (DV_lgb 가 죽은 원인).
# 부분공간(DV_sub7)은 이미 유효하게 나왔으므로 다시 안 돈다.
cd ~/aimers || exit 1
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --refit-mult 1.5 --drop-f-pre 2022"
V="--val-season 2023 --test-season 2024"
S="--seeds 3,4,5,6,8,13"
$P src/train_gbdt2.py --model cat $B $V $S --lr 0.01 --es 500 --depth 5 --l2 10 \
   --iters 3000 --failmode-cells --tag DW_cell 2>&1 | grep -E "^\[cat|미학습|!!"
$P src/train_gbdt2.py --model lgb $B $V $S --lr 0.03 --es 300 --nthreads 40 \
   --tag DW_lgb 2>&1 | grep -E "^\[lgb|미학습|!!|Error"
echo DIV2_DONE
