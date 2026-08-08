#!/bin/bash
# E155 — **기각했던 다양성 멤버들을 올바른 표면에서 다시 잰다.**
#
# 우리는 margin = 400,320 x rms^2 - D 로 LightGBM·NN·부분공간·선형 멤버를 전부
# 기각했다. 그런데 D(성능 격차)를 **2024 자기검증 표면**에서 쟀다. 그 표면에서는
# 모든 모델이 2024 로 조기종료됐으므로, 베이스가 그 시즌의 특이성에 맞춘 이점이
# D 에 그대로 들어간다. 시즌이 바뀌면 모델 간 순위는 압축되는 게 정상이다.
#
# D 가 줄면 margin 은 그만큼 그대로 커진다:
#   LightGBM  rms 0.012 -> Dmax 57.6.  D 9.8 이면 margin 47.8 (이득 +9.9)
#                                       D 5.0 이면 margin 52.6 (이득 +12.0)
#   즉 D 를 잘못 쟀다면 **다양성 경로 전체를 잘못 닫은 것**이다.
#
# 여기서는 val 2023 -> 미학습 2024 (배치와 같은 구조) 에서 다시 잰다.
# 기준선 RN1.5 = 868.56 (6시드).
cd ~/aimers || exit 1
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --refit-mult 1.5 --drop-f-pre 2022"
V="--val-season 2023 --test-season 2024"
S="--seeds 3,4,5,6,8,13"

# ① LightGBM (leaf-wise) — CatBoost 의 oblivious tree 와 결정면이 계통적으로 다르다
$P src/train_gbdt2.py --model lgb $B $V $S --lr 0.03 --es 300 --nthreads 40 \
   --tag DV_lgb 2>&1 | grep -E "^\[lgb|미학습"
# ② 피처 부분공간 0.7 — 같은 알고리즘, 다른 입력
$P src/train_gbdt2.py --model cat $B $V $S --lr 0.01 --es 500 --depth 8 --l2 10 \
   --feat-frac 0.7 --tag DV_sub7 2>&1 | grep -E "^\[cat|미학습"
# ③ 셀 다중분류 depth5 — 이미 채택한 멤버. 올바른 표면에서 margin 을 재확인한다
$P src/train_gbdt2.py --model cat $B $V $S --lr 0.01 --es 500 --depth 5 --l2 10 \
   --iters 3000 --failmode-cells --tag DV_cell 2>&1 | grep -E "^\[cat|미학습"
echo DIV_DONE
