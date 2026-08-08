#!/bin/bash
# 셀 변형 동물원을 **미학습 표면**에서 다시 잰다 (E159).
#
# DW_cell 이 뒤집혔다: 자기검증 표면 D=+2.8 / 미학습 표면 D=-18.0.
# 즉 셀 멤버는 base 보다 **더 강한** 모델인데 우리는 보조로 취급해 왔다.
# 셀 기하 변형들(d4, d6, 16셀, 다중라벨)은 전부 **틀린 표면에서** 기각됐다.
# 같은 뒤집힘이 있는지 확인한다. base 기준선 RN1.5 = 878.83 (편향제거, 6시드).
#
# LightGBM 은 미학습 표면에서도 margin -7.1 로 확정 기각. 부분공간은 +0.60 으로 미미.
# 즉 '표면 오류' 는 **셀 계열에만** 해당했다 — 나머지는 다시 안 돈다.
cd ~/aimers || exit 1
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --l2 10 --iters 3000 --refit-mult 1.5 --drop-f-pre 2022"
V="--val-season 2023 --test-season 2024"
S="--seeds 3,4,5,6,8,13"
while pgrep -f 'bash chain.sh' > /dev/null; do sleep 60; done
echo "=== chain 끝 → 셀 동물원 시작 $(date +%H:%M) ==="
run() { $P src/train_gbdt2.py --model cat $B $V $S "${@:2}" --tag "$1" 2>&1 \
        | grep -E "^\[cat|미학습|!!"; }
run CZ_d4  --depth 4 --failmode-cells
run CZ_d6  --depth 6 --failmode-cells
run CZ_c16 --depth 5 --failmode-cells --fm-modes middle,ball,reverse,strike
run CZ_ml  --depth 5 --fm-multilabel
echo CELLZOO_DONE
