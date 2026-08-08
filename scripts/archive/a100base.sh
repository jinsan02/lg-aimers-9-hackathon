#!/bin/bash
# A100 기준선 — 이게 없으면 A100 에서 잰 것을 하나도 못 쓴다.
#
# 오늘 밤 std-k 40 이 A100 에서 +15.05, 4070 에서 +4.27 로 나왔다. 같은 설정·같은
# 시드인데 11점이 머신 차이다. 기준선 RN1.5 는 4070 산이라, A100 에서 잰 모든
# 수치(SC_* 4축, DW_cell, DW_lgb, DV_sub7, CZ_* 4종)는 **머신 간 비교**였다.
#
# RN1.5 와 완전히 같은 설정을 A100 에서 6시드 돌려 A100 전용 기준선을 만든다.
cd ~/aimers || exit 1
P=~/venv451/bin/python
while pgrep -f 'bash chain3.sh' > /dev/null; do sleep 30; done
echo "=== chain3 끝 → A100 기준선 $(date +%H:%M) ==="
$P src/train_gbdt2.py --model cat \
  --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior \
  --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 \
  --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 \
  --val-season 2023 --test-season 2024 --tag AB_base --seeds 3,4,5,6,8,13 \
  2>&1 | grep -E "^\[cat|미학습|!!"
echo A100BASE_DONE
