cd /mnt/c/aimers
P=/root/venv211/bin/python
# E112: 지금 std-k 80 은 10개 컬럼 전부에 같은 값이다. 성공률은 이항(분산 0.25)이라
# 표본 잡음이 크고, 구종배합비는 투수마다 안정적이라 덜 수축해야 맞다.
# 기준은 이미 있는 Z_base (k80 전부, no-refit) 를 그대로 쓴다.
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --no-refit --lr 0.02 --es 300 --depth 8 --l2 10"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -3; }
for S in 42 7 13 3 9; do
  R --tag K_mix20_s$S  --seed $S --std-k-mix 20
  R --tag K_mix300_s$S --seed $S --std-k-mix 300
  R --tag K_bat20_s$S  --seed $S --std-k-bat 20
  R --tag K_bat300_s$S --seed $S --std-k-bat 300
done
echo KGRP_DONE
