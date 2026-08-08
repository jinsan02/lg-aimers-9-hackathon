cd /mnt/c/aimers
P=/root/venv211/bin/python
# E109: 드리프트가 계단이 아니라 기울기(E105)라면 '마지막 시즌만 x3' 대신
# 전 시즌 지수감쇠가 형태상 맞다. W3 계단형과 직접 비교한다.
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --no-refit --lr 0.02 --es 300 --depth 8 --l2 10"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -3; }
for S in 42 7 13; do
  R --tag Y_g100_s$S --seed $S
  R --tag Y_g090_s$S --seed $S --season-decay 0.9
  R --tag Y_g080_s$S --seed $S --season-decay 0.8
  R --tag Y_g070_s$S --seed $S --season-decay 0.7
  R --tag Y_w3_s$S   --seed $S --val-last-weight 3
done
for S in 42 7 13 3; do
  R --tag Z_base_s$S --seed $S
  R --tag Z_dom_s$S  --seed $S --feat-domain
  R --tag Z_rat_s$S  --seed $S --std-ratio
done
echo DECAY_DONE
