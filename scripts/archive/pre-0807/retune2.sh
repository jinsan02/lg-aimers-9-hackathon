cd /mnt/c/aimers
P=/root/venv211/bin/python
# k60 이 단조 최고 -> 최적점이 60 바깥인지 확장. 동시에 k 가 커졌을 때
# 수축 목표(prior vs career)가 여전히 prior 가 맞는지, lr01 과 결합되는지 확인.
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-season-prior --no-refit"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -3; }
for S in 42 7 13; do
  R --tag U_k100_s$S  --seed $S --lr 0.02 --es 300 --depth 8 --l2 10 --std-k 100 --std-to-prior
  R --tag U_k200_s$S  --seed $S --lr 0.02 --es 300 --depth 8 --l2 10 --std-k 200 --std-to-prior
  R --tag U_k400_s$S  --seed $S --lr 0.02 --es 300 --depth 8 --l2 10 --std-k 400 --std-to-prior
  R --tag U_k60car_s$S --seed $S --lr 0.02 --es 300 --depth 8 --l2 10 --std-k 60
  R --tag U_k60lr_s$S --seed $S --lr 0.01 --es 500 --depth 8 --l2 10 --std-k 60 --std-to-prior
done
echo RETUNE2_DONE
