cd /mnt/c/aimers
P=/root/venv211/bin/python
# lr 0.02->0.01 이 +8.5(t=4.63) 였으므로 아래쪽이 아직 안 닫혔다. 0.005 까지 내려본다.
# lr 0.005 는 최적 반복이 3000 을 넘길 수 있어 iters/es 를 키운다.
# RMSE 손실 = 0/1 타깃의 MSE = **Brier 직접 최적화**. 이전 t=0.80 기각은 std 이전 판정이다.
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --no-refit --depth 8 --l2 10"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -3; }
for S in 42 7 13 3 9; do
  R --tag L_lr01_s$S   --seed $S --lr 0.01  --es 500
  R --tag L_lr005_s$S  --seed $S --lr 0.005 --es 800 --iters 8000
  R --tag L_rmse_s$S   --seed $S --lr 0.01  --es 500 --loss RMSE
done
echo LOWLR_DONE
