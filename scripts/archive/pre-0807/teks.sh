cd /mnt/c/aimers
P=/root/venv211/bin/python
# TE 키셋/강도는 std 도입 **전에** 정한 값이다 (E107·E85 와 같은 상황).
# k80 std 위에서 재탐색한다. 기준은 이미 있는 Z_base.
B="--feat-v2 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --no-refit --lr 0.02 --es 300 --depth 8 --l2 10"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -3; }
for S in 42 7 13 3 9; do
  R --tag T_pbs_s$S  --seed $S --te p,pc,ph,b,pi,pbs
  R --tag T_bh_s$S   --seed $S --te p,pc,ph,b,pi,bh
  R --tag T_bc_s$S   --seed $S --te p,pc,ph,b,pi,bc
  R --tag T_nopi_s$S --seed $S --te p,pc,ph,b
  R --tag T_k25_s$S  --seed $S --te p,pc,ph,b,pi --te-k 25
  R --tag T_k100_s$S --seed $S --te p,pc,ph,b,pi --te-k 100
done
echo TEKS_DONE
