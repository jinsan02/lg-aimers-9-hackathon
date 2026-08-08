cd /mnt/c/aimers
P=/root/venv211/bin/python
# E111 확정용 시드 추가. rat(E110) 은 -38.5 로 확정 기각이라 뺀다.
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --no-refit --lr 0.02 --es 300 --depth 8 --l2 10"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -3; }
for S in 9 11 17 23 2 10; do
  R --tag Z_base_s$S --seed $S
  R --tag Z_dom_s$S  --seed $S --feat-domain
done
echo ZMORE_DONE
