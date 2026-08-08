cd /mnt/c/aimers
P=/root/venv211/bin/python
# k80 확정 기반 위에서 새 축 둘을 동시 검증.
#  form  = 최근경기 폼을 통산이 아니라 **당해 시즌** 대비로 (E108)
#  cross = std x TE dev 교차 (레버 H) - std 가 약했을 때 t=1.34 였으니 재시험
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --no-refit"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -3; }
for S in 42 7 13 3; do
  R --tag X_base_s$S  --seed $S --lr 0.02 --es 300 --depth 8 --l2 10
  R --tag X_form_s$S  --seed $S --lr 0.02 --es 300 --depth 8 --l2 10 --feat-form
  R --tag X_cross_s$S --seed $S --lr 0.02 --es 300 --depth 8 --l2 10 --feat-cross
done
echo NEWAX_DONE
