cd /mnt/c/aimers
P=/root/venv211/bin/python
# E116 을 2023 R리그 홀드아웃에서도 확인 (체제 변화가 없는 구간 = 2025 와 같은 구조)
B="--drop-f-pre 2030 --val-season 2023 --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --no-refit --lr 0.01 --es 500 --depth 8 --l2 10"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -4; }
for S in 42 7 13 3; do
  R --tag T3_base_s$S  --seed $S
  R --tag T3_skill_s$S --seed $S --feat-skill
done
echo E116B_DONE
