cd /mnt/c/aimers
P=/root/venv211/bin/python
# 2023 보조 홀드아웃 (2019~22 학습 -> 2023 예측).
# LB 피드백이 없어졌으므로, 채택 레버가 2024 특유의 것인지 진짜인지 여기서 가린다.
# 2023 은 R리그 ABS 도입 **이전**이라 체제가 다르다 - 절대값이 아니라
# **레버의 부호와 상대 크기가 재현되는가**만 본다.
B="--val-season 2023 --feat-v2 --te p,pc,ph,b,pi --te-dev --no-refit --depth 8 --l2 10"
S1="--feat-std --std-to-prior --std-season-prior"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -3; }
for S in 42 7 13; do
  R --tag V3_full_s$S  --seed $S $S1 --std-k 80 --feat-domain --lr 0.01 --es 500
  R --tag V3_nostd_s$S --seed $S                              --lr 0.01 --es 500
  R --tag V3_k30_s$S   --seed $S $S1 --std-k 30 --feat-domain --lr 0.01 --es 500
  R --tag V3_lr02_s$S  --seed $S $S1 --std-k 80 --feat-domain --lr 0.02 --es 300
  R --tag V3_nodom_s$S --seed $S $S1 --std-k 80               --lr 0.01 --es 500
  R --tag V3_w3_s$S    --seed $S $S1 --std-k 80 --feat-domain --lr 0.01 --es 500 --val-last-weight 3
done
echo V2023_DONE
