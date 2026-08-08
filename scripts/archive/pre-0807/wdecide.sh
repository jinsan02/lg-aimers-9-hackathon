cd /mnt/c/aimers
P=/root/venv211/bin/python
# W3 결정 실험 - 두 홀드아웃이 정반대(2024 t=+8.0 / 2023R t=-4.0)라 구조로 갈라야 한다.
#
# W3 는 "학습 마지막 시즌이 예측 시즌을 잘 대표할 때" 이득이다.
#   2024 홀드아웃 : 학습 마지막 2023(ABS 전) -> 2024(ABS 1년차). **경계를 가로지름**
#   제출          : 학습 마지막 2024(1년차)   -> 2025(2년차).     경계 없음
# F리그는 ABS 를 2023 에 도입했으므로
#   F 2019~23 학습(마지막 = 1년차) -> F 2024 예측(2년차) = **제출과 동일 구조**다.
# 이게 유일하게 구조가 맞는 대조군이다.
B="--league F --val-season 2024 --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --no-refit --lr 0.01 --es 500 --depth 8 --l2 10"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -3; }
for S in 42 7 13 3 5 9; do
  R --tag F_w1_s$S --seed $S
  R --tag F_w3_s$S --seed $S --val-last-weight 3
  R --tag F_w5_s$S --seed $S --val-last-weight 5
done
echo WDEC_DONE
