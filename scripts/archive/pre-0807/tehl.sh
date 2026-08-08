cd /mnt/c/aimers
P=/root/venv211/bin/python
# TE 재탐색 - LOO 가 드러낸 방향.
#   레버 절제에서 TE 전체가 2024 +18.0 vs **2023R +37.6** 로 갈렸다.
#   2024 는 ABS 도입 연도라 통산 정보가 낡아 TE 가 죽고 std 가 폭등한다.
#   2025 는 2024 와 체제가 같으므로 2023R 비율에 가깝다 - 즉 우리는 지금까지
#   **TE 를 과소평가하는 홀드아웃에서 튜닝**해 왔다.
# 그래서 TE 에 민감한 2023R 에서 미시험 축을 먼저 본다.
#   halflife : TE 누적의 시즌 반감기. 지금은 2019~ 균등 누적이라 옛 체제가 그대로 섞인다.
#   flat     : strat=False (기대값을 시즌 전체 평균으로) - 한 번도 절제 안 했다.
B="--drop-f-pre 2030 --val-season 2023 --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --no-refit --lr 0.01 --es 500 --depth 8 --l2 10"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -3; }
for S in 42 7 13 3; do
  R --tag H3_base_s$S --seed $S
  R --tag H3_hl1_s$S  --seed $S --te-halflife 1
  R --tag H3_hl2_s$S  --seed $S --te-halflife 2
  R --tag H3_hl4_s$S  --seed $S --te-halflife 4
  R --tag H3_flat_s$S --seed $S --te-flat
done
echo TEHL_DONE
echo REF_SKIPPED
