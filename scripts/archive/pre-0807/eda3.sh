cd /mnt/c/aimers
P=/root/venv211/bin/python
# 피처 감사(tools/feature_audit.py) 결과를 바로 시험한다. 기준은 v11f 설정.
#  DUP : 파생 중복 3열 제거 (std_pitchmix_n r=1.000 / dom_same_ball 0.9937 / dom_bat_runner -0.9886)
#  PREV: prev1/3/5 결측을 시즌 리그평균으로 채움 (결측률이 시즌마다 1.11~5.17%)
#  SEA : season 열 제거 - 드리프트 +2.119sd 로 1위인데 중요도 3위(8.32)다.
#        2025는 학습 범위 밖이라 트리가 2024 리프로 처리할 뿐인데, 그 의존이 해로울 수 있다.
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --no-refit --lr 0.01 --es 500 --depth 8 --l2 10"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -4; }
for S in 42 7 13 3 4; do
  R --tag P_base_s$S --seed $S
  R --tag P_dup_s$S  --seed $S --drop-cols std_pitchmix_n,dom_same_ball,dom_bat_runner
  R --tag P_prev_s$S --seed $S --fill-prev
  R --tag P_sea_s$S  --seed $S --drop-cols season
done
echo EDA3_DONE
