cd /mnt/c/aimers
P=/root/venv211/bin/python
# E117: 실력 추정을 **투수x볼카운트** 단위로. 신호감사 오라클이
#   투수 990.8 -> 투수x카운트 2740.9 로 최대인 축이다.
# 투수 수준 축(E116)은 두 설계 모두 편향제거 후 -0.2 로 포화 확인됐다.
# 입력에 te_pitcher_balls_before_strikes_before_* 를 물려주므로 skill 블록을
# TE 뒤로 옮겼다. 기준은 v11f.
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --no-refit --lr 0.01 --es 500 --depth 8 --l2 10"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -4; }
for S in 42 7 13 3 4 5; do
  R --tag P_pc_s$S --seed $S --feat-skill-pc
done
echo E117_DONE
