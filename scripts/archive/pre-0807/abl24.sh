cd ~/aimers
P=~/venv451/bin/python
# 레버 영향성 평가 - 최종 스택에서 **하나씩 빼는** leave-one-out 절제.
# 지금까지의 판정은 "현재 설정 위에 얹는 한계 효과"였다. 중복 검사는 되지만
# **채택된 레버끼리의 중첩**은 못 본다. 오늘 교호작용이 5번 나왔으므로 필요하다.
# 같은 구성을 2024 와 2023(R리그 전용) 두 홀드아웃에서 돌려 부호 일치를 본다.
VAL=""
COM="--no-refit --depth 8 --l2 10"
V2="--feat-v2"; TE="--te p,pc,ph,b,pi"; DEV="--te-dev"; DOM="--feat-domain"
STD="--feat-std --std-k 80 --std-to-prior --std-season-prior"
LR="--lr 0.01 --es 500"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $VAL $COM "$@" 2>&1 | tail -3; }
for S in 42 7 13 3 5; do
  R --tag A24_full_s$S   --seed $S $V2 $TE $DEV $STD $DOM $LR
  R --tag A24_nov2_s$S   --seed $S     $TE $DEV $STD $DOM $LR
  R --tag A24_note_s$S   --seed $S $V2          $STD $DOM $LR
  R --tag A24_nodev_s$S  --seed $S $V2 $TE      $STD $DOM $LR
  R --tag A24_nostd_s$S  --seed $S $V2 $TE $DEV           $LR
  R --tag A24_k30_s$S    --seed $S $V2 $TE $DEV --feat-std --std-k 30 --std-to-prior --std-season-prior $DOM $LR
  R --tag A24_career_s$S --seed $S $V2 $TE $DEV --feat-std --std-k 80 --std-season-prior $DOM $LR
  R --tag A24_nosp_s$S   --seed $S $V2 $TE $DEV --feat-std --std-k 80 --std-to-prior $DOM $LR
  R --tag A24_nodom_s$S  --seed $S $V2 $TE $DEV $STD      $LR
  R --tag A24_lr02_s$S   --seed $S $V2 $TE $DEV $STD $DOM --lr 0.02 --es 300
  R --tag A24_w3_s$S     --seed $S $V2 $TE $DEV $STD $DOM $LR --val-last-weight 3
done
echo A24_DONE
