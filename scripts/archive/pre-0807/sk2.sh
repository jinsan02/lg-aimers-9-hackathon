cd /mnt/c/aimers
P=/root/venv211/bin/python
# E116b: skill.py 설계행렬 보강 후 재시도.
# horizon 분석에서 이 설계가 46.1%, 처음 59.0% 를 낸 설계는 원시 n 과 season 을
# 썼다. 그 둘을 복원했다. 기준은 v11f.
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --no-refit --lr 0.01 --es 500 --depth 8 --l2 10"
R() { echo "### $*"; $P src/train_gbdt2.py --model cat $B "$@" 2>&1 | tail -3; }
for S in 42 7 13 3 4 5; do
  R --tag U_skill_s$S --seed $S --feat-skill
done
echo SK2_DONE
