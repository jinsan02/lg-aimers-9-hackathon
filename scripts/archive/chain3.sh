#!/bin/bash
# chain.sh 교체본 — **중복 실험 제거**.
#
# season 제거(SC_nosea, SC_nosea2)를 빼야 한다. docs/experiment_guide.md:32 에
# "season 피처 **제거 금지** (drift 캘리브레이터 — E08에서 -580점 확인)" 로
# 이미 결론이 나 있다. 표면이 바뀌어도 -580 이 뒤집힐 크기가 아니고,
# 기전(season 이 드리프트 수준을 잡아주는 캘리브레이터)도 분명하다.
#
# 남기는 축들은 '이미 스윕했지만 **자기검증 표면**에서 골랐던' 것들이다.
# 표면 효과는 셀 계열에서 실제로 뒤집혔으므로(D +2.8 -> -18.0) 다시 볼 값은 있다.
# 다만 LightGBM·부분공간은 안 뒤집혔으니 기대는 낮게 잡는다.
cd ~/aimers || exit 1
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022"
V="--val-season 2023 --test-season 2024"
S="--seeds 3,4,5,6,8,13"
C="--lr 0.01 --es 500 --l2 10 --iters 3000 --refit-mult 1.5"

while pgrep -f 'tag RQ' > /dev/null; do sleep 60; done
echo "=== rm3 끝 → 표면(중복 제거본) $(date +%H:%M) ==="
run() { $P src/train_gbdt2.py --model cat $B $V $S "${@:2}" --tag "$1" 2>&1 \
        | grep -E "^\[cat|미학습|!!"; }
run SC_k40    --std-k 40
run SC_k120   --std-k 120
run SC_tek100 --te-k 100
run SC_d7     --depth 7

echo "=== 셀 동물원 $(date +%H:%M) ==="
BC="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc $C --drop-f-pre 2022"
cz() { $P src/train_gbdt2.py --model cat $BC $V $S "${@:2}" --tag "$1" 2>&1 \
       | grep -E "^\[cat|미학습|!!"; }
cz CZ_d4  --depth 4 --failmode-cells
cz CZ_d6  --depth 6 --failmode-cells
cz CZ_c16 --depth 5 --failmode-cells --fm-modes middle,ball,reverse,strike
cz CZ_ml  --depth 5 --fm-multilabel
echo CHAIN3_DONE
