#!/bin/bash
# 노트북을 꺼도 A100 큐가 계속 돌도록 하는 연결 고리.
# div2 -> rm3 -> 미학습 표면 축 재판정. setsid 로 띄워 PPID=1 이 되게 한다.
cd ~/aimers || exit 1
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022"
V="--val-season 2023 --test-season 2024"
S="--seeds 3,4,5,6,8,13"

while pgrep -f 'bash div2.sh' > /dev/null; do sleep 60; done
echo "=== div2 끝 → rm3 시작 $(date +%H:%M) ==="
bash rm3.sh > out/rm3.log 2>&1
echo "=== rm3 끝 → 표면 재판정 시작 $(date +%H:%M) ==="

# 어젯밤 죽인 surf_a 두 축 + surf_b 네 축. 기준선 RN1.5 = 868.56 (6시드).
# season 은 2025 가 학습 범위 밖이라 트리가 무조건 2024 리프로 보낸다 —
# 드리프트 편향의 직접 원인인데 한 번도 안 건드렸다.
run() { $P src/train_gbdt2.py --model cat $B $V $S "${@:2}" --tag "$1" 2>&1 \
        | grep -E "^\[cat|미학습|!!"; }
run SC_nosea  --drop-cols season
run SC_nosea2 --drop-cols season,season_progress
run SC_k40    --std-k 40
run SC_k120   --std-k 120
run SC_tek100 --te-k 100
run SC_d7     --depth 7
echo CHAIN_DONE
