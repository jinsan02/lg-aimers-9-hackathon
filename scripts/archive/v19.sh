#!/bin/bash
# v19 — std-k 80 -> 40 을 **제출 구조**에서 검증하며 동시에 빌드한다.
#
# SC_k40 이 미학습 표면에서 +15.05 (t=+11.88) 로 채택됐다. 단조(40 > 80 > 200)라
# 방향은 믿을 만하다. 그런데 그 측정은 `--drop-f-pre 2022` 로 했고 **제출은 안 쓴다** —
# v16 을 -6.15 로 만든 바로 그 불일치다. 그래서 여기서는 제출과 완전히 같은 구조
# (val 2024, F 유지, refit 1.5) 로 돌린다.
#
# 판정 게이트: v19f 의 val2024 원점수 평균이 v14f 의 **921.0** 을 넘어야 채택.
# 못 넘으면 std-k 이득이 제출 구조로 안 넘어온 것이니 제출하지 않는다.
cd ~/aimers || exit 1
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 40 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --l2 10 --refit-mult 1.5"
while pgrep -f 'bash chain3.sh' > /dev/null; do sleep 60; done
echo "=== chain3 끝 → v19 빌드 $(date +%H:%M) ==="
$P src/train_gbdt2.py --model cat $B --depth 8 --tag v19f --seeds 42,7,13,3,4,5,6,8 2>&1 | grep -E "^\[cat|saved|!!"
echo V19F_DONE
$P src/train_gbdt2.py --model cat $B --depth 5 --iters 3000 --failmode-cells --tag v19c --seeds 42,7,13,3,4,5 2>&1 | grep -E "^\[cat|saved|!!"
echo V19_DONE
