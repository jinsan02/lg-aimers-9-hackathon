#!/bin/bash
# E165 재실행 — 교사는 이미 만들어졌다(T_self 2076.0 / T_seq 2115.1).
# 학생만 다시 돌린다. 앞선 실패 원인: 재학습 경로가 loss_function="Logloss" 로
# 하드코딩돼 있어 [0,1] 실수 타깃을 거부했다(검증 모델은 args.loss 를 따라가
# CrossEntropy 로 넘어갔고 재학습만 죽었다).
#
# 그리고 로그 필터를 없앤다 — 앞서 `grep -E "^\[cat|..."` 가 트레이스백을
# 통째로 삼켜서 "완료"로 보였다.
cd ~/aimers || exit 1
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022"
V="--val-season 2023 --test-season 2024"
S="--seeds 3,4,5,6,8,13"
$P src/train_gbdt2.py --model cat $B $V $S --soft-target ./out/teacher_T_self.npz --tag DS_self
echo DS_SELF_DONE
$P src/train_gbdt2.py --model cat $B $V $S --soft-target ./out/teacher_T_seq.npz --tag DS_seq
echo DISTILL2_DONE
