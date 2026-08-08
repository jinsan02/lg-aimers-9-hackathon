#!/bin/bash
# E165 증류 — 교사 두 종류를 만들고 각각 학생을 학습해 비교한다.
#   T_seq  : 직전 투구 결과 포함 (순차 정보, train 전용)
#   T_self : 합법 피처만 = 자기증류 (순수 잡음 감소 효과 분리)
# 학생은 둘 다 **합법 피처만** 쓰고 타깃만 교사 확률로 바꾼다.
# 기준선 AB_base 883.41 (A100 6시드). 판정은 단독 점수 + margin 둘 다.
cd ~/aimers || exit 1
P=~/venv451/bin/python
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022"
V="--val-season 2023 --test-season 2024"
S="--seeds 3,4,5,6,8,13"
$P src/teacher.py --tag T_self 2>&1 | grep -E "fold|OOF|저장|교사"
$P src/train_gbdt2.py --model cat $B $V $S --soft-target ./out/teacher_T_self.npz \
   --tag DS_self 2>&1 | grep -E "^\[cat|미학습|증류|!!"
echo DS_SELF_DONE
$P src/teacher.py --tag T_seq --prev 2>&1 | grep -E "fold|OOF|저장|교사"
$P src/train_gbdt2.py --model cat $B $V $S --soft-target ./out/teacher_T_seq.npz \
   --tag DS_seq 2>&1 | grep -E "^\[cat|미학습|증류|!!"
echo DISTILL_DONE
