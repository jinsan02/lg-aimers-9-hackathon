cd /mnt/c/aimers
P=/root/venv211/bin/python
# E120: 2단 잔차 구조. 1단(선형 회귀)이 매끄러운 실력 수준, 2단(트리)이 잔차.
# 근거: 최적 수축은 매끄러운 가중평균이라 트리가 계단으로만 근사한다
#       (실측 - 남은시즌 성공률 설명력 선형 59.0% vs GBDT 46.5%).
# 기준은 v11f (같은 검증단계 설정).
B="--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --no-refit --lr 0.01 --es 500 --depth 8 --l2 10"
SE="42,7,13,3,4,5"
# (a) 잔차 구조 (기준선 = 투수x카운트 실력 추정)
$P src/train_gbdt2.py --model cat $B --tag S1_resid --seeds $SE --resid-col skill_pc_hat 2>&1 | tail -20
# (b) 대조군: 같은 설정에서 RMSE 손실만 (잔차 없이) - 손실 변경 효과를 분리
$P src/train_gbdt2.py --model cat $B --tag S1_rmse  --seeds $SE --loss RMSE 2>&1 | tail -20
echo E120_DONE
