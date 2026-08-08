#!/bin/bash
# GPU 작업이 있을 때 80코어를 어떻게 나눌 것인가 (E151).
#
# A100 서버 토폴로지 (lscpu / nvidia-smi topo -m):
#   물리 20코어 x 2소켓 x HT = 논리 80
#   node0: 0-19, 40-59     <- **GPU 가 여기 붙어 있다**
#   node1: 20-39, 60-79
#
# 앞서 LightGBM num_threads=0(=77스레드)로 돌렸다가 동시 실행 중이던 CatBoost
# GPU 작업이 3.2배 느려졌다. GPU 학습도 데이터 공급에 CPU 를 쓴다.
#
# 각 작업을 단독 실행 시간으로 나눈 '속도비'를 재고, 두 속도비의 합으로
# 총 처리량을 본다 (간섭이 없으면 2.00).
cd ~/aimers || exit 1
P=~/venv451/bin/python
G="--device GPU --iters 1000"      # 단독 61s
C="--device CPU --iters 300"       # 단독 약 62s (40스레드)
N0="0-19,40-59"                    # GPU 소켓
N1="20-39,60-79"

run2() {   # $1=라벨  $2=GPU 명령  $3=CPU 명령
  echo "### $1"
  eval "$2" > /tmp/g.out 2>&1 &
  gp=$!
  eval "$3" > /tmp/c.out 2>&1 &
  cp=$!
  wait $gp $cp
  grep -h RESULT /tmp/g.out /tmp/c.out
}

echo "===== 단독 기준선 ====="
$P tools/bench_job.py $G --threads 8  --label base_gpu_t8   2>&1 | grep RESULT
$P tools/bench_job.py $G --threads 16 --label base_gpu_t16  2>&1 | grep RESULT
taskset -c $N1 $P tools/bench_job.py $C --threads 40 --label base_cpu_t40 2>&1 | grep RESULT
$P tools/bench_job.py $C --threads 72 --label base_cpu_t72 2>&1 | grep RESULT

echo; echo "===== 동시 실행 ====="
run2 "①핀없음 GPU8 + CPU72 (재앙 재현)" \
  "$P tools/bench_job.py $G --threads 8 --label g_free8" \
  "$P tools/bench_job.py $C --threads 72 --label c_free72"

run2 "②핀없음 GPU8 + CPU40" \
  "$P tools/bench_job.py $G --threads 8 --label g_free8b" \
  "$P tools/bench_job.py $C --threads 40 --label c_free40"

run2 "③NUMA분리 GPU8@node0 + CPU40@node1" \
  "taskset -c $N0 $P tools/bench_job.py $G --threads 8 --label g_n0_t8" \
  "taskset -c $N1 $P tools/bench_job.py $C --threads 40 --label c_n1_t40"

run2 "④GPU16@0-9,40-49 + CPU60@나머지" \
  "taskset -c 0-9,40-49 $P tools/bench_job.py $G --threads 16 --label g_p16" \
  "taskset -c 10-39,50-79 $P tools/bench_job.py $C --threads 60 --label c_p60"

run2 "⑤GPU8@0-3,40-43 + CPU72@나머지" \
  "taskset -c 0-3,40-43 $P tools/bench_job.py $G --threads 8 --label g_p8" \
  "taskset -c 4-39,44-79 $P tools/bench_job.py $C --threads 72 --label c_p72"

echo BENCH_DONE
