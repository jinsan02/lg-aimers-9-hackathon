#!/bin/bash
# 두 머신의 LEDGER.tsv 를 노트북으로 모은다.
#
# 원장은 각 머신에서 학습이 돌 때 그 머신에 쌓인다. AGENTS.md 가 원장을
# **Source of Truth 2순위**로 지정했으므로 작업 디렉토리에 합본이 있어야
# Claude·Codex 둘 다 읽을 수 있다.
#
# 08-08 에 std-k 40 이 A100 +15.05 / 4070 +4.27 로 갈렸을 때, 원장의 hostname
# 컬럼이 그걸 잡아냈다. 머신별로 흩어져 있으면 그 대조를 못 한다.
#
# 실행: bash tools/ledger_sync.sh
set -u
cd "$(dirname "$0")/.." || exit 1
T=$(mktemp -d)
# Pull the active direct host first.  Without a connection timeout, two offline
# upstream hosts can consume the caller's whole timeout before the 5070 is ever
# reached, leaving a plausible-looking but stale merged ledger.
scp -q -o ConnectTimeout=5 desktop-5070:C:/aimers/LEDGER.tsv  "$T/5070.tsv" 2>/dev/null || true
scp -q -o ConnectTimeout=5 hsu-server:~/aimers/LEDGER.tsv     "$T/a100.tsv" 2>/dev/null || true
scp -q -o ConnectTimeout=5 desktop-4070:C:/aimers/LEDGER.tsv  "$T/4070.tsv" 2>/dev/null || true
cat LEDGER.tsv "$T"/*.tsv 2>/dev/null | grep -v '^$' | sort -u > "$T/merged.tsv"
mv "$T/merged.tsv" LEDGER.tsv
rm -rf "$T"
echo "LEDGER.tsv $(wc -l < LEDGER.tsv)행"
cut -f2 LEDGER.tsv | sort | uniq -c | sed 's/^/  /'
