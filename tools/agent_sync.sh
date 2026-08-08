#!/bin/bash
# Claude <-> Codex 교대 — 시작할 때와 끝낼 때 한 번씩 부른다.
#
# 두 에이전트가 같은 작업 트리를 만진다. 조용한 덮어쓰기를 막는 유일한 방법은
# **넘기기 전에 커밋하고, 시작할 때 당겨오는 것**이다. 08-08 에 파일이 세션 중
# 외부에서 바뀐 걸 나중에야 알아챈 적이 여러 번 있다.
#
#   bash tools/agent_sync.sh start <claude|codex>
#   bash tools/agent_sync.sh end   <claude|codex> "무엇을 했는가"
set -u
cd "$(dirname "$0")/.." || exit 1
MODE="${1:-}"; WHO="${2:-}"; MSG="${3:-}"

case "$MODE" in
start)
  echo "=== $WHO 작업 시작 ==="
  git fetch -q origin 2>/dev/null || true
  # 남이 올린 게 있으면 먼저 받는다. 로컬 변경이 있으면 멈춘다 — 자동 병합 금지.
  if [ -n "$(git status --porcelain)" ]; then
    echo "!! 커밋 안 된 변경이 있다. 이전 세션이 end 를 안 불렀다:"
    git status --short | head -20
    echo "   확인하고 커밋하거나 되돌린 뒤 다시 시작할 것."
    exit 2
  fi
  git pull -q --ff-only origin main 2>/dev/null || echo "  (원격 없음/충돌 — 수동 확인)"
  echo "HEAD $(git log --oneline -1)"
  bash tools/ledger_sync.sh
  echo
  echo "읽을 것: AGENTS.md -> EXPERIMENT.md -> HANDOFF.md"
  echo "현재 Status: $(grep -A1 '^## Status' HANDOFF.md | tail -1)"
  echo "실행 중인 원격 작업:"
  ssh hsu-server "pgrep -af train_gbdt2 | grep -oE 'tag [A-Za-z0-9_.]+'" 2>/dev/null \
    | sed 's/^/  A100 /' || echo "  A100 (접속 실패)"
  ssh desktop-4070 'tasklist /fi "imagename eq python.exe" /fo table | find /c "python.exe"' \
    2>/dev/null | tr -d '\r' | sed 's/^/  4070 python 프로세스 /' || echo "  4070 (접속 실패)"
  ;;
end)
  echo "=== $WHO 작업 종료 ==="
  bash tools/ledger_sync.sh
  git add -A
  if [ -z "$(git diff --cached --name-only)" ]; then
    echo "변경 없음 — 커밋 생략"
  else
    git diff --cached --stat | tail -12
    git commit -q -m "[$WHO] ${MSG:-작업}" || exit 1
    git push -q origin main 2>/dev/null && echo "push 완료" \
      || echo "!! push 실패 — 수동 확인"
  fi
  echo
  echo "HANDOFF.md 의 Status / Next Agent 를 갱신했는지 확인할 것:"
  grep -E '^(## Current Agent|## Next Agent|## Status)' -A1 HANDOFF.md | head -9
  ;;
*)
  echo "사용법: bash tools/agent_sync.sh {start|end} <claude|codex> [메시지]"
  exit 1
  ;;
esac
