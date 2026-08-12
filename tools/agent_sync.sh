#!/bin/bash
# Claude <-> Codex handoff. Call once when you start and once when you finish.
#
# Two agents share this working tree. The only way to stop silent overwrites is to
# **commit before handing over and pull when starting**. On 08-08 files changed
# under us mid-session several times and we only noticed afterwards.
#
#   bash tools/agent_sync.sh start <claude|codex>
#   bash tools/agent_sync.sh end   <claude|codex> "what you did"
set -u
cd "$(dirname "$0")/.." || exit 1
MODE="${1:-}"; WHO="${2:-}"; MSG="${3:-}"

case "$MODE" in
start)
  echo "=== $WHO session start ==="
  git fetch -q origin 2>/dev/null || true
  # Pull whatever the other agent pushed. Halt on local changes -- never auto-merge.
  if [ -n "$(git status --porcelain)" ]; then
    echo "!! Uncommitted changes. The previous session never called end:"
    git status --short | head -20
    echo "   Inspect, then commit or revert, then start again."
    exit 2
  fi
  git pull -q --ff-only origin main 2>/dev/null || echo "  (no remote / conflict -- check manually)"
  echo "HEAD $(git log --oneline -1)"
  bash tools/ledger_sync.sh
  echo
  echo "Read: AGENTS.md -> EXPERIMENT.md -> HANDOFF.md"
  echo "Status: $(grep -A3 '^## Status' HANDOFF.md | grep -v '^##' | grep -v '^$' | head -1)"
  echo "Remote jobs running:"
  a=$(ssh -o ConnectTimeout=10 hsu-server \
        "pgrep -af train_gbdt2 | grep -oE 'tag [A-Za-z0-9_.]+'" 2>/dev/null)
  echo "  A100  ${a:-(none)}"
  b=$(ssh -o ConnectTimeout=10 desktop-4070 \
        'tasklist /fi "imagename eq python.exe" /fo table | find /c "python.exe"' \
        2>/dev/null | tr -d '\r\n ')
  echo "  4070  ${b:-?} python process(es)  (a parent+child pair means one job)"
  ;;
end)
  echo "=== $WHO session end ==="
  # No runners in the project root. scripts/README.md said so and it still
  # collected 30 files on 08-08 and 51 on 08-12. Rules alone did not hold, so
  # this is the enforcement point.
  stray=$(ls *.sh *.bat 2>/dev/null)
  if [ -n "$stray" ]; then
    echo "!! Runners in the project root. Move them to scripts/ and call again:"
    echo "$stray" | sed 's/^/   /'
    exit 2
  fi
  bash tools/ledger_sync.sh
  git add -A
  if [ -z "$(git diff --cached --name-only)" ]; then
    echo "No changes -- skipping commit"
  else
    git diff --cached --stat | tail -12
    git commit -q -m "[$WHO] ${MSG:-work}" || exit 1
    git push -q origin main 2>/dev/null && echo "push complete" \
      || echo "!! push failed -- check manually"
  fi
  echo
  echo "Confirm you updated Status / Next Agent in HANDOFF.md:"
  grep -E '^(## Current Agent|## Next Agent|## Status)' -A1 HANDOFF.md | head -9
  ;;
*)
  echo "usage: bash tools/agent_sync.sh {start|end} <claude|codex> [message]"
  exit 1
  ;;
esac
