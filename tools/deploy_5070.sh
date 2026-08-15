#!/usr/bin/env bash
# Push source to a remote worker and stamp which revision it is.
#
# The remote C:\aimers is not a git checkout -- it is populated by scp -- so
# `git rev-parse` there returns nothing and every lineage record written on it
# carried commit "". The whole B0-JL baseline landed that way (2026-08-13).
# Stamping `.deployed_commit` lets tools/lineage.py name the source anyway.
#
# Usage: bash tools/deploy_5070.sh [host]     (default desktop-5070)
set -euo pipefail

H="${1:-desktop-5070}"
R="C:/aimers"

cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain src tools)" ]; then
  echo "!! src/ or tools/ is dirty -- the stamped commit would be a lie." >&2
  git status --short src tools >&2
  exit 1
fi
C=$(git rev-parse HEAD)

ssh "$H" "if not exist $R\\src mkdir $R\\src & if not exist $R\\tools mkdir $R\\tools & if not exist $R\\out mkdir $R\\out"
scp -q src/*.py   "$H:$R/src/"
scp -q tools/*.py "$H:$R/tools/"
# cmd redirection needs a backslash path; $R is the scp (forward-slash) form.
W="${R//\//\\}"
ssh "$H" "cmd /c echo $C> $W\\.deployed_commit"
# ...and append, never overwrite. `.deployed_commit` holds only the latest
# revision, so a later deploy erases which code produced an earlier artifact.
# That happened on 2026-08-15: the RANK16 scout was written at 18:02 and the
# stamp was overwritten by the next deploy, leaving its source commit
# recoverable only by bounding it between commit timestamps. The history line
# lets an artifact's mtime be matched to a deployment.
T=$(date '+%Y-%m-%d %H:%M:%S')
ssh "$H" "cmd /c echo $T $C>> $W\\.deploy_history"

echo "deployed ${C:0:8} to $H at $T"
ssh "$H" "cmd /c type $W\\.deployed_commit" 2>/dev/null
