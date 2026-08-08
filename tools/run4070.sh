#!/bin/bash
# 4070(Windows)에서 ssh 연결이 끊겨도 살아남는 작업을 **안전하게** 띄운다.
#
# 왜 필요한가 (실제로 당한 것):
#   - WSL tmux 안에서는 Windows exe interop 이 깨진다(UtilAcceptVsock 110) -> 즉사
#   - setsid nohup / start /b / Start-Process 전부 ssh 종료와 함께 죽는다
#   - 남은 방법이 schtasks 인데, `/st 23:5x` 같은 **가까운 시각**을 더미로 쓰면
#     그 시각에 **전부 자동 재발화**한다. 2026-08-07 23:5x 에 9개가 동시에 떠서
#     한 GPU 를 9등분하고 서로의 산출물을 덮어썼다.
#
# 그래서: 발화 시각을 먼 미래로 박고, 실행은 /run 으로만 한다. 끝나면 지운다.
#
# 사용: bash tools/run4070.sh <이름> <배치파일경로(Windows)> <로그경로(Windows)>
#   예: bash tools/run4070.sh V17 C:\\aimers\\v17.bat C:\\aimers\\out\\v17.log
set -u
NAME="Aimers$1"
BAT="$2"
LOG="$3"
H=desktop-4070

ssh $H "schtasks /delete /tn $NAME /f" > /dev/null 2>&1
ssh $H "schtasks /create /tn $NAME /tr \"cmd /c $BAT > $LOG 2>&1\" /sc once /sd 2099/01/01 /st 03:00 /f" \
  | tail -1
ssh $H "schtasks /run /tn $NAME" | tail -1
echo "띄움: $NAME  (다음 자동 발화 2099-01-01 — 사실상 없음)"
echo "정리: ssh $H 'schtasks /delete /tn $NAME /f'"
