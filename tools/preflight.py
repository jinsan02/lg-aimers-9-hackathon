"""원격 배치 실행 전 필수 점검 — 08-06 사고 재발 방지.

사고 이력:
  1) argparse help 문자열에 줄바꿈이 들어가 SyntaxError → A100/4070 배치가 통째로 죽음.
     그런데 실행 스크립트가 `| grep -E "^\\[cat"` 로 stderr를 걸러서 **에러가 안 보였다.**
  2) 원격에 ./model 디렉토리가 없어 저장 단계에서 FileNotFoundError → 역시 grep에 가려짐.

규칙:
  - 원격에 scp 하기 전 반드시 이 스크립트로 구문 검사.
  - 배치 스크립트에서 stderr를 grep으로 거르지 말 것. `tail -3` 처럼 전부 남길 것.
  - 배치 전 1개 설정을 --iters 20 으로 짧게 돌려 끝까지 통과하는지 확인.

사용: python tools/preflight.py src/train_gbdt2.py src/target_enc.py ...
"""

import ast
import sys


def main(paths):
    bad = 0
    for p in paths:
        try:
            src = open(p, encoding="utf-8").read()
            ast.parse(src, filename=p)
        except SyntaxError as e:
            print(f"  XX {p}:{e.lineno}  {e.msg}")
            bad += 1
            continue
        except OSError as e:
            print(f"  XX {p}  {e}")
            bad += 1
            continue
        print(f"  OK {p}  ({len(src.splitlines())}줄)")
    if bad:
        print(f"\n구문 오류 {bad}개 - scp 금지")
        return 1
    print("\n전부 통과 - 배포 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["src/train_gbdt2.py", "src/target_enc.py",
                                   "src/features.py", "src/script_blend_v5.py"]))
