"""실험 전 중복·금지 검사. **실행 전에 반드시 통과시킬 것.**

2026-08-07 에 하루 30개 실험을 돌리고 기록을 하나도 안 남겼다. 그래서 `season`
제거를 "한 번도 안 해본 축"이라 부르며 큐에 걸었다 — E08 에서 −580 으로 끝난
질문이었다. 사람의 기억(과 대화 압축)에 의존하는 걸 그만두기 위한 도구다.

하는 일:
  ① docs/SETTLED.md 에서 BANNED/CLOSED 플래그와 겹치는지 본다 (BANNED 면 종료코드 2)
  ② LEDGER.tsv 에서 **같은 플래그 조합을 이미 돌렸는지** 찾아 결과를 보여준다
  ③ 제출 설정(SUBMIT_FLAGS)과 다른 점을 나열한다 — 후처리 상수를 여기서 뽑으면
     안 되는 실행인지 스스로 보이게

실행:
    python tools/precheck.py --model cat --drop-cols season --val-season 2023 ...
    python tools/precheck.py --file some.sh        # 스크립트 안의 모든 명령 검사
"""

import os
import re
import sys

# SETTLED.md 는 한글 + U+2212(−) 를 쓴다. Windows 콘솔(cp949)에 그대로 찍으면
# UnicodeEncodeError 로 죽는다 — 금지 축을 알려주려는 도구가 크래시하면 그냥
# 무시하고 실험을 돌리게 된다. 인코딩할 수 없는 글자는 대체 문자로 흘린다.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTLED = os.path.join(ROOT, "docs", "SETTLED.md")
LEDGER = os.path.join(ROOT, "LEDGER.tsv")

# 지금 제출본(v15)이 쓰는 학습 설정. 후처리 상수를 재는 실행은 이것과 같아야 한다.
SUBMIT_FLAGS = {
    "--feat-v2", "--te-dev", "--feat-std", "--std-to-prior",
    "--std-season-prior", "--feat-domain", "--feat-skill-pc",
}
SUBMIT_VALUES = {"--te": "p,pc,ph,b,pi", "--std-k": "80", "--lr": "0.01",
                 "--es": "500", "--depth": "8", "--l2": "10",
                 "--refit-mult": "1.5"}


def parse_settled():
    rows = []
    if not os.path.exists(SETTLED):
        return rows
    with open(SETTLED, encoding="utf-8") as f:
        for line in f:
            if line.startswith("FLAG "):
                parts = [p.strip() for p in line[5:].split("|")]
                if len(parts) >= 3:
                    rows.append(parts)      # [패턴, 판정, 수치, 근거...]
    return rows


def flags_of(argv):
    """명령에서 (플래그, 값) 쌍과 플래그 집합을 뽑는다."""
    pairs, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            v = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else ""
            pairs[a] = v
            i += 2 if v else 1
        else:
            i += 1
    return pairs


def check(argv, label=""):
    pairs = flags_of(argv)
    joined = " ".join(argv)
    bad = 0
    print(f"\n=== 검사 {label or ' '.join(argv[:6])} ===")

    for pat, verdict, num, *rest in parse_settled():
        why = rest[0] if rest else ""
        key = pat.split()[0]
        hit = (pat in joined) if " " in pat else (key in pairs)
        if not hit:
            continue
        mark = "[금지]" if verdict == "BANNED" else ("[닫힘]" if verdict == "CLOSED" else "[참고]")
        print(f"  {mark}  {pat}  [{num}]")
        print(f"        {why}")
        if verdict == "BANNED":
            bad = 2

    if os.path.exists(LEDGER):
        sig = {k for k in pairs if k not in ("--tag", "--seed", "--seeds")}
        hits = []
        with open(LEDGER, encoding="utf-8") as f:
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) < 9:
                    continue
                prev = flags_of(c[-1].split())
                prevsig = {k for k in prev if k not in ("--tag", "--seed", "--seeds")}
                if prevsig == sig:
                    hits.append(c)
        if hits:
            print(f"  [중복] 같은 플래그 조합을 이미 {len(hits)}회 실행함:")
            for c in hits[:4]:
                print(f"        {c[0]} {c[2]:<12} val {c[7]:>8}  test {c[8]:>8}")
        else:
            print("  [OK] LEDGER 에 같은 조합 없음")
    else:
        print("  (LEDGER.tsv 없음 — 첫 실행)")

    diff = [f"{k}={pairs[k]}(제출 {v})" for k, v in SUBMIT_VALUES.items()
            if k in pairs and pairs[k] != v]
    diff += [f"{k} 없음(제출엔 있음)" for k in SUBMIT_FLAGS if k not in pairs]
    diff += [f"{k} 추가(제출엔 없음)" for k in pairs
             if k.startswith("--drop-f-pre") or k.startswith("--league")]
    if diff:
        print(f"  [주의] 제출 설정과 다른 점 {len(diff)}개: {', '.join(diff[:6])}")
        print("        → 이 실행에서 잰 **후처리 상수는 제출에 쓰면 안 된다** (v16 −6.15)")
    return bad


def scan_file(path):
    """스크립트 안의 학습 명령을 전부 찾아 검사한다.

    쉘 변수(B/V/S)와 셸 함수(run/cz)를 쓰면 명령 줄에 train_gbdt2.py 가 안 보인다 —
    실제로 chain3.sh 가 그렇다. 변수는 펼치고, 함수 호출은 함수 본문의 명령에
    인자를 붙여 재구성한다. 여기서 놓치면 precheck 자체가 무의미하다.
    """
    text = open(path, encoding="utf-8").read()
    ext = os.path.splitext(path)[1].lower()
    var = dict(re.findall(r'^\s*(?:set\s+)?(\w+)\s*=\s*"?([^"\n]*)"?\s*$',
                          text, re.M))
    # 셸 함수:  name() { <cmd> ... "${@:2}" ... ; }
    funcs = dict(re.findall(r'^\s*(\w+)\(\)\s*\{(.*?)\}', text, re.M | re.S))

    def expand(s):
        for _ in range(4):
            if ext == ".bat":       # %B% 형식. 이름을 지우면 안 된다 —
                                    # 예전엔 '%'만 공백으로 바꿔서 %B% 안의
                                    # 플래그가 통째로 사라졌고, 그 안에 금지
                                    # 플래그가 있어도 못 잡았다.
                s = re.sub(r'%(\w+)%', lambda m: var.get(m.group(1), ""), s)
            s = re.sub(r'\$\{?(\w+)\}?', lambda m: var.get(m.group(1), ""), s)
        return s

    cmds = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if line.startswith("#") or line.startswith("REM "):
            continue
        if "train_gbdt2.py" in line and "()" not in line:
            cmds.append((n, expand(line)))
            continue
        m = re.match(r'^(\w+)\s+(\S+)\s*(.*)$', line)
        if m and m.group(1) in funcs:
            body = funcs[m.group(1)]
            if "train_gbdt2.py" not in body:
                continue
            body = re.sub(r'"\$\{@:\d+\}"', m.group(3), body)
            body = body.replace('"$1"', m.group(2))
            cmds.append((n, expand(body)))
    if not cmds:
        print(f"경고: {path} 에서 학습 명령을 하나도 못 찾았다 — "
              f"검사가 안 된 것이니 형식을 확인할 것")
        return 1
    worst = 0
    for n, c in cmds:
        worst = max(worst, check(c.split(), f"{os.path.basename(path)}:{n}"))
    print(f"\n총 {len(cmds)}개 명령 검사 | 최악 종료코드 {worst}")
    return worst


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--file":
        return max(scan_file(p) for p in sys.argv[2:])
    return check(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
