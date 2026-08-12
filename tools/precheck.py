"""Pre-run duplicate and ban check. **Every experiment must pass this first.**

On 2026-08-07 we ran 30 experiments in a day and recorded none of them, then
queued the `season` drop calling it "an axis we never tried" — a question E08 had
closed at -580. This tool exists to stop relying on human memory (and on
conversation, which gets compacted).

What it does:
  1. Match the command against BANNED/CLOSED flags in docs/SETTLED.md (exit 2 on BANNED)
  2. Search LEDGER.tsv for **the same flag combination already run**, and show results
  3. List every difference from the submission config (SUBMIT_FLAGS), so a run that
     must not source a post-hoc constant announces itself

Usage:
    python tools/precheck.py --model cat --drop-cols season --val-season 2023 ...
    python tools/precheck.py --file some.sh        # every command inside a script
"""

import os
import re
import sys

# SETTLED.md contains U+2212 (-) and may contain non-ASCII. Printing that to a
# Windows console (cp949) raises UnicodeEncodeError -- and a tool that crashes
# while warning you about a banned axis just gets ignored. Replace what cannot
# be encoded instead of dying.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTLED = os.path.join(ROOT, "docs", "SETTLED.md")
LEDGER = os.path.join(ROOT, "LEDGER.tsv")

# Training config the current submission uses. Any run that measures a post-hoc
# constant must match this exactly.
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
                    rows.append(parts)      # [pattern, verdict, number, mechanism...]
    return rows


def flags_of(argv):
    """Extract (flag, value) pairs from a command line."""
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
    print(f"\n=== check {label or ' '.join(argv[:6])} ===")

    for pat, verdict, num, *rest in parse_settled():
        why = rest[0] if rest else ""
        key = pat.split()[0]
        hit = (pat in joined) if " " in pat else (key in pairs)
        if not hit:
            continue
        mark = "[BANNED]" if verdict == "BANNED" else ("[CLOSED]" if verdict == "CLOSED" else "[NOTE]")
        print(f"  {mark}  {pat}  [{num}]")
        print(f"        {why}")
        if verdict == "BANNED":
            bad = 2

    if os.path.exists(LEDGER):
        # Compare (flag, value) pairs, not just flag names. Names alone called
        # `--depth 8` and `--depth 9` an "exact duplicate", so the warning could
        # not be used as a verdict.
        skip = ("--tag", "--seed", "--seeds")
        sig = {k: v for k, v in pairs.items() if k not in skip}
        hits = []
        with open(LEDGER, encoding="utf-8") as f:
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) < 9:
                    continue
                prev = flags_of(c[-1].split())
                prevsig = {k: v for k, v in prev.items() if k not in skip}
                if prevsig == sig:
                    hits.append(c)
        if hits:
            print(f"  [DUPLICATE] this exact flag combination ran {len(hits)} time(s) already:")
            for c in hits[:4]:
                print(f"        {c[0]} {c[2]:<12} val {c[7]:>8}  test {c[8]:>8}")
        else:
            print("  [OK] no identical flag combination in LEDGER")
    else:
        print("  (no LEDGER.tsv -- first run)")

    diff = [f"{k}={pairs[k]} (submission {v})" for k, v in SUBMIT_VALUES.items()
            if k in pairs and pairs[k] != v]
    diff += [f"{k} missing (submission has it)" for k in SUBMIT_FLAGS if k not in pairs]
    diff += [f"{k} added (submission lacks it)" for k in pairs
             if k.startswith("--drop-f-pre") or k.startswith("--league")]
    if diff:
        print(f"  [WARN] differs from the submission config in {len(diff)} way(s): {', '.join(diff[:6])}")
        print("        -> a post-hoc constant measured here MUST NOT be shipped (v16 -6.15)")
    return bad


def scan_file(path):
    """Find and check every training command inside a script.

    With shell variables (B/V/S) and shell functions (run/cz), train_gbdt2.py never
    appears on the command line -- chain3.sh is exactly like that. So expand the
    variables and reconstruct function calls by substituting arguments into the
    function body. Missing a command here makes precheck itself pointless.
    """
    text = open(path, encoding="utf-8").read()
    ext = os.path.splitext(path)[1].lower()
    var = dict(re.findall(r'^\s*(?:set\s+)?(\w+)\s*=\s*"?([^"\n]*)"?\s*$',
                          text, re.M))
    # shell function:  name() { <cmd> ... "${@:2}" ... ; }
    funcs = dict(re.findall(r'^\s*(\w+)\(\)\s*\{(.*?)\}', text, re.M | re.S))

    def expand(s):
        for _ in range(4):
            if ext == ".bat":       # %B% form. Do not just strip the name --
                                    # an earlier version replaced only '%' with
                                    # a space, so every flag inside %B% vanished
                                    # and a banned flag in there went unseen.
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
        print(f"WARNING: found no training command in {path} -- "
              f"nothing was checked, so verify the file format")
        return 1
    worst = 0
    for n, c in cmds:
        worst = max(worst, check(c.split(), f"{os.path.basename(path)}:{n}"))
    print(f"\n{len(cmds)} command(s) checked | worst exit code {worst}")
    return worst


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--file":
        return max(scan_file(p) for p in sys.argv[2:])
    return check(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
