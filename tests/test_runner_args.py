"""Do the flag values a runner is handed actually reach Python intact?

On 2026-08-13 three GPU launches were lost to cmd rewriting an argument before
Python saw it, and every one returned exit code 0. `--te-k b:500,*:50` was split
on the comma; `--row-filter "strikes_before == 2"` lost its quotes to `%~1` and
then shattered on the space and the `=`. A runner that starts and a runner that
starts *with the flags you wrote* are different things, and only the second one
is worth GPU time.

Two checks:
  1. the forwarding block in scripts/echo_args.bat is byte-identical to the one
     in every real runner -- otherwise this file tests a copy that has drifted
  2. awkward values survive a round trip through cmd

Run: python tests/test_runner_args.py     (Windows only; skips elsewhere)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
RUNNERS = ["run_judge_5070.bat", "run_submit_5070.bat", "run_judge2223_5070.bat"]
BLOCK = re.compile(r"set EXTRA=\s*\n\s*shift\s*\n\s*shift\s*\n\s*:more.*?goto more\s*\n\s*:go",
                   re.S | re.I)

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def block_of(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        m = BLOCK.search(f.read())
    return re.sub(r"\s+", " ", m.group(0)).strip() if m else None


def main():
    if os.name != "nt":
        print("skipped: cmd argument parsing is the thing under test")
        return 0

    print("runner argument forwarding")
    ref = block_of(os.path.join(SCRIPTS, "echo_args.bat"))
    check("harness has a forwarding block", ref is not None)
    for r in RUNNERS:
        p = os.path.join(SCRIPTS, r)
        if not os.path.exists(p):
            check(f"{r} exists", False)
            continue
        check(f"{r} forwards identically", block_of(p) == ref)

    # Values that have actually broken a launch, plus the shapes around them.
    cases = [
        (["--p1"], ["--p1"]),
        (["--te-k", "b:500,*:50"], ["--te-k", "b:500,*:50"]),
        (["--row-filter", "strikes_before == 2"], ["--row-filter", "strikes_before == 2"]),
        (["--fm-noise-rate", "0.03873"], ["--fm-noise-rate", "0.03873"]),
        (["--p1", "--fm-modes", "middle,ball,reverse"],
         ["--p1", "--fm-modes", "middle,ball,reverse"]),
    ]
    bat = os.path.join(SCRIPTS, "echo_args.bat")
    for extra, want in cases:
        # Quote every value, which is what the runners are called with.
        quoted = " ".join(a if a.startswith("--") else f'"{a}"' for a in extra)
        cmd = f'"{bat}" ECHO "3,4,5" {quoted}'
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           encoding="utf-8", cwd=ROOT)
        got = re.findall(r"ARGV\[\d+\] (.+)", r.stdout or "")
        got = [eval(g) for g in got]                    # printed with repr()
        check(f"{' '.join(extra)}", got == want,
              "" if got == want else f"got {got}")

    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
