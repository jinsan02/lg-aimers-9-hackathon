"""`precheck.py` must never report success on a script it did not open.

Measured 2026-08-28: `python tools/precheck.py --bat scripts/bnd_5070.bat`
printed `[OK] no identical flag combination in LEDGER` and a plausible `[WARN]`
block computed from the argv itself, then **exited 0** — without reading the
file. A reader would reasonably conclude the script had passed its safety check.
That guard gates every GPU launch, so a false pass is worse than no guard.
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRE = os.path.join(ROOT, "tools", "precheck.py")


def run(*args):
    # precheck echoes SETTLED/LEDGER text, which contains em dashes and arrows.
    # text=True would decode that with the console codepage (cp949 here) and
    # raise inside the harness rather than in the tool under test.
    p = subprocess.run([sys.executable, PRE, *args], capture_output=True,
                       encoding="utf-8", errors="replace", cwd=ROOT)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main():
    fails = []

    def chk(name, ok, detail=""):
        print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    script = os.path.join(ROOT, "scripts", "leafit10_s3_local.bat")
    have_script = os.path.exists(script)

    # 1. the real invocation still works and says so out loud
    if have_script:
        rc, out = run("--file", script)
        chk("1 --file checks the file and prints the count",
            rc == 0 and "command(s) checked" in out, f"rc={rc}")

    # 2. a mistyped option must NOT silently become a flag check
    if have_script:
        rc, out = run("--bat", script)
        chk("2 --bat is refused, not silently flag-checked",
            rc != 0 and "unknown option" in out, f"rc={rc}")
        chk("2b and it does not print a reassuring OK",
            "no identical flag combination" not in out)

    # 3. a bare path in the flag position is the shape a forgotten --file takes
    if have_script:
        rc, out = run("scripts/leafit10_s3_local.bat")
        chk("3 a bare script path is refused",
            rc != 0 and "looks like a path" in out, f"rc={rc}")

    # 4. no arguments is a usage error, not a pass
    rc, out = run()
    chk("4 no arguments exits non-zero with usage", rc != 0 and "usage:" in out,
        f"rc={rc}")

    # 5. --file with no path
    rc, out = run("--file")
    chk("5 --file with no path exits non-zero", rc != 0, f"rc={rc}")

    # 6. genuine training flags still go through the flag checker
    rc, out = run("--model", "cat", "--depth", "8")
    chk("6 real training flags are still checked", rc == 0, f"rc={rc}")

    # 7. a file containing no training command must not pass silently
    with tempfile.TemporaryDirectory() as d:
        empty = os.path.join(d, "empty.bat")
        open(empty, "w").write("@echo off\nREM nothing here\n")
        rc, out = run("--file", empty)
        chk("7 a script with no training command does not exit 0",
            rc != 0 and "found no training command" in out, f"rc={rc}")

    # 8. the unknown-option check reads the trainer's own argparse, so a newly
    #    added trainer flag is never rejected for being unrecognised
    rc, out = run("--cell-leaf-iters", "10")
    chk("8 a real trainer flag added today is accepted", rc == 0, f"rc={rc}")

    print(f"\n{len(fails)} failure(s)" if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
