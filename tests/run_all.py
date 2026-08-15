"""Run every test and **fail if none were collected**.

`python -m unittest discover -s tests` prints `Ran 0 tests ... OK` and exits 0,
because these tests are `main()`-style rather than `unittest.TestCase`. A green
zero is worse than a red one: it is the shape a broken CI reports forever. There
is no CI in this repo yet, so today that would mislead a person instead of a
pipeline — which is not better.

This runner collects `tests/test_*.py`, runs each module's `main()` in its own
subprocess so a native abort is reported rather than inherited, and exits
non-zero if the collection is empty.

  python tests/run_all.py            # everything
  python tests/run_all.py routing    # only names containing "routing"
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT = 1800


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    files = [f for f in files if not pat or pat in os.path.basename(f)]

    if not files:
        print(f"COLLECTED 0 TESTS (pattern {pat!r}) -- refusing to report "
              f"success. An empty run is a failure, not a pass.")
        return 2

    print(f"collected {len(files)} test module(s)\n")
    failed, t0 = [], time.time()
    for f in files:
        name = os.path.basename(f)
        s = time.time()
        try:
            r = subprocess.run([sys.executable, f], cwd=ROOT,
                               capture_output=True, text=True, timeout=TIMEOUT)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code, r = 124, None
        mark = "ok  " if code == 0 else f"FAIL({code}) "
        print(f"  {mark}{name:34} {time.time() - s:6.1f}s")
        if code != 0:
            failed.append(name)
            if r is not None:
                tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
                for line in tail[-12:]:
                    print(f"      | {line}")

    print(f"\n{len(files) - len(failed)}/{len(files)} passed in "
          f"{time.time() - t0:.0f}s")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
