"""Print argv exactly as Python received it. Nothing else.

Three GPU launches were wasted on 2026-08-13 because cmd mangled a flag value
before Python ever saw it, and each failure looked different:

    --te-k b:500,*:50        cmd split on the comma; argparse got a stray `*:50`
    --row-filter "..."       `%~1` stripped the quotes, then the token shattered
    strikes_before==2        cmd splits on `=` as well as space and comma

Every one of them exited 0 from the runner's point of view. So the check cannot
be "did it run" -- it has to be "did the bytes arrive". This script is the
target end of that check; `scripts/echo_args.bat` is the sending end.

  python tools/runner_echo_args.py --te-k "b:500,*:50"
  scripts\\echo_args.bat TAG "3,4,5" --row-filter "strikes_before == 2"
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    for i, a in enumerate(sys.argv[1:], 1):
        print(f"ARGV[{i}] {a!r}")
    print(f"ARGC {len(sys.argv) - 1}")
