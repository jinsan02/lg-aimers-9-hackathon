"""Supervisor task: wait for a stage to announce, verify it, end the worker.

Registered as its **own** scheduled task, separate from the worker. That
separation is the point: on `desktop-5070` a finished worker could not be ended
from inside itself -- `os._exit(0)` left it spinning 8h36m, and both
`TerminateProcess` and `taskkill /f /pid` returned success while the process
stayed. `schtasks /end`, issued by another process, worked.

    # 1. worker, doing the actual stage
    bash tools/run5070.sh RK16_S12 'C:\\\\aimers\\\\scripts\\\\rk16_s12.bat' \
                                   'C:\\\\aimers\\\\out\\\\rk16_s12.log'
    # 2. supervisor, watching it
    bash tools/run5070.sh RK16_S12_SUP 'C:\\\\aimers\\\\scripts\\\\rk16_s12_sup.bat' \
                                       'C:\\\\aimers\\\\out\\\\rk16_s12_sup.log'

where the supervisor batch runs:

    C:\\aimers\\.conda\\python.exe C:\\aimers\\tools\\supervise_stage.py ^
        --stage rank12_RK16_s3 --run-id 20260815a --task RK16_S12 --timeout 5400

and the worker batch passes `--stage-run-id 20260815a` to `train_gbdt2.py`.

Exit codes: 0 clean, 2 the stage announced a corrupt artifact, 3 it never
announced. The next stage should call `stage_contract.require_ok`, which reads
`out/handoff/<stage>.ended.json` and refuses anything but a clean record for
its own run id.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import stage_contract as sc                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    help="stage name the worker announces, e.g. rank12_RK16_s3")
    ap.add_argument("--run-id", required=True,
                    help="must match the worker's --stage-run-id; a ready file "
                         "from any other run is ignored, so a leftover from a "
                         "previous attempt cannot satisfy this one")
    ap.add_argument("--task", default="",
                    help="scheduled task name to end once the stage is "
                         "verified. Omit to watch without terminating.")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()

    print(f"supervising {a.stage} (run {a.run_id}) "
          f"task={a.task or '(none)'} timeout={a.timeout:.0f}s", flush=True)
    rec = sc.supervise(a.stage, a.run_id, task_name=a.task or None,
                       timeout=a.timeout, poll=a.poll, root=a.root)
    print(json.dumps({k: v for k, v in rec.items() if k != "ready"},
                     indent=2, ensure_ascii=False), flush=True)

    if rec["status"] == "ok":
        n = len((rec.get("ready") or {}).get("artifacts", []))
        print(f"OK: {n} artifact(s) verified, worker ended.", flush=True)
        return 0
    if rec["status"] == "corrupt":
        print("CORRUPT: " + "; ".join(rec["problems"]), flush=True)
        print("The artifacts are kept. Name the cause before deleting any of "
              "them (AGENTS.md).", flush=True)
        return 2
    print(f"TIMEOUT after {rec['waited_s']}s with no announce. The worker may "
          f"still be running -- check before ending it, because nothing has "
          f"been verified.", flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
