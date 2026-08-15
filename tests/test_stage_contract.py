"""Nine contract checks for the external stage-termination protocol.

The protocol exists because a worker cannot be trusted to end itself. On
`desktop-5070`, after a stage finished: `os._exit(0)` left the process spinning
**8h36m**; `TerminateProcess(GetCurrentProcess(), 0)` returned and the process
stayed; `taskkill /f /pid` reported success and the process stayed. Only
`schtasks /end`, issued from outside, ended it.

Every check here runs against a **fake worker** -- a real subprocess, no GPU, no
scheduled task -- so the protocol can be proved on the laptop. The terminator is
injected, so "the supervisor ended the worker" is asserted rather than assumed.

**RANK16 stage 12 must not start until all nine pass.**

Run: python tests/test_stage_contract.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import stage_contract as sc                                      # noqa: E402

FAIL = []

# A worker that announces and then refuses to end -- the 5070's behaviour.
FAKE_WORKER = r'''
import os, sys, time
sys.path.insert(0, sys.argv[1])
import stage_contract as sc
root, stage, run_id, mode = sys.argv[2:6]
art = os.path.join(root, "out", "fake.bin")
os.makedirs(os.path.dirname(art), exist_ok=True)
if mode != "no_artifact":
    with open(art, "wb") as f:
        f.write(b"x" * 4096)
if mode == "silent":
    time.sleep(60)          # never announces
    sys.exit(0)
sc.announce_ready(stage, run_id, ["out/fake.bin"], {"mode": mode}, root=root)
if mode == "hang":
    while True:             # announced, then will not die
        time.sleep(0.2)
sys.exit(0)
'''


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def spawn(tmp, stage, run_id, mode):
    p = os.path.join(tmp, "worker.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write(FAKE_WORKER)
    return subprocess.Popen(
        [sys.executable, p, os.path.join(ROOT, "tools"), tmp, stage,
         str(run_id), mode],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def wait_ready(tmp, stage, limit=30):
    ready, _ = sc.paths(stage, tmp)
    t0 = time.time()
    while time.time() - t0 < limit:
        if os.path.exists(ready):
            return True
        time.sleep(0.05)
    return False


def main():
    tmp = tempfile.mkdtemp(prefix="stagecontract_")
    try:
        # 1 -------------------------------------------------------- announce
        print("announce")
        w = spawn(tmp, "st1", "run-1", "exit")
        got = wait_ready(tmp, "st1")
        w.wait(timeout=30)
        ready, _ = sc.paths("st1", tmp)
        rec = json.load(open(ready, encoding="utf-8")) if got else {}
        check("1 the worker announces after writing its artifact", got
              and rec.get("stage") == "st1" and rec.get("run_id") == "run-1",
              f"pid {rec.get('pid')}, {len(rec.get('artifacts', []))} artifact(s)")
        check("2 the announce records size and sha per artifact",
              bool(rec.get("artifacts"))
              and all({"path", "bytes", "sha"} <= set(a)
                      for a in rec["artifacts"]),
              str(rec.get("artifacts", [{}])[0].get("bytes")) + " bytes")

        # 3 ------------------------------------------------- atomic announce
        # A poller must never observe a partial file. Hammer the path while a
        # large announce is written; every read either fails to find it or
        # parses completely.
        big = os.path.join(tmp, "out", "big.bin")
        with open(big, "wb") as f:
            f.write(b"y" * (2 << 20))
        rp, _ = sc.paths("st_atomic", tmp)
        seen = {"partial": 0, "whole": 0}
        stop = threading.Event()

        def poll():
            while not stop.is_set():
                try:
                    with open(rp, encoding="utf-8") as f:
                        json.load(f)
                    seen["whole"] += 1
                except FileNotFoundError:
                    pass
                except ValueError:
                    seen["partial"] += 1
                except OSError:
                    pass

        th = threading.Thread(target=poll, daemon=True)
        th.start()
        published = 0
        try:
            for i in range(40):
                sc.announce_ready("st_atomic", f"r{i}", ["out/big.bin"],
                                  {"i": i, "pad": "z" * 4000}, root=tmp)
                published += 1
        finally:
            stop.set()
            th.join(timeout=5)
        check("3 no poller ever observes a half-written announce",
              seen["partial"] == 0 and seen["whole"] > 0,
              f"{seen['whole']} complete reads, {seen['partial']} partial")
        # Windows `os.replace` fails WinError 5 while the destination is open,
        # which is precisely what a polling supervisor does. Re-announcing a
        # stage must survive that or a finished stage looks hung.
        check("3b and every announce is published despite a concurrent reader",
              published == 40, f"{published}/40 overwrote an open ready file")

        # 4 ---------------------------------------------- announce must exist
        print("\nrefusals")
        try:
            sc.announce_ready("st_missing", "r", ["out/does_not_exist.bin"],
                              root=tmp)
            check("4 announcing a missing artifact raises", False, "it returned")
        except FileNotFoundError:
            check("4 announcing a missing artifact raises", True)

        # 5 ------------------------------------------------ corrupt artifact
        w = spawn(tmp, "st5", "run-5", "exit")
        wait_ready(tmp, "st5")
        w.wait(timeout=30)
        with open(os.path.join(tmp, "out", "fake.bin"), "wb") as f:
            f.write(b"z" * 4096)                     # same size, different bytes
        ended = sc.supervise("st5", "run-5", task_name="T5", timeout=5,
                             poll=0.05, end_fn=lambda t: {"cmd": t, "rc": 0},
                             root=tmp)
        check("5 an artifact changed after the announce -> corrupt",
              ended["status"] == "corrupt" and ended["problems"],
              ended["problems"][0][:70] if ended["problems"] else "")
        check("6 and the worker is ended anyway -- a bad stage must not keep "
              "the GPU", ended["termination"] is not None,
              str(ended["termination"]))

        # 7 --------------------------------------------------------- stale
        # A leftover ready.json from an earlier run must not satisfy a new one.
        t0 = time.time()
        ended = sc.supervise("st5", "run-DIFFERENT", task_name="T5", timeout=1,
                             poll=0.05, end_fn=lambda t: {"cmd": t, "rc": 0},
                             root=tmp)
        check("7 a ready file from another run is ignored",
              ended["status"] == "timeout" and ended["termination"] is None,
              f"waited {ended['waited_s']}s on an existing but stale file")

        # 8 ------------------------------------------- silent worker timeout
        w = spawn(tmp, "st8", "run-8", "silent")
        ended = sc.supervise("st8", "run-8", task_name="T8", timeout=1.5,
                             poll=0.05, end_fn=lambda t: {"cmd": t, "rc": 0},
                             root=tmp)
        w.kill()
        check("8 a worker that never announces -> timeout, no termination",
              ended["status"] == "timeout" and ended["termination"] is None,
              "nothing announced means nothing is known to end")
        try:
            sc.require_ok("st8", "run-8", root=tmp)
            check("8b and the next stage refuses to start", False, "it returned")
        except SystemExit as e:
            check("8b and the next stage refuses to start", True, str(e)[:60])

        # 9 ------------------------------- the real case: announced, then hangs
        print("\nthe 5070 case: announced, then will not die")
        w = spawn(tmp, "st9", "run-9", "hang")
        check("9a the worker announces", wait_ready(tmp, "st9"))
        alive_before = w.poll() is None
        killed = {}
        ended = sc.supervise("st9", "run-9", task_name="T9", timeout=5,
                             poll=0.05,
                             end_fn=lambda t: killed.setdefault(
                                 "r", {"cmd": f"schtasks /end /tn {t}", "rc": 0}),
                             root=tmp)
        check("9b it is still alive after announcing -- exactly the 5070 case",
              alive_before)
        check("9c the supervisor verifies, then ends it from outside",
              ended["status"] == "ok" and killed.get("r", {}).get("rc") == 0,
              killed.get("r", {}).get("cmd", ""))
        check("9d and records the termination in ended.json",
              sc.read_ended("st9", tmp)["termination"]["cmd"].startswith(
                  "schtasks /end"))
        e = sc.require_ok("st9", "run-9", root=tmp)
        check("9e so the next stage may start", e["status"] == "ok")
        w.kill()
        w.wait(timeout=10)

        print("\n" + ("all passed -- RANK16 stage 12 may start"
                      if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
        return 1 if FAIL else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
