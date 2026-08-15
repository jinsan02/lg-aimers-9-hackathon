"""A stage ends when a **separate process** says it ended.

The problem this exists for, measured on `desktop-5070` on 2026-08-14/15: a
CatBoost worker finished its stage, wrote its artifact, and then would not die.
`os._exit(0)` left it spinning for **8h36m**. In the next attempt
`TerminateProcess(GetCurrentProcess(), 0)` returned and the process stayed;
`taskkill /f /pid` reported success and the process stayed. The only thing that
ended it was `schtasks /end`, issued from outside.

So a worker may not be trusted to report its own completion **or** to end
itself. The contract splits those two jobs across two scheduled tasks:

    worker      does the work, writes the artifacts, then announces
                `<stage>.ready.json` by atomic rename, then does whatever it
                wants -- exit, hang, spin. Nothing downstream depends on it.

    supervisor  polls for `<stage>.ready.json`, verifies every artifact it
                names against the sha recorded in it, ends the worker task from
                outside with `schtasks /end`, and writes `<stage>.ended.json`.

Three properties matter and each one is a past failure:

  * **Atomic announce.** `open(...,'w')` + `json.dump` is observable
    half-written; a supervisor polling every second will read `{"stage": "st`
    and crash or, worse, parse a truncated artifact list. Written to `.tmp` and
    `os.replace`d, which is atomic on NTFS within a volume.
  * **Artifacts are verified, not assumed.** The 2026-08-15 rank incident
    produced a `.cbm` that loaded, reported the right tree count, dumped valid
    JSON, and then died `0xC0000005` on a one-row predict. A ready file that
    merely says "done" would have advanced the chain. `ready` records size and
    sha per artifact and the supervisor re-checks them.
  * **Staleness.** A `<stage>.ready.json` left over from an earlier run makes
    the next run look instantly complete. Every ready file carries a `run_id`
    and the supervisor accepts only the one it was told to wait for.

The terminator is injected (`end_fn`) so the whole contract is testable on the
laptop with a fake worker and no scheduled task at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time

HANDOFF = os.path.join("out", "handoff")


def _sha(path, cap=1 << 30):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
            cap -= len(b)
            if cap <= 0:
                break
    return h.hexdigest()[:16]


def paths(stage, root="."):
    d = os.path.join(root, HANDOFF)
    return (os.path.join(d, f"{stage}.ready.json"),
            os.path.join(d, f"{stage}.ended.json"))


def _write_atomic(path, rec, retries=60, backoff=0.05):
    """Write via a temp file and one atomic rename.

    The rename needs a retry loop **on Windows specifically**. `os.replace`
    calls `MoveFileEx(..., REPLACE_EXISTING)`, which fails `WinError 5` when the
    destination is currently open by anyone -- and the supervisor polls this
    exact path with `open()`. Found by `tests/test_stage_contract.py` check 3:
    a stage re-run overwrites an existing ready file while the supervisor is
    reading it, and the announce died with a PermissionError. The announce is
    the worker's last act, so losing it means the supervisor waits out its full
    timeout on a stage that actually finished.

    The reader's grip is microseconds; three seconds of retries is far more than
    it needs. If it still fails, raise -- a silently unannounced stage is the
    failure mode this whole module exists to remove.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    last = None
    for i in range(retries):
        try:
            os.replace(tmp, path)          # atomic on NTFS within a volume
            return path
        except PermissionError as e:       # destination held open by a reader
            last = e
            time.sleep(backoff)
    try:
        os.remove(tmp)
    except OSError:
        pass
    raise OSError(
        f"could not publish {path} after {retries} attempts ({last}). Something "
        f"is holding it open; the stage finished but nothing downstream can "
        f"learn that.")


# --------------------------------------------------------------------- worker

def announce_ready(stage, run_id, artifacts, payload=None, root="."):
    """Called by the worker once its artifacts are on disk.

    `artifacts` are paths relative to `root`. Every one is hashed here, so a
    stage that announces an artifact it did not actually write fails at the
    announce rather than three stages later.
    """
    ready, _ = paths(stage, root)
    recs = []
    for rel in artifacts:
        full = os.path.join(root, rel)
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"{stage} announced {rel}, which does not exist. A ready file "
                f"that names a missing artifact is worse than no ready file.")
        recs.append({"path": rel, "bytes": os.path.getsize(full),
                     "sha": _sha(full)})
    rec = {"stage": stage, "run_id": str(run_id), "host": socket.gethostname(),
           "pid": os.getpid(), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "epoch": time.time(), "artifacts": recs, "payload": payload or {}}
    return _write_atomic(ready, rec)


# ----------------------------------------------------------------- supervisor

def schtasks_end(task_name):
    """The only termination that has ever worked on the 5070."""
    r = subprocess.run(["schtasks", "/end", "/tn", task_name],
                       capture_output=True, text=True)
    return {"cmd": f"schtasks /end /tn {task_name}", "rc": r.returncode,
            "out": (r.stdout or "").strip()[:400],
            "err": (r.stderr or "").strip()[:400]}


def verify(rec, root="."):
    """Re-check every artifact the ready file names. Returns a list of problems."""
    bad = []
    for a in rec.get("artifacts", []):
        full = os.path.join(root, a["path"])
        if not os.path.exists(full):
            bad.append(f"{a['path']}: announced but missing")
            continue
        size = os.path.getsize(full)
        if size != a["bytes"]:
            bad.append(f"{a['path']}: {size} bytes, announced {a['bytes']}")
            continue
        got = _sha(full)
        if got != a["sha"]:
            bad.append(f"{a['path']}: sha {got}, announced {a['sha']}")
    return bad


def supervise(stage, run_id, task_name=None, timeout=7200, poll=2.0,
              end_fn=None, root=".", now=time.time, sleep=time.sleep):
    """Wait for the stage to announce, verify it, end the worker, record it.

    Returns the `ended` record. `status` is one of:

      ok        announced, artifacts verified, worker ended
      corrupt   announced but an artifact does not match what was announced
      timeout   never announced within `timeout` seconds

    The worker is ended in the `ok` **and** the `corrupt` case: a worker that
    wrote a bad artifact is exactly the one that must not stay alive holding the
    GPU. It is not ended on `timeout` unless a task name was given, because
    with nothing announced there is no evidence the right process is being
    ended.
    """
    ready, ended_p = paths(stage, root)
    end_fn = end_fn or schtasks_end
    t0 = now()
    rec, status, problems = None, "timeout", []

    while now() - t0 < timeout:
        if os.path.exists(ready):
            try:
                with open(ready, encoding="utf-8") as f:
                    cand = json.load(f)
            except (ValueError, OSError):
                # A half-written file should be impossible (atomic rename), so
                # this is a real anomaly rather than a race to retry silently.
                sleep(poll)
                continue
            if str(cand.get("run_id")) != str(run_id):
                # Stale announce from an earlier run of the same stage. Keep
                # waiting rather than advance on it.
                sleep(poll)
                continue
            rec = cand
            problems = verify(rec, root)
            status = "corrupt" if problems else "ok"
            break
        sleep(poll)

    term = None
    if task_name and (status in ("ok", "corrupt")):
        term = end_fn(task_name)
    out = {"stage": stage, "run_id": str(run_id), "status": status,
           "problems": problems, "waited_s": round(now() - t0, 1),
           "host": socket.gethostname(),
           "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "task_name": task_name, "termination": term,
           "ready": rec}
    _write_atomic(ended_p, out)
    return out


def read_ended(stage, root="."):
    _, p = paths(stage, root)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def require_ok(stage, run_id, root="."):
    """Gate for the next stage. Raises unless this stage ended cleanly."""
    e = read_ended(stage, root)
    if e is None:
        raise SystemExit(f"{stage} has no ended record -- the supervisor never "
                         f"ran, so nothing knows whether it finished.")
    if str(e.get("run_id")) != str(run_id):
        raise SystemExit(f"{stage} ended record is from run {e.get('run_id')}, "
                         f"not {run_id}. Refusing to build on another run.")
    if e.get("status") != "ok":
        raise SystemExit(f"{stage} ended with status {e.get('status')}: "
                         + "; ".join(e.get("problems") or ["(no detail)"]))
    return e
