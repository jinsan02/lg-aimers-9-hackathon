"""Build the two pre-registered cell-weight probes from the E-LB1 champion."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "submissions", "elb1_final_0816.zip")
BASE_SHA = "e99a5772d0bd3f649a6e5cee8972cb0f3db205ce7be957a662446f358e0dbfd3"
OLD = "_W_CELL = 0.55"
ARMS = {
    "w045": (0.45, "elb2_w045_0816.zip"),
    "w065": (0.65, "elb2_w065_0816.zip"),
}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_names(zf):
    for name in zf.namelist():
        parts = name.replace("\\", "/").split("/")
        if "\\" in name or name.startswith("/") or ".." in parts:
            raise SystemExit(f"unsafe member: {name}")


def main():
    if sha_file(BASE) != BASE_SHA:
        raise SystemExit("E-LB1 champion ZIP hash drift")
    with zipfile.ZipFile(BASE) as zf:
        safe_names(zf)
        original = {n: zf.read(n) for n in zf.namelist()}
    script = original["script.py"].decode("utf-8")
    if script.count(OLD) != 1:
        raise SystemExit("_W_CELL hook is not unique")

    manifest = {"baseline": os.path.basename(BASE),
                "baseline_sha256": BASE_SHA, "arms": {}}
    for arm, (weight, filename) in ARMS.items():
        patched = script.replace(OLD, f"_W_CELL = {weight:.2f}").encode("utf-8")
        payload = dict(original); payload["script.py"] = patched
        out = os.path.join(ROOT, "submissions", filename)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for name in original:
                zf.writestr(name, payload[name])
        with zipfile.ZipFile(out) as zf:
            safe_names(zf)
            rebuilt = {n: zf.read(n) for n in zf.namelist()}
        changed = [n for n in original if rebuilt[n] != original[n]]
        if changed != ["script.py"] or set(rebuilt) != set(original):
            raise SystemExit(f"{arm}: unexpected package delta {changed}")
        manifest["arms"][arm] = {
            "weight": weight, "file": filename, "zip_sha256": sha_file(out),
            "script_sha256": hashlib.sha256(patched).hexdigest(),
            "changed_members": changed, "members": len(rebuilt),
            "bytes": os.path.getsize(out),
        }
        print(f"{arm}: {out} | sha256 {manifest['arms'][arm]['zip_sha256']}")
    report = os.path.join(ROOT, "out", "elb2_manifest.json")
    with open(report, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest: {report}")


if __name__ == "__main__":
    main()
