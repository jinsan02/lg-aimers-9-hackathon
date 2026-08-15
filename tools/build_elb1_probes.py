"""Build the two pre-registered E-LB1 probes from the verified champion ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "submissions", "b1s8_20260813.zip")
BASE_SHA = "c2771bfdbbd9d81f9e43632d57fea5befeb16ff59478af06fb86114a4c6e7332"
OLD = "    return np.clip(p + mid + pb, 0, 1)"
ARMS = {
    "plus": (0.010, "elb1_plus_0816.zip"),
    "minus": (-0.010, "elb1_minus_0816.zip"),
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


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
            raise SystemExit(f"unsafe champion member: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-delta", type=float)
    args = ap.parse_args()
    arms = ARMS
    report_name = "elb1_manifest.json"
    if args.final_delta is not None:
        if not (-0.02 <= args.final_delta <= 0.02):
            raise SystemExit("final delta outside pre-registered [-0.02, 0.02]")
        arms = {"final": (args.final_delta, "elb1_final_0816.zip")}
        report_name = "elb1_final_manifest.json"
    if sha_file(BASE) != BASE_SHA:
        raise SystemExit("champion ZIP hash drift")
    with zipfile.ZipFile(BASE) as zf:
        safe_names(zf)
        original = {n: zf.read(n) for n in zf.namelist()}
    script = original["script.py"].decode("utf-8")
    if script.count(OLD) != 1:
        raise SystemExit("champion final-output hook is not unique")

    manifest = {"baseline": os.path.basename(BASE), "baseline_sha256": BASE_SHA,
                "arms": {}}
    os.makedirs(os.path.join(ROOT, "submissions"), exist_ok=True)
    for arm, (delta, filename) in arms.items():
        signed = f"{delta:+.12f}" if arm == "final" else f"{delta:+.6f}"
        new = ("    # E-LB1 fixed final-output probe; no test-frame aggregate.\n"
               f"    return np.clip(p + mid + pb + ({signed}), 0, 1)")
        patched = script.replace(OLD, new).encode("utf-8")
        payload = dict(original); payload["script.py"] = patched
        out = os.path.join(ROOT, "submissions", filename)
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as zf:
            for name in original:  # preserve the champion's member ordering
                zf.writestr(name, payload[name])
        with zipfile.ZipFile(out) as zf:
            safe_names(zf)
            rebuilt = {n: zf.read(n) for n in zf.namelist()}
        changed = [n for n in original if rebuilt[n] != original[n]]
        if changed != ["script.py"] or set(rebuilt) != set(original):
            raise SystemExit(f"{arm}: package delta is not script-only: {changed}")
        manifest["arms"][arm] = {
            "delta": delta, "file": filename, "zip_sha256": sha_file(out),
            "script_sha256": sha_bytes(patched), "changed_members": changed,
            "members": len(rebuilt), "bytes": os.path.getsize(out),
        }
        print(f"{arm}: {out} | sha256 {manifest['arms'][arm]['zip_sha256']}")

    report = os.path.join(ROOT, "out", report_name)
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
