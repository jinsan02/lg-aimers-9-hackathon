"""Replace the eight binary members in a verified submission template zip."""

import argparse
import os
import shutil
import tempfile
import zipfile


SEEDS = [42, 7, 13, 3, 4, 5, 6, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--base-tag", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", default="model")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(dir=".") as td:
        with zipfile.ZipFile(args.template) as zf:
            zf.extractall(td)
        for seed in SEEDS:
            src = os.path.join(args.model_dir, f"cat_{args.base_tag}_s{seed}.pkl")
            dst = os.path.join(td, "model", f"cat_v14f_s{seed}.pkl")
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as zf:
            for root, _, files in os.walk(td):
                for name in sorted(files):
                    path = os.path.join(root, name)
                    arc = os.path.relpath(path, td).replace(os.sep, "/")
                    zf.write(path, arc)
    with zipfile.ZipFile(args.output) as zf:
        names = zf.namelist()
        if "script.py" not in names or any("\\" in n for n in names):
            raise ValueError("invalid zip layout")
        model_count = sum(n.startswith("model/") and not n.endswith("/") for n in names)
    print(f"saved {args.output} size={os.path.getsize(args.output)/1e6:.1f}MB "
          f"models={model_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
