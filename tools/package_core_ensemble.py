"""Build a v11-compatible package with an equal four-family base ensemble."""

import argparse
import os
import shutil
import tempfile
import zipfile


SEEDS = [42, 7, 13, 3, 4, 5, 6, 8]
FAMILIES = ["v14f", "CE1V_lr5", "CE1V_rs05", "CE1V_rs2"]


WEIGHT_BLOCK = '''_F = [42, 7, 13, 3, 4, 5, 6, 8]
_C = [42, 7, 13, 3, 4, 5]
_CORE = ["v14f", "CE1V_lr5", "CE1V_rs05", "CE1V_rs2"]
WEIGHTS = ([(os.path.join(SCRIPT_DIR, "model", f"cat_{tag}_s{s}.pkl"),
             (1-_W_CELL)/(len(_CORE)*len(_F))) for tag in _CORE for s in _F]
           + [(os.path.join(SCRIPT_DIR, "model", f"cat_ZD5_s{s}.pkl"),
               _W_CELL/len(_C)) for s in _C])'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", default="model")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(dir=".") as td:
        with zipfile.ZipFile(args.template) as zf:
            zf.extractall(td)
        script_path = os.path.join(td, "script.py")
        with open(script_path, encoding="utf-8") as f:
            script = f.read()
        start = script.index("_F =")
        end = script.index("\n\nMID_COL", start)
        script = script[:start] + WEIGHT_BLOCK + script[end:]
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(script)
        for family in FAMILIES[1:]:
            for seed in SEEDS:
                name = f"cat_{family}_s{seed}.pkl"
                src = os.path.join(args.model_dir, name)
                if not os.path.exists(src):
                    raise FileNotFoundError(src)
                shutil.copy2(src, os.path.join(td, "model", name))
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
    if model_count != 39:  # 32 core + 6 cell + 1 matchup constants
        raise ValueError(f"unexpected model payload count: {model_count}")
    print(f"saved {args.output} size={os.path.getsize(args.output)/1e6:.1f}MB "
          f"payloads={model_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
