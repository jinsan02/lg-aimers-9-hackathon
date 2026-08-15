"""Did the code that produces a training number change between two commits?

Commit equality is the wrong question. After the P0/P1/P2 work of 2026-08-15 the
deployed commit differs from the one that produced the RANK16 scout by four
commits — but almost all of that is documents, tests, judging tools and
comments, none of which can move a fitted number. Refusing to reuse the scout on
commit inequality alone would burn a 20-minute GPU fit to reproduce a byte-equal
result; reusing it on a hand-wave would repeat the v16/v17 mistake.

So this compares **abstract syntax trees**, not text. Comments and docstrings do
not appear in an AST, and reformatting does not change one. For each file on the
training path:

  identical AST  -> that file provably cannot have changed a number
  different AST  -> the differing top-level functions are named, and the caller
                    decides whether any of them is on the path

The file list is explicit rather than "everything under src/", because a change
to, say, `teacher.py` cannot affect a run that never imports it. Passing
`--for rank` narrows it to the modules a ranker fit actually loads.

  python tools/training_path_diff.py 0b06e60 HEAD
  python tools/training_path_diff.py 0b06e60 HEAD --for rank

Exit 0 when nothing on the path changed, 1 when something did.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys

# Modules whose code can move a fitted number. `build_*.py`, `teacher.py` and
# the tools tree are deliberately absent: they either run before training as a
# separate step or do not run at all during a fit.
# Exactly the modules `fpipe` imports while building a training frame, plus the
# trainer itself. Verified against the local imports in src/fpipe.py rather
# than guessed -- an earlier list named a `src/domain.py` that does not exist,
# and a file that cannot be read reports as "changed", which would have forced
# a pointless 20-minute re-fit.
_CORE = ["src/train_gbdt2.py", "src/fpipe.py", "src/features.py",
         "src/season_std.py", "src/target_enc.py", "src/skill.py",
         "src/roster_transition.py", "src/graph_features.py"]
PATHS = {
    "common": _CORE,
    "rank": _CORE,
    "cell": _CORE + ["src/failmode.py"],
}


def show(commit, path):
    # Bytes, decoded as utf-8 explicitly. `text=True` decodes with the console
    # codepage, which is cp949 here, and every source file with a Korean
    # comment raised UnicodeDecodeError inside subprocess's reader thread --
    # where it does not propagate, so `git show` merely returned nothing and
    # every file read as "absent at both commits".
    r = subprocess.run(["git", "show", f"{commit}:{path}"],
                       capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", errors="replace")


def strip_docstrings(tree):
    """Remove docstrings so a comment-only edit reads as no change."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def fingerprint(src):
    return ast.dump(strip_docstrings(ast.parse(src)), annotate_fields=True)


def top_level(src):
    """{name: dumped body} for every top-level def, for naming what changed."""
    out = {}
    tree = strip_docstrings(ast.parse(src))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out[node.name] = ast.dump(node)
        else:
            out.setdefault("<module level>", "")
            out["<module level>"] += ast.dump(node)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new", nargs="?", default="HEAD")
    ap.add_argument("--for", dest="which", default="common",
                    choices=sorted(PATHS))
    a = ap.parse_args()

    files = PATHS[a.which]
    print(f"training path ({a.which}): {a.old} -> {a.new}\n")
    changed = []
    for path in files:
        o, n = show(a.old, path), show(a.new, path)
        if o is None or n is None:
            where = ", ".join(c for c, v in ((a.old, o), (a.new, n))
                              if v is None)
            print(f"  {'MISSING':<10}{path}   (absent at {where})")
            changed.append(path)
            continue
        if o == n:
            print(f"  {'same':<10}{path}   (byte-identical)")
            continue
        try:
            same_ast = fingerprint(o) == fingerprint(n)
        except SyntaxError as e:
            print(f"  {'UNPARSED':<10}{path}   {e}")
            changed.append(path)
            continue
        if same_ast:
            print(f"  {'same':<10}{path}   (text differs; AST identical — "
                  f"comments/docstrings only)")
            continue
        to, tn = top_level(o), top_level(n)
        names = sorted({k for k in set(to) | set(tn)
                        if to.get(k) != tn.get(k)})
        print(f"  {'CHANGED':<10}{path}   {len(names)} definition(s): "
              + ", ".join(names[:8]) + (" ..." if len(names) > 8 else ""))
        changed.append(path)

    print()
    if not changed:
        print("No file on this training path changed behaviour. A handoff "
              "produced at the old commit is reusable as-is.")
        return 0
    print(f"{len(changed)} file(s) changed behaviour on this path. Decide per "
          f"definition whether a fitted number can move; if any can, re-run "
          f"the stage rather than reusing its handoff.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
