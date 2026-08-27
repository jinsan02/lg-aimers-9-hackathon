"""`--cell-leaf-iters` must reach BOTH cell constructors and nothing else.

The failure this guards against is not hypothetical. On 2026-08-14
`--max-ctr-complexity 3` reached the binary arm and not the cell arm; the
packaged model reported 4, one seed reproduced its own control to the cent, and
the run exited 0. A flag that silently reaches one constructor produces a
"candidate" that is the control.
"""
import ast
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import train_gbdt2 as T  # noqa: E402
import lineage  # noqa: E402

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "train_gbdt2.py")


def _args(**kw):
    a = types.SimpleNamespace(max_ctr_complexity=0, boosting_type="",
                              bootstrap_type="", cell_leaf_iters=0)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _cell_param_call_sites():
    """Every CatBoostClassifier call that splats a `_cell_params(args)` dict."""
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    holders = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name)
                and n.value.func.id == "_cell_params"
                and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)):
            holders.add(n.targets[0].id)
    sites = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "CatBoostClassifier"):
            for kw in n.keywords:
                if (kw.arg is None and isinstance(kw.value, ast.Name)
                        and kw.value.id in holders):
                    sites.append((n.lineno, kw.value.id))
    return holders, sites


def main():
    fails = []

    def chk(name, ok, detail=""):
        print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    # 1. default 0 emits nothing, so every existing cell result stays reproducible
    d0 = T._cell_params(_args(cell_leaf_iters=0))
    chk("1 default 0 does not emit leaf_estimation_iterations",
        "leaf_estimation_iterations" not in d0, str(d0))

    d10 = T._cell_params(_args(cell_leaf_iters=10))
    chk("1b value 10 emits exactly that key",
        d10.get("leaf_estimation_iterations") == 10, str(d10))

    # 2 + 3. both cell constructors receive the dict. This is the max_ctr_complexity
    #        guard: one site is not enough, and the count must be exactly two.
    holders, sites = _cell_param_call_sites()
    lines = sorted(ln for ln, _ in sites)
    chk("2 selection and refit cell constructors both splat _cell_params",
        len(sites) == 2 and len({h for _, h in sites}) == 2,
        f"holders={sorted(holders)} sites@lines={lines}")

    # 4. the flag cannot reach the base arm: the only writer is _cell_params, and
    #    the attribute is read exactly once in the whole trainer.
    src = open(SRC, encoding="utf-8").read()
    body = src.split("def _cell_params")[1].split("\ndef ")[0]
    chk("4 every read of args.cell_leaf_iters is inside _cell_params",
        src.count("args.cell_leaf_iters") == body.count("args.cell_leaf_iters") > 0,
        f"file={src.count('args.cell_leaf_iters')} in _cell_params={body.count('args.cell_leaf_iters')}")
    chk("4b nothing above _cell_params mentions leaf_estimation_iterations",
        "leaf_estimation_iterations" not in src.split("def _cell_params")[0])

    # 5. lineage records it, so a later provenance reconstruction can see it
    chk("5 lineage TRAINING_RELEVANT carries cell_leaf_iters",
        "cell_leaf_iters" in lineage.TRAINING_RELEVANT)
    flags = lineage.training_flags(_args(cell_leaf_iters=10))
    chk("5b training_flags reports the value",
        flags.get("cell_leaf_iters") == 10, str(flags))

    # 6 + 7. the effective parameter a real fitted model reports. A dict key is
    #        not evidence -- max_ctr_complexity had the key and the model said 4.
    from catboost import CatBoostClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = rng.integers(0, 12, size=400)

    def eff(extra):
        m = CatBoostClassifier(iterations=10, depth=3, loss_function="MultiClass",
                               classes_count=12, verbose=0, task_type="CPU",
                               **extra)
        m.fit(X, y)
        return m.get_all_params().get("leaf_estimation_iterations")

    ctrl = eff(T._cell_params(_args(cell_leaf_iters=0)))
    cand = eff(T._cell_params(_args(cell_leaf_iters=10)))
    chk("7 control fitted model reports the MultiClass default 1", ctrl == 1,
        f"got {ctrl}")
    chk("6 candidate fitted model reports 10", cand == 10, f"got {cand}")

    # 6b. The alignment target is 10 because that is what this project's binary
    #     arms have ALWAYS carried. Assert it from the artifacts, not from a
    #     synthetic fit: on a small synthetic frame CatBoost resolves the Logloss
    #     default to 1 (measured 2026-08-28 on catboost 1.2.10, both devices, all
    #     depths), because the default resolution depends on dataset shape --
    #     max_ctr_complexity came out 1 against the real data's 4 and
    #     data_partition DocParallel against FeatureParallel. On the real frame
    #     every Logloss/CrossEntropy/MultiLogloss artifact from 2026-08-07 to
    #     2026-08-16 reports 10 and every MultiClass artifact reports 1. So a
    #     synthetic assertion here would test the wrong thing and could fail for
    #     a reason that has nothing to do with this flag.
    import glob
    import joblib
    seen = {}
    for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..",
                                              "model", "cat_*_s3.pkl")))[:40]:
        try:
            m = joblib.load(path)["model"]
            a = m.get_all_params()
            seen.setdefault(a.get("loss_function"), set()).add(
                a.get("leaf_estimation_iterations"))
        except Exception:
            continue
    if not seen:
        chk("6b artifact survey found models", False, "no model/cat_*_s3.pkl loaded")
    else:
        chk("6b binary-family artifacts all carry 10 (the alignment target)",
            seen.get("Logloss") == {10}, f"Logloss -> {seen.get('Logloss')}")
        chk("6c MultiClass artifacts all carry 1 (what the cell arm has today)",
            seen.get("MultiClass") == {1}, f"MultiClass -> {seen.get('MultiClass')}")

    print(f"\n{len(fails)} failure(s)" if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
