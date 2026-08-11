"""실패모드 joint cell을 계층적 조건부 확률로 분해하는 로컬 파일럿.

기존 ``--failmode-cells``는 14개 joint cell을 한 번에 softmax로 맞춘다. 이 파일은

    P(middle|x)
    P(ball|x,middle)
    P(reverse|x,middle,ball)
    P(success|x,middle,ball,reverse)

를 네 이진 CatBoost로 맞춘 뒤, 추론 시 8개 (middle, ball, reverse) 상태를 행별로
전부 주변화한다. 검증/test의 실제 실패모드는 예측 입력에 사용하지 않는다.

같은 NPZ, 같은 seed, 같은 하이퍼파라미터로 flat 14-cell 대조군도 함께 학습한다.
따라서 다른 머신·전처리·표면 차이가 아니라 출력 기하 하나만 비교한다.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


def bss(y, p, center=False):
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    if center:
        p = p + (y.mean() - p.mean())
    r = y.mean()
    return float(100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1.0 - r))))


def make_frame(xn, xc, rows):
    """NPZ 행렬을 CatBoost 입력으로 복원한다. 범주 코드는 정수 범주다."""
    idx = np.flatnonzero(rows)
    df = pd.DataFrame(xn[idx], columns=[f"n{i}" for i in range(xn.shape[1])],
                      copy=False)
    for j in range(xc.shape[1]):
        df[f"c{j}"] = xc[idx, j]
    return df, idx


def model(seed, args, loss="Logloss", classes_count=None):
    kw = dict(
        iterations=args.iters,
        learning_rate=args.lr,
        depth=args.depth,
        l2_leaf_reg=args.l2,
        border_count=args.border_count,
        task_type="GPU",
        devices="0",
        loss_function=loss,
        eval_metric=loss,
        random_seed=seed,
        early_stopping_rounds=args.es,
        allow_writing_files=False,
        verbose=args.verbose,
    )
    if classes_count is not None:
        kw["classes_count"] = classes_count
    return CatBoostClassifier(**kw)


def p1(m, frame):
    return np.asarray(m.predict_proba(frame), np.float64)[:, 1]


def add_conditions(frame, values, names):
    out = frame.copy(deep=False)
    for j, name in enumerate(names):
        out[name] = np.asarray(values[:, j], np.int8)
    return out


def marginalize(models, frame):
    """실패모드 관측 없이 8개 상태를 합산해 P(success|x)를 계산한다."""
    pm = p1(models[0], frame)
    ans = np.zeros(len(frame), np.float64)
    for m, b, r in itertools.product((0, 1), repeat=3):
        mm = np.full((len(frame), 1), m, np.int8)
        xb = add_conditions(frame, mm, ["_middle"])
        pb = p1(models[1], xb)

        mb = np.column_stack((mm[:, 0], np.full(len(frame), b, np.int8)))
        xr = add_conditions(frame, mb, ["_middle", "_ball"])
        pr = p1(models[2], xr)

        mbr = np.column_stack((mb, np.full(len(frame), r, np.int8)))
        xy = add_conditions(frame, mbr, ["_middle", "_ball", "_reverse"])
        py = p1(models[3], xy)

        w = (pm if m else 1.0 - pm)
        w = w * (pb if b else 1.0 - pb)
        w = w * (pr if r else 1.0 - pr)
        ans += w * py
    return np.clip(ans, 0.0, 1.0)


def report(name, y, p):
    print(f"{name:<18} BSS={bss(y, p):9.3f} centered={bss(y, p, True):9.3f} "
          f"mean={np.mean(p):.6f} y={np.mean(y):.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="out/mtnn_pitch_2324.npz")
    ap.add_argument("--tag", default="HFC1_s42")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--border-count", type=int, default=128)
    ap.add_argument("--es", type=int, default=120)
    ap.add_argument("--verbose", type=int, default=100)
    ap.add_argument("--skip-flat", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    z = np.load(args.npz, allow_pickle=True)
    xn, xc = z["Xn"], z["Xc"]
    y = z["y"].astype(np.float64)
    aux = z["aux"].astype(np.float64)
    cell = z["cell"].astype(np.int64)
    success = z["cell_success"].astype(np.int64)
    is_val, is_test = z["is_val"], z["is_test"]
    is_train = ~(is_val | is_test)
    known = np.isfinite(aux).all(1)
    cat = [f"c{i}" for i in range(xc.shape[1])]
    print(f"rows train/val/test={is_train.sum():,}/{is_val.sum():,}/{is_test.sum():,} "
          f"known={known.mean():.6f} num/cat={xn.shape[1]}/{xc.shape[1]}", flush=True)

    xtr, itr = make_frame(xn, xc, is_train)
    xva, iva = make_frame(xn, xc, is_val)
    xte, ite = make_frame(xn, xc, is_test)
    flat_val = flat_test = None

    if not args.skip_flat:
        print("\n[flat 14-cell control]", flush=True)
        flat = model(args.seed, args, "MultiClass", len(z["cell_names"]))
        flat.fit(Pool(xtr, cell[itr], cat_features=cat),
                 eval_set=Pool(xva, cell[iva], cat_features=cat))
        flat_val = flat.predict_proba(xva)[:, success].sum(1)
        flat_test = flat.predict_proba(xte)[:, success].sum(1)
        print(f"flat best_iter={flat.get_best_iteration()}", flush=True)
        report("val2023 flat", y[iva], flat_val)
        report("test2024 flat", y[ite], flat_test)
        del flat
        gc.collect()

    print("\n[hierarchical conditional chain]", flush=True)
    models = []
    labels = [aux[:, 0], aux[:, 1], aux[:, 2], y]
    cond_names = [[], ["_middle"], ["_middle", "_ball"],
                  ["_middle", "_ball", "_reverse"]]
    head_names = ["middle", "ball|middle", "reverse|middle,ball",
                  "success|middle,ball,reverse"]
    for h, (target, cond, head_name) in enumerate(zip(labels, cond_names, head_names)):
        tr_ok = known[itr] if h else np.isfinite(target[itr])
        va_ok = known[iva] if h else np.isfinite(target[iva])
        tr_cond = aux[itr, :h] if h else np.empty((len(itr), 0), np.int8)
        va_cond = aux[iva, :h] if h else np.empty((len(iva), 0), np.int8)
        xh_tr = add_conditions(xtr.loc[tr_ok], tr_cond[tr_ok], cond)
        xh_va = add_conditions(xva.loc[va_ok], va_cond[va_ok], cond)
        cats = cat + cond
        clf = model(args.seed + h * 1009, args)
        clf.fit(Pool(xh_tr, target[itr][tr_ok].astype(np.int8), cat_features=cats),
                eval_set=Pool(xh_va, target[iva][va_ok].astype(np.int8),
                              cat_features=cats))
        print(f"head {head_name:<31} best_iter={clf.get_best_iteration()}", flush=True)
        models.append(clf)
        del xh_tr, xh_va
        gc.collect()

    chain_val = marginalize(models, xva)
    chain_test = marginalize(models, xte)
    report("val2023 chain", y[iva], chain_val)
    report("test2024 chain", y[ite], chain_test)

    if flat_val is not None:
        print("\n[paired geometry delta]", flush=True)
        print(f"val chain-flat  {bss(y[iva], chain_val)-bss(y[iva], flat_val):+.3f} "
              f"centered {bss(y[iva], chain_val, True)-bss(y[iva], flat_val, True):+.3f}")
        print(f"test chain-flat {bss(y[ite], chain_test)-bss(y[ite], flat_test):+.3f} "
              f"centered {bss(y[ite], chain_test, True)-bss(y[ite], flat_test, True):+.3f}")
        print(f"RMS flat-chain val/test "
              f"{np.sqrt(np.mean((flat_val-chain_val)**2)):.6f}/"
              f"{np.sqrt(np.mean((flat_test-chain_test)**2)):.6f}")
        for w in (0.25, 0.5, 0.75):
            pv = (1-w)*flat_val + w*chain_val
            pt = (1-w)*flat_test + w*chain_test
            print(f"fixed blend chain_w={w:.2f}: val "
                  f"{bss(y[iva],pv)-bss(y[iva],flat_val):+.3f}, test "
                  f"{bss(y[ite],pt)-bss(y[ite],flat_test):+.3f}")

    os.makedirs("out", exist_ok=True)
    np.savez_compressed(
        f"out/{args.tag}_preds.npz",
        val_y=y[iva], val_flat=flat_val, val_chain=chain_val,
        test_y=y[ite], test_flat=flat_test, test_chain=chain_test,
        seed=args.seed, args=np.array(vars(args), dtype=object),
    )
    print(f"saved out/{args.tag}_preds.npz | elapsed={(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
