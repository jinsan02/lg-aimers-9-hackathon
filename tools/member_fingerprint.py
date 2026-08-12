"""Check that blend members came from **the same training data and hyperparameters**.

On 2026-08-08 we treated `VB_base` as a reproduction of `cat_v14f` and were about to
pick blend weights from its predictions. It was a different, 52-point-weaker model —
a `--drop-f-pre 2022` had leaked into the handoff command, so the training data
differed. `fpipe['priors']` inside the pkl is computed from the training rows, which
makes it a **fingerprint of the training set**: same value, same data.

We made this mistake in v16 (-6.15), again in v17 (-53.6), and again here.

Usage: python tools/member_fingerprint.py v14f ZD5 DX_seq
"""

import glob
import sys

import joblib

try:                                   # survive a cp949 Windows console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

KEYS = ("learning_rate", "depth", "l2_leaf_reg", "loss_function", "border_count")


def main(tags):
    if not tags:
        tags = sorted({f.split("cat_")[1].rsplit("_s", 1)[0]
                       for f in glob.glob("./model/cat_*_s*.pkl")})
    rows = []
    for t in tags:
        fs = sorted(glob.glob(f"./model/cat_{t}_s*.pkl"))
        if not fs:
            print(f"not found: {t}")
            continue
        d = joblib.load(fs[0])
        p = d["model"].get_all_params()
        rows.append((t, len(fs),
                     d["fpipe"]["priors"]["asof_pitcher_success_rate"],
                     len(d["features"]),
                     tuple(round(p[k], 6) if isinstance(p.get(k), float)
                           else p.get(k) for k in KEYS),
                     d["best_iteration"], d["val_bss"]))

    print(f"{'tag':<12}{'seeds':>6}{'train-set fingerprint':>24}{'feat':>6}"
          f"{'best_it':>9}{'val_bss':>10}")
    for t, n, pr, nf, hp, bi, vb in rows:
        print(f"{t:<12}{n:>6}{pr:>24.10f}{nf:>6}{bi:>9}{vb:>10.2f}")

    prs = {r[2] for r in rows}
    hps = {r[4] for r in rows}
    print()
    if len(prs) > 1:
        print("!! Fingerprints differ -- these members were trained on different data.")
        print("   Do not pick blend weights across them. Compare the flags.")
    else:
        print("Training-set fingerprint matches")
    if len(hps) > 1:
        print("note: hyperparameters differ (a different loss_function is expected)")
        for t, _, _, _, hp, _, _ in rows:
            print(f"   {t:<12}{hp}")
    return 2 if len(prs) > 1 else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
