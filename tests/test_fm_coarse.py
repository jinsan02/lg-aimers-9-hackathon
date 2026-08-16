"""`--fm-coarse failure` must change only which classes exist, never the binary
meaning. Run against the real train frame, because the whole point is the
taxonomy the real data produces.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import failmode as fm

COLS = ["row_id", "season", "pitcher_id", "control_success", "asof_pitcher_n",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_reverse_rate"]


def _frame():
    """Real rows, in file order -- the labels come from differencing asof
    counters within a pitcher, so a shuffled or sampled frame is meaningless."""
    p = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
    return pd.read_csv(p, usecols=COLS)


def main():
    fails = []

    def chk(name, ok, detail=""):
        print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    tr = _frame()
    is_val = (tr["season"] == 2024).to_numpy()
    fit = ~is_val

    code_f, names_f, succ_f = fm.build_cells(tr, verbose=False, fit_mask=fit)
    code_c, names_c, succ_c = fm.build_cells(tr, verbose=False, fit_mask=fit,
                                             coarse="failure")

    # 1. the default path is untouched
    code_d, names_d, succ_d = fm.build_cells(tr, verbose=False, fit_mask=fit,
                                             coarse="")
    chk("1 coarse='' is bit-identical to the default",
        np.array_equal(code_d.to_numpy(), code_f.to_numpy())
        and names_d == names_f and succ_d == succ_f)

    # 2. exactly four classes, and they are the expected ones
    chk("2 four classes with the expected names", names_c == ["0", "1000", "1010", "1xxx"],
        str(names_c))

    # 3. success set has three members and they are the non-'0' names
    chk("3 succ = the three success-bit classes",
        succ_c == {i for i, v in enumerate(names_c) if v[0] == "1"},
        f"succ={sorted(succ_c)}")

    # 4. no empty class -- the reserve name must not sit unused, or classes_
    #    desynchronises from names and success_prob indexes the wrong columns
    used = set(np.unique(code_c.to_numpy()))
    chk("4 every class index is taken by at least one row",
        used == set(range(len(names_c))),
        f"used={sorted(used)} of {len(names_c)}")

    # 5. the binary meaning is preserved row by row
    succ_row_f = np.isin(code_f.to_numpy(), sorted(succ_f))
    succ_row_c = np.isin(code_c.to_numpy(), sorted(succ_c))
    chk("5 the success bit of every row is unchanged",
        np.array_equal(succ_row_f, succ_row_c),
        f"differ on {int((succ_row_f != succ_row_c).sum())} rows")

    # 6. every failure row lands in one class
    failure_codes = set(np.unique(code_c.to_numpy()[~succ_row_c]))
    chk("6 the whole failure block is a single class", len(failure_codes) == 1,
        f"codes={sorted(failure_codes)}")

    # 7. the summation identity still holds for an arbitrary proba matrix
    rng = np.random.default_rng(0)
    P = rng.random((5000, len(names_c)))
    P /= P.sum(1, keepdims=True)
    want = P[:, [i for i, v in enumerate(names_c) if v[0] == "1"]].sum(1)
    chk("7 success_prob == sum over success-bit classes",
        np.allclose(fm.success_prob(P, succ_c), want, atol=0, rtol=0))

    # 8. the success side is untouched: the same rows carry the same success cell
    m = succ_row_c
    nf = np.array(names_f, dtype=object)[code_f.to_numpy()[m]]
    nc = np.array(names_c, dtype=object)[code_c.to_numpy()[m]]
    chk("8 success rows keep their exact cell name", bool((nf == nc).all()),
        f"differ on {int((nf != nc).sum())} of {int(m.sum())}")

    # 9. an unknown mode is refused rather than silently ignored
    try:
        fm.build_cells(tr.head(1000), verbose=False, coarse="bogus")
        chk("9 unknown coarse mode raises", False)
    except ValueError:
        chk("9 unknown coarse mode raises", True)

    # 10. share of the collapsed block, reported so the change is visible
    share = float((~succ_row_c).mean())
    chk("10 failure block is a plausible majority-ish share",
        0.40 < share < 0.60, f"share={share:.4f}")

    print(f"\n{len(fails)} failure(s)" if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
