"""Baseline sanity table for a B0/B1 lineage, plus the fixed 0.45/0.55 core.

This is the "new zero point" report the takeover plan asks for. It refuses to
print a comparison unless the two members really are comparable: same seeds,
same host, same row set. Matching seed numbers alone has let mismatched arms
through before.

`.45/.55` is not a re-optimised weight. It is the legacy constant, held fixed so
a candidate's delta is about the candidate.

  python tools/b0_report.py B0JL_base B0JL_cell
  python tools/b0_report.py B0JL_base B0JL_cell --cand MYARM_base
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from invalidated import guard as _guard_invalidated

W_CELL = 0.55


def _seed(path: str) -> str:
    return path.rsplit("_s", 1)[1].split("_", 1)[0]


def load(tag: str, split: str) -> dict[str, dict]:
    out = {}
    for p in sorted(glob.glob(f"./out/*_{tag}_s*_{split}_preds.npz")):
        z = np.load(p, allow_pickle=True)
        out[_seed(p)] = {
            "pred": z["pred"].astype(np.float64),
            "y": z["y"].astype(np.float64),
            "row_id": z["row_id"] if "row_id" in z.files else None,
            "host": str(z["host"]) if "host" in z.files else "?",
            "surface": str(z["surface"]) if "surface" in z.files else "?",
        }
    return out


def raw_bss(y, p):
    r = float(y.mean())
    return 1e5 * (1 - float(np.mean((np.clip(p, 0, 1) - y) ** 2)) / (r * (1 - r)))


def centered(y, p):
    return raw_bss(y, p - (p.mean() - y.mean()))


def agree(a: dict, b: dict, what: str) -> list[str]:
    """Common seeds, but only where host / surface / row set actually match."""
    ok = []
    for s in sorted(set(a) & set(b), key=int):
        if a[s]["host"] != b[s]["host"]:
            print(f"  !! seed {s}: host {a[s]['host']} vs {b[s]['host']} -- dropped")
            continue
        if a[s]["surface"] != b[s]["surface"]:
            print(f"  !! seed {s}: surface {a[s]['surface']} vs {b[s]['surface']} -- dropped")
            continue
        ra, rb = a[s]["row_id"], b[s]["row_id"]
        if ra is None or rb is None:
            print(f"  !! seed {s}: {what} missing row_id -- cannot verify alignment, dropped")
            continue
        if not np.array_equal(ra, rb):
            print(f"  !! seed {s}: row sets differ -- dropped")
            continue
        ok.append(s)
    return ok


def table(name: str, arms: dict[str, dict], seeds: list[str]) -> np.ndarray:
    print(f"\n  {name:<10}{'seed':>6}{'raw BSS':>11}{'centered':>11}"
          f"{'MSE':>10}{'pred-y':>10}")
    vals = []
    for s in seeds:
        y, p = arms[s]["y"], arms[s]["pred"]
        vals.append(raw_bss(y, p))
        print(f"  {'':<10}{s:>6}{vals[-1]:>11.2f}{centered(y, p):>11.2f}"
              f"{float(np.mean((p - y) ** 2)):>10.5f}{p.mean() - y.mean():>+10.5f}")
    v = np.array(vals)
    print(f"  {'':<10}{'mean':>6}{v.mean():>11.2f}"
          f"   sd {v.std(ddof=1) if len(v) > 1 else 0:.2f}")
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_tag")
    ap.add_argument("cell_tag")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--cand", default="", help="candidate tag replacing base")
    a = ap.parse_args()
    _guard_invalidated([a.base_tag, a.cell_tag] + ([a.cand] if a.cand else []))

    base, cell = load(a.base_tag, a.split), load(a.cell_tag, a.split)
    if not base or not cell:
        print(f"missing predictions: base {len(base)} / cell {len(cell)}")
        return 1
    seeds = agree(base, cell, "base/cell")
    if not seeds:
        print("no comparable seeds")
        return 1
    print(f"\n=== {a.base_tag} + {a.cell_tag} | {a.split} | seeds {','.join(seeds)} "
          f"| host {base[seeds[0]]['host']} | {len(base[seeds[0]]['y']):,} rows ===")

    vb = table("base", base, seeds)
    vc = table("cell", cell, seeds)

    y = base[seeds[0]]["y"]
    core = np.array([raw_bss(y, (1 - W_CELL) * base[s]["pred"] + W_CELL * cell[s]["pred"])
                     for s in seeds])
    ens = raw_bss(y, np.mean([(1 - W_CELL) * base[s]["pred"] + W_CELL * cell[s]["pred"]
                              for s in seeds], axis=0))
    print(f"\n  core .45/.55   per-seed mean {core.mean():>8.2f}   "
          f"ensemble {ens:>8.2f}")

    for lin in (f"./out/lineage_{t}.json" for t in (a.base_tag, a.cell_tag)):
        if os.path.exists(lin):
            d = json.load(open(lin, encoding="utf-8"))
            trees = {k: v["refit_trees"] for k, v in d["seeds"].items()}
            print(f"  {os.path.basename(lin):<28} feat {d['features']['fingerprint']} "
                  f"| rows fit {d['rows']['fit']:,} val {d['rows']['val']:,} "
                  f"| refit trees {trees}")

    bad = [s for s in seeds
           if not np.isfinite(base[s]["pred"]).all()
           or base[s]["pred"].std() < 1e-6
           or raw_bss(y, base[s]["pred"]) < 0]
    print(f"\n  collapse check: {'FAIL ' + str(bad) if bad else 'ok'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
