"""Read-only audit for row-local structure hidden in anonymized player IDs.

This deliberately never uses another evaluation row.  It only asks whether pieces of
the ID string define cohorts whose historical control rate transfers to a later year.
"""

from __future__ import annotations

import pandas as pd


def summary(df: pd.DataFrame, col: str) -> None:
    s = df[col].astype(str)
    print(f"\n[{col}] unique={s.nunique()} lengths={s.str.len().value_counts().to_dict()}")
    print("examples:", ", ".join(s.drop_duplicates().head(30)))
    for name, code in {
        "prefix1": s.str[:1],
        "prefix2": s.str[:2],
        "prefix3": s.str[:3],
        "suffix1": s.str[-1:],
        "suffix2": s.str[-2:],
    }.items():
        z = df.assign(code=code).groupby("code", observed=True).agg(
            rows=("control_success", "size"),
            players=(col, "nunique"),
            rate=("control_success", "mean"),
            first=("season", "min"),
            last=("season", "max"),
        )
        z = z[z["rows"] >= 1000].sort_values("rows", ascending=False)
        spread = float(z["rate"].max() - z["rate"].min()) if len(z) else float("nan")
        print(f"\n{name}: groups={len(z)} rate_spread={spread:.6f}")
        print(z.head(25).to_string())


def main() -> None:
    df = pd.read_csv("data/train.csv", usecols=[
        "pitcher_id", "batter_id", "season", "control_success"
    ])
    print(df.head(20).to_string(index=False))
    for col in ("pitcher_id", "batter_id"):
        summary(df, col)


if __name__ == "__main__":
    main()
