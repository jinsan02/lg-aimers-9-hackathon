"""Print actual CatBoost feature names and aggregate importance families."""

import argparse
import os

import joblib
import pandas as pd


def family(name):
    if name.startswith("te_"):
        return "te"
    if name.startswith("std_") and name.endswith("_delta"):
        return "std_delta"
    if name.startswith("std_"):
        if "batter" in name:
            return "std_batter"
        if any(x in name for x in ("fastball", "breaking", "offspeed", "pitchmix")):
            return "std_pitchmix"
        return "std_pitcher"
    if name.startswith("skill_"):
        return "skill"
    if name.startswith(("dom_", "x_")):
        return "domain"
    return "raw_or_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--bottom", type=int, default=40)
    args = ap.parse_args()
    pack = joblib.load(args.model)
    model = pack["model"]
    names = list(model.feature_names_)
    values = model.get_feature_importance()
    tab = pd.DataFrame({"feature": names, "importance": values})
    tab["family"] = tab.feature.map(family)
    print(f"model={os.path.basename(args.model)} features={len(tab)}")
    print("\n=== families ===")
    print(tab.groupby("family").agg(n=("feature", "size"),
                                     importance=("importance", "sum"))
          .sort_values("importance", ascending=False).to_string())
    print("\n=== bottom features ===")
    print(tab.sort_values("importance").head(args.bottom).to_string(index=False))
    print("\n=== generated feature names ===")
    print("\n".join(tab.loc[tab.family != "raw_or_v2", "feature"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
