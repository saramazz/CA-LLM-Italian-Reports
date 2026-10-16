"""Masked anamneses for the explicit-term-present and explicit-term-absent strata.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, "analysis")
sys.path.insert(0, "analysis")
from baselines import load_demo_and_text, EXPLICIT_FLAG  # noqa: E402
from leakage_spec import mask_text  # noqa: E402

OUT = "work/_lexical_arm_masked_current_spec.csv"


def build():
    df = load_demo_and_text().dropna(subset=["age", "sex", "anamnesi"])
    flag = pd.read_csv(EXPLICIT_FLAG)[["ID", "explicit_term"]]
    df = df.merge(flag, on="ID", how="left")
    masked = df["anamnesi"].astype(str).map(mask_text)
    return pd.DataFrame({"ID": df["ID"], "gold": df["gold"], "explicit_term": df["explicit_term"],
                         "masked_anamnesi": masked.map(lambda r: r[0]),
                         "n_masked_spans": masked.map(lambda r: r[1])})


if __name__ == "__main__":
    out = build()
    print(f"n={len(out)}, explicit present={int(out['explicit_term'].sum())}, "
          f"masked spans={int(out['n_masked_spans'].sum())}")
    if "--write" in sys.argv:
        out.to_csv(OUT, index=False)
