"""Supplementary table of zero-shot F1 at the 0.5 threshold with the count of patients without
a usable prediction, by cause.
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _patient_ids import AMBIGUOUS_IDS  # noqa: E402

REPO = "."
PRED_GLOB = os.path.join(REPO, "saved_results/llm_zero_shot_prediction/amylo_predictions_*.csv")
METRICS = os.path.join(REPO, "results/zero_shot_threshold05.csv")
OUT = os.path.join(REPO, "results/zero_shot_threshold05_all_models.csv")
PROPRIETARY = {"gpt-4.1-mini", "gpt-4.1", "gpt-4o"}


def main():
    metrics = pd.read_csv(METRICS).set_index("model")
    rows = []
    for f in sorted(glob.glob(PRED_GLOB)):
        model = f.split("amylo_predictions_")[1].removesuffix(".csv")
        df = pd.read_csv(f)
        df = df[~df["ID"].isin(AMBIGUOUS_IDS)]
        cohort = df["amylo_gold"].notna().sum()
        df = df[df["amylo_gold"].notna()]
        no_risk = df["risk"].isna().sum()
        no_binary = (df["risk"].notna() & df["amylo_predicted_llm"].isna()).sum()
        df = df.dropna(subset=["amylo_predicted_llm", "risk"])
        out_of_range = (~df["risk"].between(0, 1)).sum()
        n = int(df["risk"].between(0, 1).sum())
        m = metrics.loc[model]
        assert n == int(m["n"]), (model, n, m["n"])
        rows.append({
            "model": model,
            "type": "proprietary" if model in PROPRIETARY else "locally deployable",
            "f1": m["f1_05"], "f1_ci_lo": m["f1_ci_lo"], "f1_ci_hi": m["f1_ci_hi"],
            "n": n, "n_ca": int(m["tp"] + m["fn"]), "n_controls": int(m["fp"] + m["tn"]),
            "rows_in_file": int(cohort),
            "no_prediction_total": int(cohort - n),
            "no_risk_score": int(no_risk), "no_binary_output": int(no_binary),
            "risk_outside_0_1": int(out_of_range),
        })
    t = pd.DataFrame(rows).sort_values("f1", ascending=False)
    t.to_csv(OUT, index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
