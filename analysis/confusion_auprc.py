"""Confusion matrices and AUPRC with bootstrap CIs.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

HELD_OUT = "results/primary_comparisons_descriptive_summary.csv"
PRIMARY_NOTES = "work/_primary_endpoint_notes.csv"
OUT = "results/confusion_auprc.csv"

SEED = 82
N_BOOT = 5000

MODEL_ARMS = ["zero_shot__medgemma-27b-text-it", "B5_gold_21_features", "B5b_gold_21_plus_explicit"]
PRIMARY_MODELS = ["qwen2.5-7b-instruct", "sauerkrautlm-gemma-2-9b-it-i1"]


def bootstrap_auprc_ci(y_true, y_score, n_boot=N_BOOT, seed=SEED):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.empty(n_boot)
    for i in range(n_boot):
        yt, ys = y_true[idx[i]], y_score[idx[i]]
        vals[i] = average_precision_score(yt, ys) if yt.sum() > 0 else np.nan
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def main():
    rows = []

    # --- (1) held-out partition, already-computed confusion + AUPRC, reused ---
    desc = pd.read_csv(HELD_OUT)
    for arm in MODEL_ARMS:
        r = desc[desc["arm"] == arm].iloc[0]
        n = int(r["n"])
        prevalence = (r["TP"] + r["FN"]) / n
        rows.append({
            "cohort": "held_out_partition", "arm": arm, "n": n,
            "TP": int(r["TP"]), "FP": int(r["FP"]), "FN": int(r["FN"]), "TN": int(r["TN"]),
            "prevalence_no_skill_auprc_baseline": round(prevalence, 4),
            "AUPRC": round(r["AUPRC"], 4), "AUPRC_ci_lo": round(r["AUPRC_ci_lo"], 4),
            "AUPRC_ci_hi": round(r["AUPRC_ci_hi"], 4),
            "source": "primary_comparisons_descriptive_summary.csv (already on disk)",
        })
        print(f"[{arm}] n={n} TP={int(r.TP)} FP={int(r.FP)} FN={int(r.FN)} TN={int(r.TN)} "
              f"prevalence={prevalence:.4f} AUPRC={r.AUPRC:.4f} [{r.AUPRC_ci_lo:.4f},{r.AUPRC_ci_hi:.4f}]")

    # --- (2) primary endpoint, newly computed ---
    notes = pd.read_csv(PRIMARY_NOTES)[["CODINT", "gold", "bucket"]].rename(columns={"CODINT": "ID"})
    for model in PRIMARY_MODELS:
        risk = pd.read_csv(f"work/_primary_endpoint_risk_{model}.csv")
        wide = risk.pivot(index="CODINT", columns="corpus", values="risk").reset_index().rename(columns={"CODINT": "ID"})
        d = wide.merge(notes, on="ID", how="inner").dropna(subset=["raw", "masked"])
        y = d["gold"].to_numpy()
        n = len(d)
        prevalence = y.mean()
        for corpus in ["raw", "masked"]:
            score = d[corpus].to_numpy(dtype=float)
            pred = (score >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            auprc = average_precision_score(y, score)
            ci_lo, ci_hi = bootstrap_auprc_ci(y, score)
            rows.append({
                "cohort": f"primary_endpoint_{corpus}", "arm": model, "n": n,
                "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                "prevalence_no_skill_auprc_baseline": round(prevalence, 4),
                "AUPRC": round(auprc, 4), "AUPRC_ci_lo": round(ci_lo, 4), "AUPRC_ci_hi": round(ci_hi, 4),
                "source": "newly computed this task",
            })
            print(f"[primary_endpoint_{corpus} / {model}] n={n} TP={tp} FP={fp} FN={fn} TN={tn} "
                  f"prevalence={prevalence:.4f} AUPRC={auprc:.4f} [{ci_lo:.4f},{ci_hi:.4f}]")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\n[saved] -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
