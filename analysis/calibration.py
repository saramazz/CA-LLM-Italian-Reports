"""Calibration of the arms producing a continuous score: slope, intercept, Brier score,
Spiegelhalter's z and binned calibration tables.
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import brier_score_loss

PATIENT_LEVEL = "work/_primary_comparisons_patient_level.csv"
OUT_BINS = "results/calibration_bins.csv"
OUT_SUMMARY = "results/calibration_summary.csv"
N_BINS = 10


def calibration_intercept_slope(y_true, p_pred, eps=1e-6):
    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    X = sm.add_constant(logit_p)
    model = sm.Logit(np.asarray(y_true), X).fit(disp=0)
    intercept, slope = model.params
    return float(intercept), float(slope)


def spiegelhalter_z(y_true, p_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    num = np.sum((y - p) * (1 - 2 * p))
    var = np.sum(((1 - 2 * p) ** 2) * p * (1 - p))
    z = num / np.sqrt(var)
    pval = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(pval)


def decile_bins(y_true, p_pred, n_bins=N_BINS):
    df = pd.DataFrame({"y": y_true, "p": p_pred})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    rows = []
    for i, (interval, sub) in enumerate(df.groupby("bin", observed=True), start=1):
        rows.append({
            "bin": i, "bin_range_lo": interval.left, "bin_range_hi": interval.right,
            "n": len(sub), "n_observed_positive": int(sub["y"].sum()),
            "observed_rate": sub["y"].mean(), "expected_rate_mean_predicted": sub["p"].mean(),
        })
    return pd.DataFrame(rows)


def process_arm(name, y_true, p_pred):
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred, dtype=float)
    n = len(y_true)
    print(f"\n[{name}] n={n}")

    intercept, slope = calibration_intercept_slope(y_true, p_pred)
    brier = brier_score_loss(y_true, p_pred)
    z, pval = spiegelhalter_z(y_true, p_pred)
    print(f"  calibration intercept={intercept:.4f} slope={slope:.4f} "
          f"Brier={brier:.4f} Spiegelhalter z={z:.4f} p={pval:.4g}")

    bins = decile_bins(y_true, p_pred)
    bins.insert(0, "arm", name)
    print(bins.to_string(index=False))

    summary = {"arm": name, "n": n, "calibration_intercept": intercept,
               "calibration_slope": slope, "brier_score": brier,
               "spiegelhalter_z": z, "spiegelhalter_p": pval}
    return summary, bins


def main():
    pl = pd.read_csv(PATIENT_LEVEL)

    b5b = pl.dropna(subset=["b5b_proba"])
    s1, bins1 = process_arm("B5b_gold_21_plus_explicit", b5b["gold"], b5b["b5b_proba"])

    zs = pl.dropna(subset=["zs_risk__medgemma-27b-text-it"])
    s2, bins2 = process_arm("zero_shot__medgemma-27b-text-it", zs["gold"], zs["zs_risk__medgemma-27b-text-it"])

    pd.DataFrame([s1, s2]).to_csv(OUT_SUMMARY, index=False)
    pd.concat([bins1, bins2], ignore_index=True).to_csv(OUT_BINS, index=False)
    print(f"\n[saved] -> {OUT_SUMMARY}")
    print(f"[saved] -> {OUT_BINS}")


if __name__ == "__main__":
    sys.exit(main())
