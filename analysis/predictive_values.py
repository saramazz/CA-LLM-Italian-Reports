"""PPV, NPV and number needed to scan at 1, 2 and 5 percent prevalence and in the study cohort.
"""
import sys

import numpy as np
import pandas as pd
from baselines import load_demo_and_text, EXPLICIT_FLAG, bootstrap_ci
from sklearn.metrics import confusion_matrix

PATIENT_LEVEL = "work/_primary_comparisons_patient_level.csv"
OUT = "results/predictive_values.csv"

SEED = 82
N_BOOT = 5000
LITERATURE_PREVALENCES = [0.01, 0.02, 0.05]


def sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return sens, spec


def bootstrap_sens_spec_pairs(y_true, y_pred, n_boot=N_BOOT, seed=SEED):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    sens_arr = np.empty(n_boot)
    spec_arr = np.empty(n_boot)
    for i in range(n_boot):
        yt, yp = y_true[idx[i]], y_pred[idx[i]]
        if yt.sum() == 0 or yt.sum() == len(yt):
            sens_arr[i], spec_arr[i] = np.nan, np.nan
            continue
        sens_arr[i], spec_arr[i] = sens_spec(yt, yp)
    return sens_arr, spec_arr


def ppv_npv_nns(sens, spec, prev):
    ppv = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
    npv = (spec * (1 - prev)) / (spec * (1 - prev) + (1 - sens) * prev)
    nns = 1.0 / ppv if ppv and ppv > 0 else np.nan
    return ppv, npv, nns


def process_arm(name, y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    n_ca = int(y_true.sum())
    own_prev = n_ca / n
    sens, spec = sens_spec(y_true, y_pred)
    sens_arr, spec_arr = bootstrap_sens_spec_pairs(y_true, y_pred)
    sens_lo, sens_hi = np.nanpercentile(sens_arr, [2.5, 97.5])
    spec_lo, spec_hi = np.nanpercentile(spec_arr, [2.5, 97.5])

    print(f"\n[{name}] n={n} CA={n_ca} own-cohort prevalence={own_prev:.4f} "
          f"Sens={sens:.4f} [{sens_lo:.4f},{sens_hi:.4f}] "
          f"Spec={spec:.4f} [{spec_lo:.4f},{spec_hi:.4f}]")

    prevalences = LITERATURE_PREVALENCES + [own_prev]
    prev_labels = ["1pct", "2pct", "5pct", "own_cohort_held_out_partition"]
    rows = []
    for label, prev in zip(prev_labels, prevalences):
        ppv, npv, nns = ppv_npv_nns(sens, spec, prev)
        boot_ppv = np.empty(N_BOOT)
        boot_npv = np.empty(N_BOOT)
        boot_nns = np.empty(N_BOOT)
        for i in range(N_BOOT):
            s, sp = sens_arr[i], spec_arr[i]
            if np.isnan(s) or np.isnan(sp):
                boot_ppv[i] = boot_npv[i] = boot_nns[i] = np.nan
                continue
            boot_ppv[i], boot_npv[i], boot_nns[i] = ppv_npv_nns(s, sp, prev)
        ppv_lo, ppv_hi = np.nanpercentile(boot_ppv, [2.5, 97.5])
        npv_lo, npv_hi = np.nanpercentile(boot_npv, [2.5, 97.5])
        nns_lo, nns_hi = np.nanpercentile(boot_nns, [2.5, 97.5])
        rows.append({
            "arm": name, "n": n, "n_ca": n_ca,
            "prevalence_label": label, "prevalence_value": prev,
            "sensitivity": sens, "sensitivity_ci_lo": sens_lo, "sensitivity_ci_hi": sens_hi,
            "specificity": spec, "specificity_ci_lo": spec_lo, "specificity_ci_hi": spec_hi,
            "PPV": ppv, "PPV_ci_lo": ppv_lo, "PPV_ci_hi": ppv_hi,
            "NPV": npv, "NPV_ci_lo": npv_lo, "NPV_ci_hi": npv_hi,
            "NNS_1_over_PPV": nns, "NNS_ci_lo": nns_lo, "NNS_ci_hi": nns_hi,
        })
        print(f"    [{label}={prev:.4f}] PPV={ppv:.4f} [{ppv_lo:.4f},{ppv_hi:.4f}]  "
              f"NPV={npv:.4f} [{npv_lo:.4f},{npv_hi:.4f}]  NNS={nns:.2f} [{nns_lo:.2f},{nns_hi:.2f}]")
    return rows


def main():
    all_rows = []

    # --- B1 ---
    df = load_demo_and_text()
    df = df.dropna(subset=["age", "sex", "anamnesi"])
    explicit = pd.read_csv(EXPLICIT_FLAG)
    df = df.merge(explicit[["ID", "explicit_term"]], on="ID", how="left")
    df = df.dropna(subset=["explicit_term"])
    all_rows += process_arm("B1_explicit_term", df["gold"], df["explicit_term"].astype(int))

    # --- B5 / B5b / medgemma zero-shot ---
    pl = pd.read_csv(PATIENT_LEVEL)
    b5 = pl.dropna(subset=["b5_proba"])
    all_rows += process_arm("B5_gold_21_features", b5["gold"], b5["b5_pred"].astype(int))

    b5b = pl.dropna(subset=["b5b_proba"])
    all_rows += process_arm("B5b_gold_21_plus_explicit", b5b["gold"], b5b["b5b_pred"].astype(int))

    zs = pl.dropna(subset=["zs_risk__medgemma-27b-text-it"])
    all_rows += process_arm("zero_shot__medgemma-27b-text-it", zs["gold"],
                             zs["zs_pred__medgemma-27b-text-it"].astype(int))

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT, index=False)
    print(f"\n[saved] -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
