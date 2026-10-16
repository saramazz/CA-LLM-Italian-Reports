"""Age- and sex-matched sensitivity analysis (1:1, exact sex, nearest-neighbour age within a
caliper, without replacement) on already fitted predictions.
"""
import sys

import numpy as np
import pandas as pd
from baselines import load_demo_and_text, AMBIGUOUS_IDS
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

PATIENT_LEVEL = "work/_primary_comparisons_patient_level.csv"
OUT = "results/matched_analysis.csv"

SEED = 82
N_BOOT = 5000
CALIPER_YEARS = 5  # overridden per-run in main()


def build_matching_pool():
    demo = load_demo_and_text()[["ID", "age", "sex"]].dropna()
    pl = pd.read_csv(PATIENT_LEVEL)
    pool = pl.dropna(subset=["b5b_proba", "zs_risk__medgemma-27b-text-it"]).merge(demo, on="ID", how="inner")
    return pool


def greedy_match(pool, caliper_years):
    cases = pool[pool["gold"] == 1].sort_values("ID").copy()
    controls = pool[pool["gold"] == 0].sort_values("ID").copy()
    used_control_idx = set()
    pairs = []
    n_dropped = 0
    for _, case in cases.iterrows():
        candidates = controls[
            (~controls.index.isin(used_control_idx))
            & (controls["sex"] == case["sex"])
            & ((controls["age"] - case["age"]).abs() <= caliper_years)
        ]
        if candidates.empty:
            n_dropped += 1
            continue
        candidates = candidates.assign(age_diff=(candidates["age"] - case["age"]).abs())
        best = candidates.sort_values(["age_diff", "ID"]).iloc[0]
        used_control_idx.add(best.name)
        pairs.append((case["ID"], best["ID"]))
    return pairs, n_dropped, len(cases)


def bootstrap_metrics(y_true, y_pred, y_score, n_boot=N_BOOT, seed=SEED):
    y_true, y_pred, y_score = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    f1_arr = np.empty(n_boot)
    sens_arr = np.empty(n_boot)
    spec_arr = np.empty(n_boot)
    auc_arr = np.empty(n_boot)
    for i in range(n_boot):
        yt, yp, ys = y_true[idx[i]], y_pred[idx[i]], y_score[idx[i]]
        if yt.sum() == 0 or yt.sum() == len(yt):
            f1_arr[i] = sens_arr[i] = spec_arr[i] = auc_arr[i] = np.nan
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        f1_arr[i] = f1_score(yt, yp, zero_division=0)
        sens_arr[i] = tp / (tp + fn) if (tp + fn) else np.nan
        spec_arr[i] = tn / (tn + fp) if (tn + fp) else np.nan
        try:
            auc_arr[i] = roc_auc_score(yt, ys)
        except ValueError:
            auc_arr[i] = np.nan
    def ci(arr):
        return tuple(np.nanpercentile(arr, [2.5, 97.5]))
    return {
        "f1_ci": ci(f1_arr), "sens_ci": ci(sens_arr), "spec_ci": ci(spec_arr), "auc_ci": ci(auc_arr),
    }


def score_arm(name, matched, pred_col, score_col, caliper_years):
    y = matched["gold"].to_numpy()
    pred = matched[pred_col].to_numpy()
    score = matched[score_col].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    f1 = f1_score(y, pred, zero_division=0)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    auc = roc_auc_score(y, score)
    ci = bootstrap_metrics(y, pred, score)
    print(f"  [{name}] n={len(matched)} F1={f1:.4f} {ci['f1_ci']}  Sens={sens:.4f} {ci['sens_ci']}  "
          f"Spec={spec:.4f} {ci['spec_ci']}  AUROC={auc:.4f} {ci['auc_ci']}")
    return {
        "arm": name, "n": len(matched), "caliper_years": caliper_years,
        "f1": f1, "f1_ci_lo": ci["f1_ci"][0], "f1_ci_hi": ci["f1_ci"][1],
        "sensitivity": sens, "sensitivity_ci_lo": ci["sens_ci"][0], "sensitivity_ci_hi": ci["sens_ci"][1],
        "specificity": spec, "specificity_ci_lo": ci["spec_ci"][0], "specificity_ci_hi": ci["spec_ci"][1],
        "auroc": auc, "auroc_ci_lo": ci["auc_ci"][0], "auroc_ci_hi": ci["auc_ci"][1],
    }


def run_one_caliper(pool, caliper_years):
    n_cases_pool = int((pool["gold"] == 1).sum())
    n_controls_pool = int((pool["gold"] == 0).sum())
    print(f"\n=== CALIPER {caliper_years}y ===")
    print(f"[pool] n={len(pool)} (cases={n_cases_pool}, controls={n_controls_pool}), exact sex match")

    pairs, n_dropped, n_cases_total = greedy_match(pool, caliper_years)
    print(f"[matching] {len(pairs)} pairs formed, {n_dropped}/{n_cases_total} cases dropped for want of a match")

    matched_ids = set(c for c, _ in pairs) | set(k for _, k in pairs)
    matched = pool[pool["ID"].isin(matched_ids)].copy()
    print(f"[matched set] n={len(matched)} ({len(pairs)} cases + {len(pairs)} controls)")

    rows = []
    rows.append({"arm": "MATCHING_META", "n": len(matched), "n_pairs": len(pairs),
                  "n_cases_dropped": n_dropped, "n_cases_in_pool": n_cases_total,
                  "n_controls_in_pool": n_controls_pool, "caliper_years": caliper_years})

    for name, pred_col, score_col in [
        ("B5_gold_21_features", "b5_pred", "b5_proba"),
        ("B5b_gold_21_plus_explicit", "b5b_pred", "b5b_proba"),
        ("zero_shot__medgemma-27b-text-it", "zs_pred__medgemma-27b-text-it", "zs_risk__medgemma-27b-text-it"),
    ]:
        rows.append(score_arm(name, matched, pred_col, score_col, caliper_years))
    return rows


def main():
    pool = build_matching_pool()
    all_rows = []
    for caliper_years in (3, 5):
        all_rows += run_one_caliper(pool, caliper_years)

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT, index=False)
    print(f"\n[saved] -> {OUT} (both calipers; 3-year is the pre-specified/primary one, "
          f"5-year kept alongside per prior request)")


if __name__ == "__main__":
    sys.exit(main())
