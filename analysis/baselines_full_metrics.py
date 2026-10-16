"""Full metrics (specificity, AUROC) for the baselines.
"""
import sys

import numpy as np
import pandas as pd
from baselines import (
    load_demo_and_text, redflag_families, bootstrap_ci, fit_cv,
    N_BOOT, RNG, SEED,
    EXPLICIT_FLAG, REDFLAG_DICT, GOLD_21, VARS_12CF, VARS_9RF,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import json

OUT = "results/baselines_full_metrics.csv"


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def full_row(name, y_true, y_pred, y_score, n, f1_expected=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score, dtype=float)

    f1 = f1_score(y_true, y_pred)
    if f1_expected is not None:
        assert abs(f1 - f1_expected) < 1e-9, (
            f"[{name}] F1 mismatch vs baseline_ladder_summary.csv: "
            f"{f1:.6f} (here) vs {f1_expected:.6f} (stored) -- fold assignment diverged"
        )
    spec = specificity_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_score)

    f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, f1_score)
    spec_lo, spec_hi = bootstrap_ci(y_true, y_pred, specificity_score)
    auroc_lo, auroc_hi = bootstrap_ci(y_true, y_score, roc_auc_score)

    print(f"[{name}] n={n}  F1={f1:.4f} [{f1_lo:.4f},{f1_hi:.4f}]  "
          f"Specificity={spec:.4f} [{spec_lo:.4f},{spec_hi:.4f}]  "
          f"AUROC={auroc:.4f} [{auroc_lo:.4f},{auroc_hi:.4f}]")
    return {
        "baseline": name, "n": n,
        "f1": f1, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
        "specificity": spec, "specificity_ci_lo": spec_lo, "specificity_ci_hi": spec_hi,
        "auroc": auroc, "auroc_ci_lo": auroc_lo, "auroc_ci_hi": auroc_hi,
    }


def fit_cv_proba(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(random_state=SEED, max_iter=2000)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def main():
    stored = pd.read_csv("results/baseline_ladder_summary.csv").set_index("baseline")["f1"]
    rows = []

    df = load_demo_and_text()
    df = df.dropna(subset=["age", "sex", "anamnesi"])
    explicit = pd.read_csv(EXPLICIT_FLAG)
    df = df.merge(explicit[["ID", "explicit_term"]], on="ID", how="left")
    df = df.dropna(subset=["explicit_term"])
    y = df["gold"].to_numpy()

    # --- B1 ---
    pred_b1 = df["explicit_term"].astype(int).to_numpy()
    rows.append(full_row("B1_explicit_term", y, pred_b1, pred_b1, len(df),
                          f1_expected=stored["B1_explicit_term"]))

    # --- B2 ---
    with open(REDFLAG_DICT, encoding="utf-8") as f:
        term_dict = json.load(f)
    df["n_redflag_families"] = df["anamnesi"].apply(lambda t: redflag_families(t, term_dict))
    score_b2 = df["n_redflag_families"].to_numpy(dtype=float)
    for k in (1, 2):
        pred = (df["n_redflag_families"] >= k).astype(int).to_numpy()
        rows.append(full_row(f"B2_redflag_ge{k}_families", y, pred, score_b2, len(df),
                              f1_expected=stored[f"B2_redflag_ge{k}_families"]))

    # --- B3 ---
    X3 = df[["age", "male"]].to_numpy()
    pred_b3, score_b3 = fit_cv_proba(X3, y)
    rows.append(full_row("B3_age_sex", y, pred_b3, score_b3, len(df),
                          f1_expected=stored["B3_age_sex"]))

    # --- B4 / B5 / B5b ---
    gold = pd.read_csv(GOLD_21).rename(columns={"ID": "ID_gold"})
    gold["ID_gold"] = gold["ID_gold"].astype(df["ID"].dtype, errors="ignore")
    dfg = df.merge(gold, left_on="ID", right_on="ID_gold", how="inner")
    yg = dfg["gold"].to_numpy()
    for col in VARS_12CF + VARS_9RF:
        dfg[col] = dfg[col].fillna(0)

    X4 = dfg[["age", "male"] + VARS_12CF].to_numpy()
    pred_b4, score_b4 = fit_cv_proba(X4, yg)
    rows.append(full_row("B4_demographics_plus_12cf", yg, pred_b4, score_b4, len(dfg),
                          f1_expected=stored["B4_demographics_plus_12cf"]))

    X5 = dfg[["age", "male"] + VARS_12CF + VARS_9RF].to_numpy()
    pred_b5, score_b5 = fit_cv_proba(X5, yg)
    rows.append(full_row("B5_gold_21_features", yg, pred_b5, score_b5, len(dfg),
                          f1_expected=stored["B5_gold_21_features"]))

    X5b = dfg[["age", "male"] + VARS_12CF + VARS_9RF + ["explicit_term"]].astype(float).to_numpy()
    pred_b5b, score_b5b = fit_cv_proba(X5b, yg)
    rows.append(full_row("B5b_gold_21_plus_explicit_flag", yg, pred_b5b, score_b5b, len(dfg),
                          f1_expected=stored["B5b_gold_21_plus_explicit_flag"]))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\n[saved] -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
