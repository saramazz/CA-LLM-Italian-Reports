"""Baselines B1-B5b: explicit-term rule, red-flag rules, logistic regression on age and sex, on
the 12 routine variables, on the 21 gold variables, and on the 21 variables plus the
explicit-term flag.
"""
import json
import sys

import numpy as np
import pandas as pd
from _patient_ids import AMBIGUOUS_IDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score

CASI_XLSX = "database/unlabeled/amylo/Casi_new_noname.xlsx"
CONTROLLI_XLSX = "database/unlabeled/amylo/Controlli_new_noname.xlsx"
EXPLICIT_FLAG = "work/_explicit_flag_main_text.csv"
REDFLAG_DICT = "work/_redflag_term_dictionary.json"
GOLD_21 = "database/labeled/amylo_garf_gold_labelstudio.csv"
OUT = "results/baseline_ladder_summary.csv"

SEED = 82
N_BOOT = 5000
RNG = np.random.default_rng(SEED)

VARS_12CF = ["HY", "DY", "DB", "CODP", "NYHA", "EF", "RS", "FA", "FL", "PM", "BBSx", "BBDx"]
VARS_9RF = ["TNC", "RTS", "SCM", "PDA", "PDG", "IPC", "DSM", "PAR", "IPGMM"]


# 2 CODINT values (found while building this script) appear in BOTH sheets with
# opposite labels (one Casi row, one Controlli row) -- a patient cannot be both a
# case and a control. They are excluded
# item flag a different, within-registry duplicate). Excluded from every analysis
# in this file rather than arbitrarily assigned a label.


def load_demo_and_text():
    casi = pd.read_excel(CASI_XLSX, sheet_name="Casi")[["CODINT", "sex", "age", "anamnesi"]]
    casi["gold"] = 1
    ctrl = pd.read_excel(CONTROLLI_XLSX, sheet_name="AMYLO Dati CONTROLLI")[["CODINT", "sex", "age", "anamnesi"]]
    ctrl["gold"] = 0
    df = pd.concat([casi, ctrl], ignore_index=True).rename(columns={"CODINT": "ID"})
    n_ambiguous = df["ID"].isin(AMBIGUOUS_IDS).sum()
    print(f"[coverage] dropping {n_ambiguous} rows for {len(AMBIGUOUS_IDS)} cross-labeled (case AND control) IDs")
    df = df[~df["ID"].isin(AMBIGUOUS_IDS)]
    df["male"] = (df["sex"] == 1).astype(float)
    return df


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=N_BOOT, rng=RNG):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.empty(n_boot)
    for i in range(n_boot):
        yt, yp = y_true[idx[i]], y_pred[idx[i]]
        vals[i] = metric_fn(yt, yp) if yt.sum() > 0 else np.nan
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return lo, hi


def score(name, y_true, y_pred, n):
    f1 = f1_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, f1_score)
    p_lo, p_hi = bootstrap_ci(y_true, y_pred, precision_score)
    r_lo, r_hi = bootstrap_ci(y_true, y_pred, recall_score)
    print(f"[{name}] n={n}  F1={f1:.4f} [{f1_lo:.4f},{f1_hi:.4f}]  "
          f"P={p:.4f} [{p_lo:.4f},{p_hi:.4f}]  R={r:.4f} [{r_lo:.4f},{r_hi:.4f}]")
    return {"baseline": name, "n": n, "f1": f1, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
            "precision": p, "precision_ci_lo": p_lo, "precision_ci_hi": p_hi,
            "recall": r, "recall_ci_lo": r_lo, "recall_ci_hi": r_hi}


def redflag_families(text, term_dict):
    t = " ".join(str(text).lower().split())
    hit = 0
    for fam, terms in term_dict.items():
        if any(term in t for term in terms):
            hit += 1
    return hit


def fit_cv(X, y, n):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(random_state=SEED, max_iter=2000)
    return cross_val_predict(clf, X, y, cv=cv)


def main():
    rows = []

    df = load_demo_and_text()
    n_before = len(df)
    df = df.dropna(subset=["age", "sex", "anamnesi"])
    print(f"[coverage] complete demographics+text rows: {len(df)}/{n_before}")

    explicit = pd.read_csv(EXPLICIT_FLAG)  # ID, gold, explicit_term
    df = df.merge(explicit[["ID", "explicit_term"]], on="ID", how="left")
    n_no_flag = df["explicit_term"].isna().sum()
    print(f"[coverage] missing explicit-term flag: {n_no_flag}/{len(df)} (dropped)")
    df = df.dropna(subset=["explicit_term"])

    # --- B1: explicit-term rule ---
    y = df["gold"].to_numpy()
    pred_b1 = df["explicit_term"].astype(int).to_numpy()
    rows.append(score("B1_explicit_term", y, pred_b1, len(df)))

    # --- B2: red-flag rule, >=1 and >=2 families ---
    with open(REDFLAG_DICT, encoding="utf-8") as f:
        term_dict = json.load(f)
    df["n_redflag_families"] = df["anamnesi"].apply(lambda t: redflag_families(t, term_dict))
    for k in (1, 2):
        pred = (df["n_redflag_families"] >= k).astype(int).to_numpy()
        rows.append(score(f"B2_redflag_ge{k}_families", y, pred, len(df)))

    # --- B3: age + sex ---
    X3 = df[["age", "male"]].to_numpy()
    pred_b3 = fit_cv(X3, y, len(df))
    rows.append(score("B3_age_sex", y, pred_b3, len(df)))

    # --- B4 / B5 / B5b: needs the 21-variable gold table ---
    gold = pd.read_csv(GOLD_21).rename(columns={"ID": "ID_gold"})
    gold["ID_gold"] = gold["ID_gold"].astype(df["ID"].dtype, errors="ignore")
    dfg = df.merge(gold, left_on="ID", right_on="ID_gold", how="inner")
    print(f"[coverage] rows with 21-variable gold available: {len(dfg)}/{len(df)}")
    yg = dfg["gold"].to_numpy()

    for col in VARS_12CF + VARS_9RF:
        dfg[col] = dfg[col].fillna(0)

    X4 = dfg[["age", "male"] + VARS_12CF].to_numpy()
    pred_b4 = fit_cv(X4, yg, len(dfg))
    rows.append(score("B4_demographics_plus_12cf", yg, pred_b4, len(dfg)))

    X5 = dfg[["age", "male"] + VARS_12CF + VARS_9RF].to_numpy()
    pred_b5 = fit_cv(X5, yg, len(dfg))
    rows.append(score("B5_gold_21_features", yg, pred_b5, len(dfg)))

    X5b = dfg[["age", "male"] + VARS_12CF + VARS_9RF + ["explicit_term"]].astype(float).to_numpy()
    pred_b5b = fit_cv(X5b, yg, len(dfg))
    rows.append(score("B5b_gold_21_plus_explicit_flag", yg, pred_b5b, len(dfg)))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\n[saved] -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
