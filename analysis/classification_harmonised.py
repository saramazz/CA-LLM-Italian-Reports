"""Classification of CA from the 21 variables of each feature source (gold standard and every
model extraction) under one protocol: LR, RF and XGBoost with fixed hyperparameters, 5-fold
stratified cross-validation with pooled out-of-fold predictions, seed 82, bootstrap CIs.
"""
import glob
import sys

import numpy as np
import pandas as pd
from _patient_ids import AMBIGUOUS_IDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier

CASI_XLSX = "database/unlabeled/amylo/Casi_new_noname.xlsx"
CONTROLLI_XLSX = "database/unlabeled/amylo/Controlli_new_noname.xlsx"
GOLD_21_RAW = "database/labeled/amylo_garf_gold_labelstudio.csv"
GOLD_21_REBUILT = "work/_gold_rebuild/amylo_garf_gold_labelstudio_rebuilt12cf.csv"
LLM_GLOB = "database/labeled/amylo_casi_controlli/llm_annotations_*.csv"
OUT = "results/classification_harmonised_summary.csv"

CLASSIFIERS = {
    "LR": lambda: LogisticRegression(random_state=SEED, max_iter=2000),
    "RF": lambda: RandomForestClassifier(random_state=SEED, n_estimators=300),
    "XGB": lambda: XGBClassifier(random_state=SEED, eval_metric="logloss", n_estimators=300),
}

SEED = 82
N_BOOT = 5000
RNG = np.random.default_rng(SEED)

VARS_12CF = ["HY", "DY", "DB", "CODP", "NYHA", "EF", "RS", "FA", "FL", "PM", "BBSx", "BBDx"]
VARS_9RF = ["TNC", "RTS", "SCM", "PDA", "PDG", "IPC", "DSM", "PAR", "IPGMM"]
ALL_21 = VARS_12CF + VARS_9RF


# Excluded: synthetic-named test files (not patient extractions;
# casi_controlli_merged_gold.csv), duplicate-of-spacy files, and files already
# established as one of a same-model case-duplicate pair without a
# params_log (no recorded provenance) -- keep only the provenanced one per model.
EXCLUDE_SUBSTR = ["synthetic", "spacy"]
PREFER_LOWERCASE_DUPES = {"qwen2.5-7b-instruct", "deepseek-r1-distill-qwen-7b", "sauerkrautlm-gemma-2-9b-it-i1"}


def load_cohort():
    casi = pd.read_excel(CASI_XLSX, sheet_name="Casi")[["CODINT"]]
    casi["gold"] = 1
    ctrl = pd.read_excel(CONTROLLI_XLSX, sheet_name="AMYLO Dati CONTROLLI")[["CODINT"]]
    ctrl["gold"] = 0
    df = pd.concat([casi, ctrl], ignore_index=True).rename(columns={"CODINT": "ID"})
    df = df[~df["ID"].isin(AMBIGUOUS_IDS)]
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


def score(name, clf_name, y_true, y_pred, n):
    f1 = f1_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, f1_score)
    print(f"[{name:32s} / {clf_name:3s}] n={n:3d}  F1={f1:.4f} [{f1_lo:.4f},{f1_hi:.4f}]  P={p:.4f}  R={r:.4f}")
    return {"source": name, "classifier": clf_name, "n": n, "f1": f1, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
            "precision": p, "recall": r}


def fit_cv(X, y, clf_name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = CLASSIFIERS[clf_name]()
    return cross_val_predict(clf, X, y, cv=cv)


def pick_llm_files():
    files = {}
    for f in sorted(glob.glob(LLM_GLOB)):
        base = f.split("llm_annotations_")[1].removesuffix(".csv")
        if any(s in base.lower() for s in EXCLUDE_SUBSTR):
            continue
        key = base.lower()
        if key in files and key in PREFER_LOWERCASE_DUPES:
            # keep whichever of the two case-variants IS the lowercase one
            # (only the lower-case files have a recorded parameter log)
            if base == key:
                files[key] = f
            continue
        files[key] = f
    return files


def main():
    cohort = load_cohort()
    rows = []

    # --- gold standard with explicit negations recovered (gold_rebuild.py) ---
    gold = pd.read_csv(GOLD_21_REBUILT).rename(columns={"ID": "ID_src"})
    dfg = cohort.merge(gold, left_on="ID", right_on="ID_src", how="inner")
    for col in ALL_21:
        dfg[col] = dfg[col].fillna(0)
    Xg = dfg[ALL_21].to_numpy()
    yg = dfg["gold"].to_numpy()
    for clf_name in CLASSIFIERS:
        pred_g = fit_cv(Xg, yg, clf_name)
        rows.append(score("gold_21_rebuilt (reference)", clf_name, yg, pred_g, len(dfg)))

    # --- each LLM-structured dataset, identical protocol, all 3 classifiers ---
    llm_files = pick_llm_files()
    print(f"[coverage] {len(llm_files)} distinct LLM-structured datasets after de-duplication: {sorted(llm_files)}")

    for model, path in sorted(llm_files.items()):
        src = pd.read_csv(path).rename(columns={"ID": "ID_src"})
        dfm = cohort.merge(src, left_on="ID", right_on="ID_src", how="inner")
        if len(dfm) < 30 or dfm["gold"].nunique() < 2:
            print(f"[skip] {model}: n={len(dfm)}, insufficient for stratified CV")
            continue
        for col in ALL_21:
            dfm[col] = dfm[col].fillna(0)
        Xm = dfm[ALL_21].to_numpy()
        ym = dfm["gold"].to_numpy()
        for clf_name in CLASSIFIERS:
            pred_m = fit_cv(Xm, ym, clf_name)
            rows.append(score(model, clf_name, ym, pred_m, len(dfm)))

    out = pd.DataFrame(rows).sort_values("f1", ascending=False)
    out.to_csv(OUT, index=False)
    print(f"\n[saved] -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
