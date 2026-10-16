"""Paired comparisons between arms (exact McNemar, bootstrap F1 difference, DeLong), calibration,
decision curve analysis and predictive values transported to lower prevalences.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss,
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

sys.path.insert(0, "analysis")
from baselines import (  # noqa: E402
    load_demo_and_text, redflag_families, AMBIGUOUS_IDS, SEED,
    CASI_XLSX, CONTROLLI_XLSX, EXPLICIT_FLAG, REDFLAG_DICT, GOLD_21,
    VARS_12CF, VARS_9RF,
)
import json

N_BOOT = 2000
RNG = np.random.default_rng(SEED)

ZERO_SHOT_GLOB_DIR = "saved_results/llm_zero_shot_prediction"
ZERO_SHOT_MODELS = [
    "medgemma-27b-text-it", "gpt-4o", "gpt-4.1", "gpt-4.1-mini",
    "DeepSeek-R1-Distill-Qwen-7B", "Llama-3-8B-UltraMedical",
    "MMed-Llama-3-8B", "Phi-4-mini-instruct", "Phi-4-mini-reasoning",
    "Qwen2.5-7B-Instruct", "SauerkrautLM-gemma-2-9b-it-i1",
]
PRIMARY_ZS_MODEL = "medgemma-27b-text-it"  # manuscript's stated headline model

OUT_PATIENT = "work/_primary_comparisons_patient_level.csv"
OUT_DESC = "results/primary_comparisons_descriptive_summary.csv"
OUT_PAIRED = "results/primary_comparisons_paired_tests.csv"
OUT_CAL = "results/primary_comparisons_calibration.csv"
OUT_PREV = "results/primary_comparisons_prevalence_transport.csv"
OUT_DCA = "results/primary_comparisons_dca.csv"


# --------------------------------------------------------------------------
# Stratified bootstrap (resamples positives and negatives separately, so
# every resample preserves the observed class prevalence)
# --------------------------------------------------------------------------
def stratified_bootstrap_ci(y_true, score, metric_fn, n_boot=N_BOOT, seed=SEED):
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        try:
            vals[i] = metric_fn(y_true[idx], score[idx])
        except Exception:
            vals[i] = np.nan
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_stratified_bootstrap_diff_ci(y_true, score_a, score_b, metric_fn, n_boot=N_BOOT, seed=SEED):
    """Same resample indices applied to both arms -> CI on the paired metric difference."""
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        try:
            diffs[i] = metric_fn(y_true[idx], score_a[idx]) - metric_fn(y_true[idx], score_b[idx])
        except Exception:
            diffs[i] = np.nan
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return float(np.nanmean(diffs)), float(lo), float(hi)


# --------------------------------------------------------------------------
# DeLong test for two correlated ROC AUCs (Sun & Xu 2014 fast algorithm)
# --------------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _delong_auc_var(y_true, score):
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    m, n = len(pos), len(neg)
    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)
    tz = _compute_midrank(np.concatenate([pos, neg]))
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    auc = tz[:m].sum() / (m * n) - (m + 1.0) / (2.0 * n)
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    var = s01 / m + s10 / n
    return auc, var, v01, v10, m, n


def delong_test(y_true, score_a, score_b):
    """Paired DeLong test for AUC(a) vs AUC(b) on the SAME patients."""
    y_true = np.asarray(y_true)
    auc_a, var_a, v01_a, v10_a, m, n = _delong_auc_var(y_true, score_a)
    auc_b, var_b, v01_b, v10_b, _, _ = _delong_auc_var(y_true, score_b)
    # covariance between the two AUCs via the structural components
    cov01 = np.cov(v01_a, v01_b)[0, 1] / m
    cov10 = np.cov(v10_a, v10_b)[0, 1] / n
    cov = cov01 + cov10
    var_diff = var_a + var_b - 2 * cov
    diff = auc_a - auc_b
    if var_diff <= 0:
        z = np.inf if diff != 0 else 0.0
    else:
        z = diff / np.sqrt(var_diff)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    se = np.sqrt(max(var_diff, 0))
    ci_lo, ci_hi = diff - 1.96 * se, diff + 1.96 * se
    return {"auc_a": auc_a, "auc_b": auc_b, "diff": diff, "z": z, "p": p,
            "diff_ci_lo": ci_lo, "diff_ci_hi": ci_hi}


# --------------------------------------------------------------------------
# Calibration: Brier score + calibration intercept/slope via logistic
# recalibration (Cox, 1958): logit(y) ~ a + b * logit(p_pred)
# --------------------------------------------------------------------------
def calibration_intercept_slope(y_true, p_pred, eps=1e-6):
    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    X = sm.add_constant(logit_p)
    model = sm.Logit(np.asarray(y_true), X).fit(disp=0)
    intercept, slope = model.params
    return float(intercept), float(slope)


# ==========================================================================
# 1. Load cohort, gold, B2/B5/B5b per-patient arms
# ==========================================================================
def build_baseline_arms():
    df = load_demo_and_text()
    n_before = len(df)
    df = df.dropna(subset=["age", "sex", "anamnesi"])

    explicit = pd.read_csv(EXPLICIT_FLAG)
    df = df.merge(explicit[["ID", "explicit_term"]], on="ID", how="left")
    df = df.dropna(subset=["explicit_term"])

    with open(REDFLAG_DICT, encoding="utf-8") as f:
        term_dict = json.load(f)
    df["n_redflag_families"] = df["anamnesi"].apply(lambda t: redflag_families(t, term_dict))
    df["b2_ge1"] = (df["n_redflag_families"] >= 1).astype(int)
    df["b2_ge2"] = (df["n_redflag_families"] >= 2).astype(int)
    df["b2_score"] = df["n_redflag_families"] / 9.0  # pseudo-continuous, for AUROC/DeLong only

    gold = pd.read_csv(GOLD_21).rename(columns={"ID": "ID_gold"})
    gold["ID_gold"] = gold["ID_gold"].astype(df["ID"].dtype, errors="ignore")
    dfg = df.merge(gold, left_on="ID", right_on="ID_gold", how="inner")
    for col in VARS_12CF + VARS_9RF:
        dfg[col] = dfg[col].fillna(0)
    yg = dfg["gold"].to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(random_state=SEED, max_iter=2000)

    X5 = dfg[["age", "male"] + VARS_12CF + VARS_9RF].to_numpy()
    dfg["b5_proba"] = cross_val_predict(clf, X5, yg, cv=cv, method="predict_proba")[:, 1]
    dfg["b5_pred"] = (dfg["b5_proba"] >= 0.5).astype(int)

    X5b = dfg[["age", "male"] + VARS_12CF + VARS_9RF + ["explicit_term"]].astype(float).to_numpy()
    dfg["b5b_proba"] = cross_val_predict(clf, X5b, yg, cv=cv, method="predict_proba")[:, 1]
    dfg["b5b_pred"] = (dfg["b5b_proba"] >= 0.5).astype(int)

    keep_cols = ["ID", "gold", "b2_ge1", "b2_ge2", "n_redflag_families", "b2_score",
                 "b5_proba", "b5_pred", "b5b_proba", "b5b_pred"]
    out = df.merge(dfg[["ID", "b5_proba", "b5_pred", "b5b_proba", "b5b_pred"]], on="ID", how="left")
    out = out[["ID", "gold", "b2_ge1", "b2_ge2", "n_redflag_families", "b2_score"]].merge(
        dfg[["ID", "b5_proba", "b5_pred", "b5b_proba", "b5b_pred"]], on="ID", how="left"
    )
    return out, n_before


def load_zero_shot(model):
    path = f"{ZERO_SHOT_GLOB_DIR}/amylo_predictions_{model}.csv"
    df = pd.read_csv(path)
    df = df[~df["ID"].isin(AMBIGUOUS_IDS)]
    n_total = len(df)
    df = df[(df["risk"] >= 0) & (df["risk"] <= 1)]
    n_valid = len(df)
    # Per-instruction: the file's own binary output has no defined operating
    # point and is not used. The binary call for confusion/F1/McNemar is
    # derived here from the continuous risk score at threshold 0.5.
    df = df.rename(columns={"amylo_gold": "gold", "risk": f"zs_risk__{model}"})
    df[f"zs_pred__{model}"] = (df[f"zs_risk__{model}"] >= 0.5).astype(int)
    return df[["ID", "gold", f"zs_risk__{model}", f"zs_pred__{model}"]], n_total, n_valid


def main():
    baseline_df, n_before = build_baseline_arms()
    print(f"[coverage] baseline-arm cohort (B2/B5/B5b, ambiguous IDs excluded): {len(baseline_df)}")

    zs_frames = {}
    zs_coverage = []
    for model in ZERO_SHOT_MODELS:
        df, n_total, n_valid = load_zero_shot(model)
        zs_frames[model] = df
        zs_coverage.append({"model": model, "n_rows_after_dedup": n_total,
                             "n_valid_risk_0_1": n_valid,
                             "n_dropped_out_of_range": n_total - n_valid})
    cov_df = pd.DataFrame(zs_coverage)
    print(cov_df.to_string(index=False))

    # ---- Descriptive: confusion / AUROC / AUPRC / bootstrap CI, all models in the primary comparisons ----
    desc_rows = []

    def add_desc(name, y_true, y_pred_binary, y_score_cont, n):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        f1 = f1_score(y_true, y_pred_binary)
        prec = precision_score(y_true, y_pred_binary)
        rec = recall_score(y_true, y_pred_binary)
        auroc = roc_auc_score(y_true, y_score_cont)
        auprc = average_precision_score(y_true, y_score_cont)
        f1_ci = stratified_bootstrap_ci(y_true, y_pred_binary, lambda yt, yp: f1_score(yt, yp))
        auroc_ci = stratified_bootstrap_ci(y_true, y_score_cont, lambda yt, sc: roc_auc_score(yt, sc))
        auprc_ci = stratified_bootstrap_ci(y_true, y_score_cont, lambda yt, sc: average_precision_score(yt, sc))
        desc_rows.append({
            "arm": name, "n": n, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "F1": f1, "F1_ci_lo": f1_ci[0], "F1_ci_hi": f1_ci[1],
            "precision": prec, "recall": rec,
            "AUROC": auroc, "AUROC_ci_lo": auroc_ci[0], "AUROC_ci_hi": auroc_ci[1],
            "AUPRC": auprc, "AUPRC_ci_lo": auprc_ci[0], "AUPRC_ci_hi": auprc_ci[1],
        })

    for model in ZERO_SHOT_MODELS:
        df = zs_frames[model]
        add_desc(f"zero_shot__{model}", df["gold"], df[f"zs_pred__{model}"],
                  df[f"zs_risk__{model}"], len(df))

    add_desc("B2_redflag_ge1", baseline_df["gold"], baseline_df["b2_ge1"],
              baseline_df["b2_score"], len(baseline_df))
    add_desc("B2_redflag_ge2", baseline_df["gold"], baseline_df["b2_ge2"],
              baseline_df["b2_score"], len(baseline_df))
    b5_mask = baseline_df["b5_proba"].notna()
    add_desc("B5_gold_21_features", baseline_df.loc[b5_mask, "gold"],
              baseline_df.loc[b5_mask, "b5_pred"], baseline_df.loc[b5_mask, "b5_proba"], b5_mask.sum())
    add_desc("B5b_gold_21_plus_explicit", baseline_df.loc[b5_mask, "gold"],
              baseline_df.loc[b5_mask, "b5b_pred"], baseline_df.loc[b5_mask, "b5b_proba"], b5_mask.sum())

    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(OUT_DESC, index=False)
    print("\n=== Descriptive (all models in the primary comparisons) ===")
    print(desc_df.to_string(index=False))

    # ---- Paired primary comparisons ----
    paired_rows = []
    zs_primary = zs_frames[PRIMARY_ZS_MODEL].rename(
        columns={f"zs_risk__{PRIMARY_ZS_MODEL}": "zs_risk", f"zs_pred__{PRIMARY_ZS_MODEL}": "zs_pred"})

    # Comparison 1: zero-shot (medgemma) vs 21-feature (B5)
    m1 = zs_primary.merge(baseline_df[["ID", "gold", "b5_proba", "b5_pred"]], on="ID", suffixes=("", "_b5"))
    m1 = m1.dropna(subset=["b5_proba"])
    y1 = m1["gold"].to_numpy()
    table1 = pd.crosstab(m1["zs_pred"], m1["b5_pred"])
    mcnemar1 = mcnemar(table1, exact=True)
    f1_diff1 = paired_stratified_bootstrap_diff_ci(y1, m1["zs_pred"], m1["b5_pred"], f1_score)
    delong1 = delong_test(y1, m1["zs_risk"], m1["b5_proba"])
    paired_rows.append({"comparison": "zero_shot_vs_21feature", "n": len(m1),
                         "mcnemar_stat": mcnemar1.statistic, "mcnemar_p": mcnemar1.pvalue,
                         "f1_diff": f1_diff1[0], "f1_diff_ci_lo": f1_diff1[1], "f1_diff_ci_hi": f1_diff1[2],
                         "delong_auc_a": delong1["auc_a"], "delong_auc_b": delong1["auc_b"],
                         "delong_diff": delong1["diff"], "delong_p": delong1["p"]})

    # Comparison 2: zero-shot (medgemma) vs B2 red-flag (>=1 family, primary threshold)
    m2 = zs_primary.merge(baseline_df[["ID", "gold", "b2_ge1", "b2_score"]], on="ID", suffixes=("", "_b2"))
    y2 = m2["gold"].to_numpy()
    table2 = pd.crosstab(m2["zs_pred"], m2["b2_ge1"])
    mcnemar2 = mcnemar(table2, exact=True)
    f1_diff2 = paired_stratified_bootstrap_diff_ci(y2, m2["zs_pred"], m2["b2_ge1"], f1_score)
    delong2 = delong_test(y2, m2["zs_risk"], m2["b2_score"])
    paired_rows.append({"comparison": "zero_shot_vs_B2_redflag", "n": len(m2),
                         "mcnemar_stat": mcnemar2.statistic, "mcnemar_p": mcnemar2.pvalue,
                         "f1_diff": f1_diff2[0], "f1_diff_ci_lo": f1_diff2[1], "f1_diff_ci_hi": f1_diff2[2],
                         "delong_auc_a": delong2["auc_a"], "delong_auc_b": delong2["auc_b"],
                         "delong_diff": delong2["diff"], "delong_p": delong2["p"]})

    # Comparison 3: masked vs raw zero-shot -- BLOCKED
    paired_rows.append({"comparison": "masked_vs_raw_zero_shot", "n": None,
                         "mcnemar_stat": None, "mcnemar_p": None,
                         "f1_diff": None, "f1_diff_ci_lo": None, "f1_diff_ci_hi": None,
                         "delong_auc_a": None, "delong_auc_b": None,
                         "delong_diff": None, "delong_p": None})

    paired_df = pd.DataFrame(paired_rows)
    # Holm-Bonferroni across the 3 pre-specified primary comparisons' DeLong p-value
    # (the masked-vs-raw slot is reserved but NA -- excluded from the correction's
    # denominator would understate m; we keep m=3 and report its slot as pending)
    pvals = paired_df["delong_p"].tolist()
    available = [p for p in pvals if p is not None]
    reject, p_adj_available, _, _ = multipletests(available, alpha=0.05, method="holm")
    p_adj_full, reject_full = [], []
    it = iter(zip(p_adj_available, reject))
    for p in pvals:
        if p is None:
            p_adj_full.append(None)
            reject_full.append(None)
        else:
            pa, rj = next(it)
            p_adj_full.append(pa)
            reject_full.append(bool(rj))
    paired_df["delong_p_holm_adj_m3"] = p_adj_full
    paired_df["reject_at_0.05_holm_m3"] = reject_full
    paired_df.to_csv(OUT_PAIRED, index=False)
    print("\n=== Paired primary comparisons (Holm-Bonferroni, m=3 reserved) ===")
    print(paired_df.to_string(index=False))

    # ---- Calibration (Brier, intercept, slope) on continuous scores only ----
    cal_rows = []
    for model in ZERO_SHOT_MODELS:
        df = zs_frames[model]
        y = df["gold"].to_numpy()
        p = df[f"zs_risk__{model}"].to_numpy()
        brier = brier_score_loss(y, p)
        try:
            intercept, slope = calibration_intercept_slope(y, p)
        except Exception as e:
            intercept, slope = np.nan, np.nan
        cal_rows.append({"arm": f"zero_shot__{model}", "n": len(df), "brier": brier,
                          "cal_intercept": intercept, "cal_slope": slope})
    y5 = baseline_df.loc[b5_mask, "gold"].to_numpy()
    p5 = baseline_df.loc[b5_mask, "b5_proba"].to_numpy()
    i5, s5 = calibration_intercept_slope(y5, p5)
    cal_rows.append({"arm": "B5_gold_21_features", "n": int(b5_mask.sum()),
                      "brier": brier_score_loss(y5, p5), "cal_intercept": i5, "cal_slope": s5})
    p5b = baseline_df.loc[b5_mask, "b5b_proba"].to_numpy()
    i5b, s5b = calibration_intercept_slope(y5, p5b)
    cal_rows.append({"arm": "B5b_gold_21_plus_explicit", "n": int(b5_mask.sum()),
                      "brier": brier_score_loss(y5, p5b), "cal_intercept": i5b, "cal_slope": s5b})
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(OUT_CAL, index=False)
    print("\n=== Calibration (Brier, intercept, slope) -- B2 excluded, not a probabilistic model ===")
    print(cal_df.to_string(index=False))

    # ---- Prevalence transport (1/5/10/13%) ----
    prev_rows = []
    prevalences = [0.01, 0.05, 0.10, 0.13]

    def transport(name, y_true, y_pred_binary, n):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        se = tp / (tp + fn)
        sp = tn / (tn + fp)
        for prev in prevalences:
            ppv = (se * prev) / (se * prev + (1 - sp) * (1 - prev))
            npv = (sp * (1 - prev)) / (sp * (1 - prev) + (1 - se) * prev)
            flagged_per_1000 = 1000 * (se * prev + (1 - sp) * (1 - prev))
            true_cases_per_1000 = 1000 * se * prev
            missed_per_1000 = 1000 * (1 - se) * prev
            nns = 1.0 / ppv if ppv > 0 else np.inf
            prev_rows.append({"arm": name, "n": n, "sensitivity": se, "specificity": sp,
                               "prevalence": prev, "PPV": ppv, "NPV": npv,
                               "flagged_per_1000_screened": flagged_per_1000,
                               "true_cases_detected_per_1000": true_cases_per_1000,
                               "cases_missed_per_1000": missed_per_1000,
                               "number_needed_to_scan": nns})

    for model in ZERO_SHOT_MODELS:
        df = zs_frames[model]
        transport(f"zero_shot__{model}", df["gold"], df[f"zs_pred__{model}"], len(df))
    transport("B2_redflag_ge1", baseline_df["gold"], baseline_df["b2_ge1"], len(baseline_df))
    transport("B5_gold_21_features", baseline_df.loc[b5_mask, "gold"], baseline_df.loc[b5_mask, "b5_pred"], b5_mask.sum())
    transport("B5b_gold_21_plus_explicit", baseline_df.loc[b5_mask, "gold"], baseline_df.loc[b5_mask, "b5b_pred"], b5_mask.sum())

    prev_df = pd.DataFrame(prev_rows)
    prev_df.to_csv(OUT_PREV, index=False)
    print("\n=== Prevalence transport (1/5/10/13%) ===")
    print(prev_df.to_string(index=False))

    # ---- Decision curve analysis, thresholds 1-20%, at 5% assumed prevalence ----
    dca_rows = []
    assumed_prev = 0.05
    thresholds = np.arange(0.01, 0.201, 0.01)

    def dca_for_continuous(name, y_true, score):
        y_true = np.asarray(y_true)
        score = np.asarray(score)
        for pt in thresholds:
            pred = (score >= pt).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            se = tp / (tp + fn) if (tp + fn) else 0.0
            sp = tn / (tn + fp) if (tn + fp) else 0.0
            nb = se * assumed_prev - (1 - sp) * (1 - assumed_prev) * (pt / (1 - pt))
            dca_rows.append({"arm": name, "threshold": pt, "net_benefit": nb})
        # reference strategies (prevalence-transported, threshold-dependent for screen-all)
        for pt in thresholds:
            nb_all = assumed_prev - (1 - assumed_prev) * (pt / (1 - pt))
            dca_rows.append({"arm": "screen_all", "threshold": pt, "net_benefit": nb_all})
            dca_rows.append({"arm": "screen_none", "threshold": pt, "net_benefit": 0.0})

    dca_for_continuous(f"zero_shot__{PRIMARY_ZS_MODEL}", zs_primary["gold"], zs_primary["zs_risk"])
    dca_for_continuous("B5_gold_21_features", baseline_df.loc[b5_mask, "gold"], baseline_df.loc[b5_mask, "b5_proba"])
    dca_for_continuous("B5b_gold_21_plus_explicit", baseline_df.loc[b5_mask, "gold"], baseline_df.loc[b5_mask, "b5b_proba"])

    dca_df = pd.DataFrame(dca_rows).drop_duplicates(subset=["arm", "threshold"])
    dca_df.to_csv(OUT_DCA, index=False)
    print(f"\n[saved] DCA -> {OUT_DCA} ({dca_df['arm'].nunique()} arms x {len(thresholds)} thresholds)")

    # Save patient-level arms (gitignored)
    merged_patient = baseline_df.copy()
    for model in ZERO_SHOT_MODELS:
        merged_patient = merged_patient.merge(zs_frames[model], on=["ID", "gold"], how="outer")
    merged_patient.to_csv(OUT_PATIENT, index=False)
    print(f"[saved] patient-level (gitignored) -> {OUT_PATIENT}")


if __name__ == "__main__":
    sys.exit(main())
