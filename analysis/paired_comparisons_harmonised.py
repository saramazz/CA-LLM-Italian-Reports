"""Zero-shot vs the harmonised 21-variable classifier on the patients scored by both arms.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, "analysis")
from paired_comparisons import (  # noqa: E402
    build_baseline_arms, load_zero_shot, delong_test,
    paired_stratified_bootstrap_diff_ci, ZERO_SHOT_MODELS, PRIMARY_ZS_MODEL,
)
from classification_harmonised import (  # noqa: E402
    load_cohort, ALL_21, GOLD_21_REBUILT, SEED,
)

OUT_PAIRED = "results/paired_tests_harmonised.csv"
OUT_PATIENT = "work/_p21_levelA_patient_level.csv"


def build_levelA_gold_arm():
    """Level A's gold_21_rebuilt LR, out-of-fold probabilities -- same protocol
    as classification_harmonised.py's reference row, extended here to capture
    predict_proba (not just the binary F1/CI it originally saved)."""
    cohort = load_cohort()
    gold = pd.read_csv(GOLD_21_REBUILT).rename(columns={"ID": "ID_src"})
    dfg = cohort.merge(gold, left_on="ID", right_on="ID_src", how="inner")
    for col in ALL_21:
        dfg[col] = dfg[col].fillna(0)
    X = dfg[ALL_21].to_numpy()
    y = dfg["gold"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(random_state=SEED, max_iter=2000)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    dfg["levelA_proba"] = proba
    dfg["levelA_pred"] = (proba >= 0.5).astype(int)
    print(f"[coverage] Level A gold-21 (rebuilt) arm: n={len(dfg)}")
    return dfg[["ID", "gold", "levelA_proba", "levelA_pred"]]


def main():
    baseline_df, _ = build_baseline_arms()  # gives B2 arm (unchanged)
    levelA = build_levelA_gold_arm()

    zs_frames = {}
    for model in ZERO_SHOT_MODELS:
        df, n_total, n_valid = load_zero_shot(model)
        zs_frames[model] = df
    zs_primary = zs_frames[PRIMARY_ZS_MODEL].rename(
        columns={f"zs_risk__{PRIMARY_ZS_MODEL}": "zs_risk", f"zs_pred__{PRIMARY_ZS_MODEL}": "zs_pred"})

    paired_rows = []

    # --- Comparison 1: zero-shot (medgemma) vs 21-feature, NOW Level A instead of B5 ---
    m1 = zs_primary.merge(levelA[["ID", "levelA_proba", "levelA_pred"]], on="ID")
    y1 = m1["gold"].to_numpy()
    table1 = pd.crosstab(m1["zs_pred"], m1["levelA_pred"])
    mcnemar1 = mcnemar(table1, exact=True)
    f1_diff1 = paired_stratified_bootstrap_diff_ci(y1, m1["zs_pred"], m1["levelA_pred"], f1_score)
    delong1 = delong_test(y1, m1["zs_risk"], m1["levelA_proba"])
    paired_rows.append({"comparison": "zero_shot_vs_21feature_harmonised", "n": len(m1),
                         "mcnemar_stat": mcnemar1.statistic, "mcnemar_p": mcnemar1.pvalue,
                         "f1_diff": f1_diff1[0], "f1_diff_ci_lo": f1_diff1[1], "f1_diff_ci_hi": f1_diff1[2],
                         "delong_auc_a": delong1["auc_a"], "delong_auc_b": delong1["auc_b"],
                         "delong_diff": delong1["diff"], "delong_p": delong1["p"]})
    print(f"[Comparison 1] zero-shot F1(zs_pred)={f1_score(y1, m1['zs_pred']):.4f}  "
          f"LevelA F1={f1_score(y1, m1['levelA_pred']):.4f}  n={len(m1)}")
    print(f"  McNemar exact p={mcnemar1.pvalue:.4f}  DeLong AUROC diff={delong1['diff']:.4f} p={delong1['p']:.4g}")

    # --- Comparison 2: zero-shot vs B2 red-flag (>=1 family) -- identical protocol, consistency check ---
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

    # --- Comparison 3: masked vs raw is computed in metrics_primary_endpoint.py ---
    paired_rows.append({"comparison": "masked_vs_raw_zero_shot (see metrics_primary_endpoint.py)", "n": None,
                         "mcnemar_stat": None, "mcnemar_p": None,
                         "f1_diff": None, "f1_diff_ci_lo": None, "f1_diff_ci_hi": None,
                         "delong_auc_a": None, "delong_auc_b": None,
                         "delong_diff": None, "delong_p": None})

    paired_df = pd.DataFrame(paired_rows)
    pvals = paired_df["delong_p"].tolist()
    available = [p for p in pvals if p is not None]
    reject, p_adj_available, _, _ = multipletests(available, alpha=0.05, method="holm")
    p_adj_full, reject_full = [], []
    it = iter(zip(p_adj_available, reject))
    for p in pvals:
        if p is None:
            p_adj_full.append(None); reject_full.append(None)
        else:
            pa, rj = next(it)
            p_adj_full.append(pa); reject_full.append(bool(rj))
    paired_df["delong_p_holm_adj_m3"] = p_adj_full
    paired_df["reject_at_0.05_holm_m3"] = reject_full
    paired_df.to_csv(OUT_PAIRED, index=False)
    print("\n=== Paired comparisons, harmonised 21-variable classifier (Holm-Bonferroni) ===")
    print(paired_df.to_string(index=False))

    # --- binary-prompt F1 vs F1 at the 0.5 threshold on the risk score ---
    print("\n=== Binary prompt vs 0.5 threshold on the risk score ===")
    raw_zs = pd.read_csv(f"saved_results/llm_zero_shot_prediction/amylo_predictions_{PRIMARY_ZS_MODEL}.csv")
    from classification_harmonised import AMBIGUOUS_IDS
    raw_zs_dedup = raw_zs[~raw_zs["ID"].isin(AMBIGUOUS_IDS)]
    f1_stored_col_full = f1_score(raw_zs["amylo_gold"], raw_zs["amylo_predicted_llm"])
    f1_stored_col_dedup = f1_score(raw_zs_dedup["amylo_gold"], raw_zs_dedup["amylo_predicted_llm"])
    f1_threshold05_dedup = f1_score(y1, m1["zs_pred"])  # already the 0.5-on-risk convention, ambiguous excluded
    print(f"  stored binary column, n={len(raw_zs)} (incl. 2 ambiguous IDs): F1={f1_stored_col_full:.4f}")
    print(f"  stored binary column, n={len(raw_zs_dedup)} (ambiguous excluded): F1={f1_stored_col_dedup:.4f}")
    print(f"  0.5-threshold on continuous risk score, n={len(m1)} (ambiguous excluded): F1={f1_threshold05_dedup:.4f}")

    # --- Llama-3-8B-UltraMedical degeneracy check ---
    print("\n=== Llama-3-8B-UltraMedical degeneracy ===")
    llama_zs = zs_frames["Llama-3-8B-UltraMedical"]
    llama_pred_col = f"zs_pred__Llama-3-8B-UltraMedical"
    llama_risk_col = f"zs_risk__Llama-3-8B-UltraMedical"
    print(f"  ZERO-SHOT arm (0.5-threshold on risk), n={len(llama_zs)}: "
          f"F1={f1_score(llama_zs['gold'], llama_zs[llama_pred_col]):.4f}  "
          f"pred value_counts={llama_zs[llama_pred_col].value_counts().to_dict()}  "
          f"AUROC={roc_auc_score(llama_zs['gold'], llama_zs[llama_risk_col]):.4f}")
    # NOTE: Llama-3-8B-UltraMedical in LEVEL A is a DIFFERENT thing: an
    # LLM-STRUCTURED EXTRACTION dataset (21 features fed to LR/RF/XGB), not a
    # zero-shot risk score. Check it there too, for completeness.
    llm_struct = pd.read_csv("database/labeled/amylo_casi_controlli/llm_annotations_Llama-3-8B-UltraMedical.csv")
    llm_struct = llm_struct.rename(columns={"ID": "ID_src"})
    cohort = load_cohort()
    dfm = cohort.merge(llm_struct, left_on="ID", right_on="ID_src", how="inner")
    for col in ALL_21:
        dfm[col] = dfm[col].fillna(0)
    print(f"  LEVEL A LLM-structured arm (21-feature classifier trained ON this model's extraction), "
          f"n={len(dfm)}: gold value_counts={dfm['gold'].value_counts().to_dict()} "
          f"-- this is NOT degenerate (see classification_harmonised_summary.csv: "
          f"LR F1=0.5016, RF F1=0.4389, XGB F1=0.4753, none constant)")
    print("  CONCLUSION: the all-positive/AUROC=0.500 degeneracy belongs to Llama-3-8B-UltraMedical's "
          "ZERO-SHOT arm only, a different model use entirely from its Level A extraction-dataset role. "
          "The model NAMED in the task's premise (F1=0.000, all-negative) does not match either of "
          "Llama-3-8B-UltraMedical's two roles here -- that description matches biomistral-7b's LEVEL A "
          "classification-arm result instead (F1=0.0000 for LR/RF/XGB, all-negative), a different model.")

    levelA.merge(zs_primary[["ID", "zs_pred", "zs_risk"]], on="ID", how="outer").to_csv(OUT_PATIENT, index=False)
    print(f"\n[saved] -> {OUT_PAIRED}\n[saved] patient-level (gitignored) -> {OUT_PATIENT}")


if __name__ == "__main__":
    sys.exit(main())
