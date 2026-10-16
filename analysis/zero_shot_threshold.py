"""Zero-shot metrics at an explicit 0.5 threshold on the continuous risk score, with bootstrap
CIs, for every model with a risk score.
"""
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score as sk_f1_score

sys.path.insert(0, "analysis")
from _patient_ids import AMBIGUOUS_IDS  # noqa: E402

PRED_GLOB = "saved_results/llm_zero_shot_prediction/amylo_predictions_*.csv"
BINARY_PROMPT_SUMMARY = "saved_results/llm_zero_shot_prediction/performance_summary.csv"
BINARY_PROMPT_DETAIL = "saved_results/llm_zero_shot_prediction/performance_detailed_by_class.csv"

OUT_MAIN = "results/zero_shot_threshold05.csv"
OUT_SWEEP = "results/zero_shot_threshold_sweep.csv"
OUT_INVERSION = "results/zero_shot_binary_vs_threshold.csv"

RNG = np.random.default_rng(82)
N_BOOT = 3000
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def stratified_bootstrap_indices(y_true, n_boot, rng):
    """Resample positives and negatives separately, preserving class balance
    per resample (stratified bootstrap), matching this repo's other CI code
    (paired_comparisons.py's paired_stratified_bootstrap_diff_ci)."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        yield np.concatenate([bp, bn])


def metrics_at_threshold(y_true, score, thr):
    y_pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = sk_f1_score(y_true, y_pred, zero_division=0)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=sens, specificity=spec,
                precision=prec, recall=sens, f1=f1)


def bootstrap_ci_at_threshold(y_true, score, thr, n_boot=N_BOOT, rng=RNG):
    keys = ["sensitivity", "specificity", "precision", "recall", "f1"]
    vals = {k: [] for k in keys}
    for idx in stratified_bootstrap_indices(y_true, n_boot, rng):
        m = metrics_at_threshold(y_true[idx], score[idx], thr)
        for k in keys:
            vals[k].append(m[k])
    out = {}
    for k in keys:
        arr = np.array(vals[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        lo, hi = (np.percentile(arr, 2.5), np.percentile(arr, 97.5)) if len(arr) else (np.nan, np.nan)
        out[f"{k}_ci_lo"] = lo
        out[f"{k}_ci_hi"] = hi
    return out


def load_binary_prompt():
    summ = pd.read_csv(BINARY_PROMPT_SUMMARY).set_index("model")
    detail = pd.read_csv(BINARY_PROMPT_DETAIL)
    spec = (detail[detail["class"].astype(str) == "0"]
            .set_index("model")["recall"]
            .rename("binary_prompt_specificity"))
    out = summ.join(spec, how="left")
    out = out.rename(columns={
        "precision": "binary_prompt_precision",
        "recall": "binary_prompt_sensitivity",
        "f1_score": "binary_prompt_f1",
    })[["binary_prompt_precision", "binary_prompt_sensitivity", "binary_prompt_specificity", "binary_prompt_f1"]]
    return out


def main():
    binary_prompt = load_binary_prompt()

    main_rows = []
    sweep_rows = []
    inversion_rows = []

    for f in sorted(glob.glob(PRED_GLOB)):
        model = f.split("amylo_predictions_")[1].removesuffix(".csv")
        df = pd.read_csv(f)
        df = df[~df["ID"].isin(AMBIGUOUS_IDS)]
        df = df.dropna(subset=["amylo_gold", "amylo_predicted_llm", "risk"])
        n_before_risk_filter = len(df)
        df = df[(df["risk"] >= 0) & (df["risk"] <= 1)]
        n_dropped_bad_risk = n_before_risk_filter - len(df)

        y_true = df["amylo_gold"].astype(int).to_numpy()
        score = df["risk"].to_numpy()
        stored = df["amylo_predicted_llm"].astype(int).to_numpy()
        n = len(df)

        # --- main: 0.5 threshold, full metrics + bootstrap CI, delta vs binary_prompt ---
        m05 = metrics_at_threshold(y_true, score, 0.5)
        ci05 = bootstrap_ci_at_threshold(y_true, score, 0.5)
        pub = binary_prompt.loc[model] if model in binary_prompt.index else pd.Series(
            {"binary_prompt_precision": np.nan, "binary_prompt_sensitivity": np.nan,
             "binary_prompt_specificity": np.nan, "binary_prompt_f1": np.nan})

        row = {
            "model": model, "n": n, "n_dropped_bad_risk": n_dropped_bad_risk,
            "n_ambiguous_excluded": len(AMBIGUOUS_IDS),
        }
        row.update({
            "tp": m05["tp"], "fp": m05["fp"], "fn": m05["fn"], "tn": m05["tn"],
            "sensitivity_05": m05["sensitivity"], "specificity_05": m05["specificity"],
            "precision_05": m05["precision"], "recall_05": m05["recall"], "f1_05": m05["f1"],
        })
        row.update(ci05)
        row.update(pub.to_dict())
        row["delta_precision"] = row["precision_05"] - row["binary_prompt_precision"]
        row["delta_sensitivity_recall"] = row["sensitivity_05"] - row["binary_prompt_sensitivity"]
        row["delta_specificity"] = row["specificity_05"] - row["binary_prompt_specificity"]
        row["delta_f1"] = row["f1_05"] - row["binary_prompt_f1"]
        main_rows.append(row)

        print(f"[{model}] n={n} (dropped {n_dropped_bad_risk} bad-risk, "
              f"{len(AMBIGUOUS_IDS)} ambiguous excluded upstream)  "
              f"0.5-thr: sens={m05['sensitivity']:.4f} spec={m05['specificity']:.4f} "
              f"prec={m05['precision']:.4f} f1={m05['f1']:.4f}  |  "
              f"binary_prompt: sens={pub['binary_prompt_sensitivity']:.4f} "
              f"spec={pub['binary_prompt_specificity']:.4f} "
              f"prec={pub['binary_prompt_precision']:.4f} f1={pub['binary_prompt_f1']:.4f}  |  "
              f"delta_f1={row['delta_f1']:+.4f}")

        # --- threshold sweep, F1 only ---
        for thr in THRESHOLDS:
            m = metrics_at_threshold(y_true, score, thr)
            sweep_rows.append({"model": model, "threshold": thr, "n": n, "f1": m["f1"],
                                "sensitivity": m["sensitivity"], "specificity": m["specificity"]})

        # --- inversion check: does the stored column disagree with 0.5-threshold
        # in DIRECTION (majority-positive vs majority-negative), not just in a
        # minority of rows? ---
        stored_pos_rate = stored.mean()
        thresh_pos_rate = (score >= 0.5).mean()
        # row-level mismatch count/direction
        mism = stored != (score >= 0.5).astype(int)
        n_mismatch = int(mism.sum())
        n_stored0_thresh1 = int(((stored == 0) & (score >= 0.5)).sum())
        n_stored1_thresh0 = int(((stored == 1) & (score < 0.5)).sum())
        stored_f1 = sk_f1_score(y_true, stored, zero_division=0)
        thresh_f1 = m05["f1"]
        stored_degenerate = stored_pos_rate in (0.0, 1.0)
        thresh_degenerate = thresh_pos_rate in (0.0, 1.0)
        direction_inverted = stored_degenerate and thresh_degenerate and (stored_pos_rate != thresh_pos_rate)
        inversion_rows.append({
            "model": model, "n": n,
            "stored_positive_rate": stored_pos_rate, "threshold05_positive_rate": thresh_pos_rate,
            "n_mismatch": n_mismatch, "pct_mismatch": 100 * n_mismatch / n,
            "n_stored0_thresh1": n_stored0_thresh1, "n_stored1_thresh0": n_stored1_thresh0,
            "stored_f1": stored_f1, "threshold05_f1": thresh_f1,
            "stored_degenerate_allpos_or_allneg": stored_degenerate,
            "threshold05_degenerate_allpos_or_allneg": thresh_degenerate,
            "direction_inverted": direction_inverted,
        })
        flag = "  <-- INVERTED (opposite degeneracy)" if direction_inverted else ""
        print(f"    [inversion-check] stored_pos_rate={stored_pos_rate:.3f} "
              f"thr05_pos_rate={thresh_pos_rate:.3f} mismatch={n_mismatch}/{n} "
              f"({100*n_mismatch/n:.1f}%){flag}")

    pd.DataFrame(main_rows).to_csv(OUT_MAIN, index=False)
    pd.DataFrame(sweep_rows).to_csv(OUT_SWEEP, index=False)
    pd.DataFrame(inversion_rows).to_csv(OUT_INVERSION, index=False)
    print(f"\n[saved] -> {OUT_MAIN}")
    print(f"[saved] -> {OUT_SWEEP}")
    print(f"[saved] -> {OUT_INVERSION}")


if __name__ == "__main__":
    sys.exit(main())
