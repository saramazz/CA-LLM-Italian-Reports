"""Metrics for all five strata, unmodified and masked, for one model: F1, sensitivity, specificity,
AUROC and AUPRC with bootstrap CIs and confusion counts.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, "analysis")
from baselines import AMBIGUOUS_IDS  # noqa: E402

REC = "analysis"
OUT = "results/masked_five_strata_complete.csv"
PRED_DIR = "saved_results/llm_zero_shot_prediction"
N_BOOT, SEED, THR = 2000, 82, 0.5


def point(y, r):
    p = (r >= THR).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "f1": f1_score(y, p, zero_division=0),
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "auroc": roc_auc_score(y, r) if both else np.nan,
            "auprc": average_precision_score(y, r) if both else np.nan}


def metrics(y, r):
    y, r = np.asarray(y, int), np.asarray(r, float)
    m = point(y, r)
    rng = np.random.default_rng(SEED)
    keys = ["f1", "sensitivity", "specificity", "auroc", "auprc"]
    boot = {k: [] for k in keys}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        b = point(y[i], r[i])
        for k in keys:
            boot[k].append(b[k])
    short = {"f1": "f1", "sensitivity": "sens", "specificity": "spec", "auroc": "auroc", "auprc": "auprc"}
    for k in keys:
        v = np.array(boot[k], float)
        v = v[~np.isnan(v)]
        m[f"{short[k]}_ci_lo"], m[f"{short[k]}_ci_hi"] = (np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan))
    m.update(n=len(y), n_ca=int(y.sum()), n_ctrl=int((y == 0).sum()))
    m["specificity_too_small_to_estimate"] = m["n_ctrl"] < 10
    return m


def build(model, raw_name, suffix):
    rows = []

    # pre-diagnostic (primary endpoint), raw + masked
    notes = pd.read_csv(f"{REC}/_primary_endpoint_notes.csv")[["CODINT", "gold"]]
    risk = pd.read_csv(f"{REC}/_primary_endpoint_risk_{model}.csv")
    wide = risk.pivot(index="CODINT", columns="corpus", values="risk").reset_index()
    d = wide.merge(notes, on="CODINT").dropna(subset=["raw", "masked"])
    for corpus in ["raw", "masked"]:
        rows.append({"stratum": "pre_diagnostic", "corpus": corpus, "text_source": "referto",
                     **metrics(d["gold"], d[corpus])})

    # raw main-sheet predictions (five_stratum_summary.py convention)
    raw = pd.read_csv(f"{PRED_DIR}/amylo_predictions_{raw_name}.csv")
    raw = raw[~raw["ID"].isin(AMBIGUOUS_IDS)]
    raw = raw[(raw["risk"] >= 0) & (raw["risk"] <= 1)].rename(columns={"amylo_gold": "gold"})
    raw_ctrl = raw[raw["gold"] == 0]
    bucket = pd.read_csv(f"{REC}/_ca_temporal_patient_level.csv").rename(columns={"CODINT": "ID"})[["ID", "bucket"]]
    raw_ca = raw[raw["gold"] == 1].merge(bucket, on="ID")
    explicit = pd.read_csv(f"{REC}/_explicit_flag_main_text.csv")[["ID", "gold", "explicit_term"]]
    raw_flag = raw.merge(explicit, on=["ID", "gold"])

    # masked fixed controls from this model's repro run
    repro = pd.read_csv(f"{REC}/_strata_repro_{suffix}.csv")
    m_ctrl = repro[repro["gold"] == 0]

    for label, bkt in [("same_day", "same_day"), ("post_diagnostic", "after")]:
        ca = raw_ca[raw_ca["bucket"] == bkt]
        y = np.r_[ca["gold"], raw_ctrl["gold"]]
        r = np.r_[ca["risk"], raw_ctrl["risk"]]
        rows.append({"stratum": label, "corpus": "raw", "text_source": "anamnesi", **metrics(y, r)})
        s = pd.read_csv(f"{REC}/_strata_stratum_{label}_{suffix}.csv")
        s = s[s["gold"] == 1]
        y = np.r_[s["gold"], m_ctrl["gold"]]
        r = np.r_[s["risk"], m_ctrl["risk"]]
        rows.append({"stratum": label, "corpus": "masked", "text_source": "referto", **metrics(y, r)})

    for label, flag in [("explicit_present", True), ("explicit_absent", False)]:
        sub = raw_flag[raw_flag["explicit_term"] == flag]
        rows.append({"stratum": label, "corpus": "raw", "text_source": "anamnesi",
                     **metrics(sub["gold"], sub["risk"])})
        s = pd.read_csv(f"{REC}/_strata_stratum_{label}_{suffix}.csv")
        rows.append({"stratum": label, "corpus": "masked", "text_source": "anamnesi",
                     **metrics(s["gold"], s["risk"])})

    out = pd.DataFrame(rows)
    out.insert(0, "model", model)
    return out


def build_referto_raw(model):
    """Phase 3: same-day / post-diagnostic RAW rows on unmasked REFERTO, so they pair with
    the existing masked (REFERTO) rows on the same text object. CA from the Phase 3 stratum
    files; controls = the 212 fixed controls' unmasked-REFERTO scores from the primary
    endpoint run (reused, not re-scored)."""
    notes = pd.read_csv(f"{REC}/_primary_endpoint_notes.csv")[["CODINT", "gold"]]
    risk = pd.read_csv(f"{REC}/_primary_endpoint_risk_{model}.csv")
    ctrl = risk[risk["corpus"] == "raw"].merge(notes, on="CODINT")
    ctrl = ctrl[(ctrl["gold"] == 0)].dropna(subset=["risk"])
    rows = []
    for label in ["same_day", "post_diagnostic"]:
        s = pd.read_csv(f"{REC}/_strata_stratum_{label}_raw_referto_{model}.csv")
        s = s[s["gold"] == 1]
        y = np.r_[s["gold"], ctrl["gold"]]
        r = np.r_[s["risk"], ctrl["risk"]]
        rows.append({"stratum": label, "corpus": "raw", "text_source": "referto", **metrics(y, r)})
    out = pd.DataFrame(rows)
    out.insert(0, "model", model)
    return out


def label_text_source(df):
    """Add text_source to rows written before the column existed (values unchanged)."""
    if "text_source" in df.columns:
        return df
    temporal = df["stratum"].isin(["same_day", "post_diagnostic"])
    src = np.where(df["stratum"] == "pre_diagnostic", "referto",
                   np.where(temporal & (df["corpus"] == "masked"), "referto", "anamnesi"))
    df.insert(df.columns.get_loc("corpus") + 1, "text_source", src)
    return df


def main():
    model, raw_name = sys.argv[1], sys.argv[2]
    existing = label_text_source(pd.read_csv(OUT))
    pd.set_option("display.width", 250)
    if "--phase3" in sys.argv:
        new = build_referto_raw(model)[existing.columns]
        dup = (existing["model"] == model) & (existing["corpus"] == "raw") & (existing["text_source"] == "referto") \
            & existing["stratum"].isin(["same_day", "post_diagnostic"])
        guard = f"{model} Phase 3 rows already present in {OUT}; not overwriting"
    else:
        suffix = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--suffix=")), model)
        new = build(model, raw_name, suffix)[existing.columns]
        dup = existing["model"] == model
        guard = f"{model} rows already present in {OUT}; not overwriting"
    print(new.drop(columns=["model"]).round(3).to_string(index=False))
    if "--write" in sys.argv:
        if dup.any():
            raise SystemExit(guard)
        pd.concat([existing, new], ignore_index=True).to_csv(OUT, index=False)
        print(f"[saved] appended {len(new)} rows -> {OUT}")


if __name__ == "__main__":
    main()
