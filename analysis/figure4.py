"""Figure 4 (per-label extraction F1 and feature importance) and the supplementary scatter figure
with Spearman correlations (Holm-adjusted within each set).
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import classification_harmonised as H  # noqa: E402
import extraction_metrics as C  # noqa: E402

REPO = "."
RES = os.path.join(REPO, "results")
OUT_DIR = os.path.join(RES, "figures")
OLD_BASE = os.path.join(REPO, "saved_results/classification_llm/train_test_split")
OLD_SPACY = os.path.join(REPO, "saved_results/classification_llm/loo/spacy")
OLD_CORR = os.path.join(REPO, "saved_results/paper_2_img/f1_fi_scatter_correlations.csv")
DPI, WIDTH_IN = 300, 7.0
CLF_DIR = {"LR": "LogisticRegression", "RF": "RandomForest", "XGB": "XGBClassifier"}
PANELS = [("spacy", "spaCy"), ("medgemma-27b-text-it", "MedGemma-27B"),
          ("sauerkrautlm-gemma-2-9b-it-i1", "SauerkrautLM-Gemma-2-9B")]
# extraction models shown in Figure 3a (lower-case run where a model was run twice)
EXTRACTION_MODELS = ["spacy", "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "medgemma-27b-text-it",
                     "sauerkrautlm-gemma-2-9b-it-i1", "deepseek-r1-distill-qwen-7b",
                     "qwen2.5-7b-instruct", "MMed-Llama-3-8B", "Llama-3-8B-UltraMedical",
                     "phi-4-mini-instruct", "OpenBioLLM-Llama3-8B", "meditron-7b",
                     "Meta-Llama-3-8B", "BioMistral-7B"]
ONLINE = {"gpt-4.1", "gpt-4.1-mini", "gpt-4o"}


def label_f1():
    os.chdir(REPO)
    spacy_ids, _ = C.cohort_check()
    gold = pd.read_csv(C.GOLD_PATH)
    labels = [c for c in gold.columns if c != "ID"]
    rows = []
    for m in EXTRACTION_MODELS:
        mdf = C.load_model_file(m)
        counts, n = C.per_label_binary_counts(gold, mdf, spacy_ids, labels)
        assert n == 77, (m, n)
        for lab, (tp, fp, fn, sup) in counts.items():
            f1 = 2 * tp / (2 * tp + fp + fn) if sup > 0 and (2 * tp + fp + fn) else (0.0 if sup > 0 else np.nan)
            rows.append({"model": m.lower(), "label": lab, "support": sup, "f1_positive": f1})
    return pd.DataFrame(rows)


def importances_original(model):
    old = pd.read_csv(OLD_CORR)
    old["key"] = old["Model"].str.lower()
    r = old[old["key"] == model]
    if r.empty:
        return None, None
    clf, name = r["Classifier"].iloc[0], r["Model"].iloc[0]
    path = os.path.join(OLD_SPACY if model == "spacy" else os.path.join(OLD_BASE, name), clf, "full", "model.joblib")
    est = joblib.load(path)
    src = (C.GOLD_PATH if model == "gold" else f"{C.MODEL_DIR}/llm_annotations_{name}.csv")
    feats = [c for c in pd.read_csv(os.path.join(REPO, src), nrows=0).columns if c not in ("ID", "amylo")]
    imp = np.abs(est.feature_importances_ if hasattr(est, "feature_importances_") else est.coef_[0])
    return dict(zip(feats, imp / imp.sum())), clf


def importances_harmonised(model, summary, files, cohort):
    s = summary[summary["source"] == model]
    if s.empty:
        return None, None
    clf_name = s.sort_values("f1", ascending=False)["classifier"].iloc[0]
    src = pd.read_csv(files[model]).rename(columns={"ID": "ID_src"})
    d = cohort.merge(src, left_on="ID", right_on="ID_src", how="inner")
    X = d[H.ALL_21].fillna(0).to_numpy(dtype=float)
    est = H.CLASSIFIERS[clf_name]().fit(X, d["gold"].to_numpy())
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    else:
        imp = np.abs(est.coef_[0] * X.std(axis=0))
    imp = np.abs(imp)
    if not np.isfinite(imp.sum()) or imp.sum() == 0:  # e.g. BioMistral: constant features
        return None, CLF_DIR[clf_name]
    return dict(zip(H.ALL_21, imp / imp.sum())), CLF_DIR[clf_name]


def rho(f1df, model, imp):
    d = f1df[(f1df["model"] == model) & (f1df["support"] > 0)]
    x = [imp.get(l, 0.0) for l in d["label"]]
    r, p = spearmanr(x, d["f1_positive"])
    return r, p, len(d)


def main():
    f1df = label_f1()
    os.chdir(REPO)
    summary = pd.read_csv(os.path.join(RES, "classification_harmonised_summary.csv"))
    files, cohort = H.pick_llm_files(), H.load_cohort()
    old = pd.read_csv(OLD_CORR)
    old["key"] = old["Model"].str.lower()

    imps, rows = {}, []
    for m in sorted(set(f1df["model"])):
        io, co = importances_original(m)
        ih, ch = importances_harmonised(m, summary, files, cohort)
        imps[m] = {"original": (io, co), "harmonised": (ih, ch)}
        rec = {"model": m}
        o = old[old["key"] == m]
        if not o.empty:
            rec.update(original_figure_classifier=o["Classifier"].iloc[0],
                       original_figure_rho=o["Spearman_corr"].iloc[0], original_figure_p=o["Spearman_p"].iloc[0])
        for tag, (imp, clf) in imps[m].items():
            if imp is not None:
                r, p, n = rho(f1df, m, imp)
                rec.update({f"{tag}_classifier": clf, f"{tag}_rho": r, f"{tag}_p": p, "n_labels": n})
        rows.append(rec)
    corr = pd.DataFrame(rows)
    # Rows with no correlation under any importance source (BioMistral: all-zero importances).
    corr = corr.dropna(subset=["harmonised_rho", "original_rho"], how="all").reset_index(drop=True)
    # Holm-Bonferroni within each set of correlations (one test per model in that set).
    for tag in ("harmonised", "original", "original_figure"):
        m = corr[f"{tag}_p"].notna()
        corr.loc[m, f"{tag}_p_holm"] = multipletests(corr.loc[m, f"{tag}_p"], method="holm")[1]
        corr[f"{tag}_n_tests"] = int(m.sum())
    corr = corr[["model", "n_labels",
                 "harmonised_classifier", "harmonised_rho", "harmonised_p", "harmonised_p_holm", "harmonised_n_tests",
                 "original_classifier", "original_rho", "original_p", "original_p_holm", "original_n_tests",
                 "original_figure_classifier", "original_figure_rho", "original_figure_p", "original_figure_p_holm", "original_figure_n_tests"]]
    corr.to_csv(os.path.join(RES, "figure4_spearman.csv"), index=False)
    f1df.to_csv(os.path.join(RES, "figure4_label_f1_positive_class.csv"), index=False)
    pd.set_option("display.width", 250)
    print(corr.round(3).to_string(index=False))

    # --- main figure: per-label overview only ---
    LABEL_NAMES = {
        "HY": "Hypertension", "DY": "Dyslipidaemia", "DB": "Diabetes", "CODP": "COPD",
        "NYHA": "NYHA class", "EF": "Ejection fraction", "RS": "Sinus rhythm",
        "FA": "Atrial fibrillation", "FL": "Atrial flutter", "PM": "Pacemaker",
        "BBSx": "LBBB", "BBDx": "RBBB", "TNC": "Carpal tunnel syndrome",
        "RTS": "Spontaneous tendon rupture", "SCM": "Spinal canal stenosis",
        "PDA": "Hip prosthesis", "PDG": "Knee prosthesis",
        "IPC": "Family history of cardiac hypertrophy", "DSM": "Sensory/motor disturbance",
        "PAR": "Paraesthesia", "IPGMM": "Monoclonal gammopathy"}
    MODEL_NAMES = {
        "spacy": "spaCy (supervised)", "gpt-4.1": "GPT-4.1", "gpt-4.1-mini": "GPT-4.1-mini",
        "gpt-4o": "GPT-4o", "medgemma-27b-text-it": "MedGemma-27B",
        "sauerkrautlm-gemma-2-9b-it-i1": "SauerkrautLM-Gemma-2-9B",
        "deepseek-r1-distill-qwen-7b": "DeepSeek-R1-Distill-Qwen-7B",
        "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct", "mmed-llama-3-8b": "MMed-Llama-3-8B",
        "llama-3-8b-ultramedical": "Llama-3-8B-UltraMedical", "phi-4-mini-instruct": "Phi-4-mini-instruct",
        "openbiollm-llama3-8b": "OpenBioLLM-Llama3-8B", "meditron-7b": "Meditron-7B",
        "meta-llama-3-8b": "Meta-Llama-3-8B", "biomistral-7b": "BioMistral-7B"}

    sup = f1df.drop_duplicates("label").set_index("label")["support"]
    labs = [l for l in sup.index if sup[l] > 0]
    avg = f1df[f1df["label"].isin(labs)].groupby("label")["f1_positive"].mean()
    labs = sorted(labs, key=lambda l: avg[l])
    yy = np.arange(len(labs))
    w_models = [m for m in imps if imps[m]["harmonised"][0] is not None]
    micro = pd.read_csv(os.path.join(RES, "common_cohort_macro_micro_f1.csv"))
    micro["key"] = micro["model"].str.lower()
    w = np.array([micro.loc[micro["key"] == m, "micro_f1"].max() for m in w_models])
    unweighted = {l: np.mean([imps[m]["harmonised"][0].get(l, 0) for m in w_models]) for l in labs}
    weighted = {l: np.average([imps[m]["harmonised"][0].get(l, 0) for m in w_models], weights=w) for l in labs}
    pd.DataFrame({"label": labs, "support": [int(sup[l]) for l in labs], "mean_f1": [avg[l] for l in labs],
                  "importance_unweighted": [unweighted[l] for l in labs],
                  "importance_weighted_by_micro_f1": [weighted[l] for l in labs]}
                 ).to_csv(os.path.join(RES, "figure4_label_overview.csv"), index=False)
    scale = 1 / max(max(unweighted.values()), max(weighted.values()))

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 7.4), constrained_layout=True)
    ax.barh(yy + 0.2, [weighted[l] * scale for l in labs], 0.38, color="salmon", alpha=0.55,
            label="Importance, weighted by micro-F1")
    ax.barh(yy - 0.2, [unweighted[l] * scale for l in labs], 0.38, color="lightsteelblue", alpha=0.75,
            label="Importance, unweighted mean")
    cmap = plt.get_cmap("tab20")
    for k, m in enumerate(EXTRACTION_MODELS):
        key = m.lower()
        d = f1df[f1df["model"] == key].set_index("label")
        marker = "s" if key == "spacy" else ("x" if key in ONLINE else "o")
        ax.scatter([d.loc[l, "f1_positive"] for l in labs], yy, s=14, marker=marker,
                   color="black" if key == "spacy" else cmap(k % 20), alpha=0.8, linewidths=0.9,
                   label=MODEL_NAMES[key], zorder=3)
    ax.scatter([avg[l] for l in labs], yy, s=40, color="red", edgecolor="black", lw=0.5, zorder=5,
               label="Mean extraction F1")
    ax.set_yticks(yy, [f"{LABEL_NAMES[l]} (n={int(sup[l])})" for l in labs], fontsize=8)
    ax.set_xlim(0, 1.02); ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Extraction F1, positive class (markers)\nFeature importance, rescaled to max 1 (bars)",
                  fontsize=8.5)
    ax.grid(axis="x", color="0.92", lw=0.5); ax.set_axisbelow(True)
    # Legend anchored to the figure, not the axes: the long row labels shift the axes to the
    # right, and an axes-centred legend ran past the right edge of the 7-inch figure.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, loc="outside lower center", ncol=3, frameon=False)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 0.975, 1.0))  # keep a right margin for the last tick
    out = os.path.join(OUT_DIR, "figure4")
    _save(fig, out)

    # --- supplementary: the three scatter panels with Spearman statistics ---
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH_IN, 3.3), constrained_layout=True, sharey=True)
    for k, ((m, disp), ax) in enumerate(zip(PANELS, axs)):
        tag = "original" if m == "spacy" else "harmonised"
        imp, clf = imps[m][tag]
        d = f1df[(f1df["model"] == m) & (f1df["support"] > 0)]
        x = np.array([imp.get(l, 0.0) for l in d["label"]]); y = d["f1_positive"].to_numpy()
        ax.scatter(x, y, s=14, color="purple")
        c = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, np.polyval(c, xs), "--", color="orange", linewidth=1)
        for xi, yi, lab in zip(x, y, d["label"]):
            ax.annotate(lab, (xi, yi), fontsize=5, alpha=0.85, xytext=(2, 1), textcoords="offset points")
        r, p, n = rho(f1df, m, imp)
        short = {"LogisticRegression": "logistic regression", "RandomForest": "random forest",
                 "XGBClassifier": "XGBoost"}[clf]
        ax.set_title(f"({'abc'[k]}) {disp}\n{short}{', original LOO pipeline' if tag == 'original' else ', harmonised'}\n"
                     f"Spearman ρ = {r:.2f}, p = {p:.2f} ({n} labels)", fontsize=6.5, loc="left")
        ax.set_xlabel("Normalised feature importance", fontsize=7)
        if k == 0:
            ax.set_ylabel("Extraction F1, positive class", fontsize=7)
        ax.set_ylim(-0.05, 1.05); ax.tick_params(labelsize=6.5); ax.grid(color="0.92", linewidth=0.5)
    _save(fig, os.path.join(OUT_DIR, "figure_s_feature_importance_scatter"))


def _save(fig, base):
    png, tif = f"{base}.png", f"{base}.tiff"
    fig.savefig(png, dpi=DPI)
    with Image.open(png) as im:
        im.convert("RGB").save(tif, format="TIFF", dpi=(DPI, DPI), compression="tiff_lzw")
        print(f"{os.path.basename(base)}: {im.size[0]}x{im.size[1]}px = {im.size[0]/DPI:.2f}x{im.size[1]/DPI:.2f} in")
    plt.close(fig)


if __name__ == "__main__":
    main()
