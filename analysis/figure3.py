"""Figure 3: extraction performance on the common cohort and classification F1 under the
harmonised protocol, with zero-shot at the 0.5 threshold, and risk-score distributions.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_helpers import plot_risk_kde_ax, add_panel_label  # noqa: E402

REPO = "."
RES = os.path.join(REPO, "results")
OUT_DIR = os.path.join(RES, "figures")
DPI = 300
WIDTH_IN = 7.0

LABEL_PT, TICK_PT, VALUE_PT, LEGEND_PT = 11, 10, 9, 9
KDE_TITLE_PT, KDE_LABEL_PT, KDE_TICK_PT, KDE_LEGEND_PT = 11, 12, 11, 10

LOCAL_COLOR, API_COLOR, REF_COLOR, ZS_COLOR = "#4C72B0", "#8C8C8C", "#2F2F2F", "#DD8452"

DISPLAY = {
    "gold_21_rebuilt (reference)": "Gold standard (21 variables)",
    "gpt-4.1-mini": "GPT-4.1-mini", "gpt-4.1": "GPT-4.1", "gpt-4o": "GPT-4o",
    "gpt-4o_iter": "GPT-4o (iterative)",
    "medgemma-27b-text-it": "MedGemma-27B", "sauerkrautlm-gemma-2-9b-it-i1": "SauerkrautLM-Gemma-2-9B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct", "meta-llama-3-8b": "Meta-Llama-3-8B",
    "mmed-llama-3-8b": "MMed-Llama-3-8B", "deepseek-r1-distill-qwen-7b": "DeepSeek-R1-Distill-Qwen-7B",
    "phi-4-mini-instruct": "Phi-4-mini-instruct", "llama-3-8b-ultramedical": "Llama-3-8B-UltraMedical",
    "openbiollm-llama3-8b": "OpenBioLLM-Llama3-8B", "meditron-7b": "Meditron-7B",
    "biomistral-7b": "BioMistral-7B", "spacy": "spaCy (supervised)",
}
ZERO_SHOT_MODELS = ["medgemma-27b-text-it", "gpt-4o"]
PROPRIETARY = {"gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-4o_iter"}


def panel_b_table():
    h = pd.read_csv(os.path.join(RES, "classification_harmonised_summary.csv"))
    h = h[h["classifier"] == "LR"].copy()
    h["arm"] = np.where(h["source"].str.startswith("gold"), "reference",
                        np.where(h["source"].isin(PROPRIETARY), "proprietary", "local"))
    rows = h[["source", "arm", "f1", "f1_ci_lo", "f1_ci_hi", "n"]].copy()
    rows["protocol"] = "LR, 5-fold OOF, boot 5000"

    z = pd.read_csv(os.path.join(RES, "zero_shot_threshold05.csv")).set_index("model")
    zs = pd.DataFrame([{
        "source": f"zero-shot {m}", "arm": "zero-shot",
        "f1": z.loc[m, "f1_05"], "f1_ci_lo": z.loc[m, "f1_ci_lo"], "f1_ci_hi": z.loc[m, "f1_ci_hi"],
        "n": int(z.loc[m, "n"]), "protocol": "no training, threshold 0.5 on risk, boot 3000"}
        for m in ZERO_SHOT_MODELS])
    return pd.concat([rows, zs], ignore_index=True).sort_values("f1", ascending=False)


def panel_a_table():
    c = pd.read_csv(os.path.join(RES, "common_cohort_macro_micro_f1.csv"))
    c["key"] = c["model"].str.lower()
    # Where a model was run twice, two rows differ only in case; keep the
    # lower-case run, whose values Section 4.2 quotes (Qwen 0.57, Sauerkraut 0.66).
    dup = c["key"].duplicated(keep=False) & (c["model"] != c["key"])
    keep = ~dup & c["key"].isin(DISPLAY) & (c["key"] != "gpt-4o_iter") & (c["n_patients"] == 77)
    c = c[keep].copy()
    c["model"] = c["key"]
    c["arm"] = np.where(c["model"] == "spacy", "reference",
                        np.where(c["model"].isin(PROPRIETARY), "proprietary", "local"))
    return c.sort_values("micro_f1", ascending=False)


def check_against_manuscript(b):
    expected = {  # Section 4.3 / Abstract, rounded to 2 dp
        "gpt-4.1-mini": (0.82, 0.77, 0.86), "gold_21_rebuilt (reference)": (0.76, 0.72, 0.81),
        "sauerkrautlm-gemma-2-9b-it-i1": (0.79, None, None),
        "zero-shot medgemma-27b-text-it": (0.88, None, None),
        "zero-shot gpt-4o": (0.92, None, None),
    }
    ok = True
    for src, (f1, lo, hi) in expected.items():
        r = b[b["source"] == src].iloc[0]
        got = (round(r["f1"], 2), round(r["f1_ci_lo"], 2), round(r["f1_ci_hi"], 2))
        match = got[0] == f1 and (lo is None or (got[1], got[2]) == (lo, hi))
        ok &= match
        print(f"  {src:34s} manuscript {f1} [{lo}, {hi}]  data {got}  {'OK' if match else 'MISMATCH'}")
    return ok


def _color(arm):
    return {"reference": REF_COLOR, "proprietary": API_COLOR, "local": LOCAL_COLOR,
            "zero-shot": ZS_COLOR}[arm]


def draw_panel_a(ax, a):
    y = np.arange(len(a))
    ax.barh(y, a["micro_f1"], color=[_color(x) for x in a["arm"]], height=0.7)
    ax.scatter(a["macro_f1"], y, marker="D", s=14, color="white", edgecolor="black",
               linewidth=0.7, zorder=3, label="Macro-F1")
    for yi, v in zip(y, a["micro_f1"]):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=VALUE_PT)
    ax.set_yticks(y, [DISPLAY[m] for m in a["model"]], fontsize=TICK_PT)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Extraction F1, positive class", fontsize=LABEL_PT)
    ax.tick_params(axis="x", labelsize=TICK_PT)
    ax.grid(axis="x", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)


def draw_panel_b(ax, b):
    y = np.arange(len(b))
    err = np.vstack([b["f1"] - b["f1_ci_lo"], b["f1_ci_hi"] - b["f1"]])
    bars = ax.barh(y, b["f1"], xerr=err, color=[_color(x) for x in b["arm"]], height=0.7,
                   error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "black"})
    for bar, arm in zip(bars, b["arm"]):
        if arm == "zero-shot":
            bar.set_hatch("///")
            bar.set_edgecolor("black")
    for yi, v, hi in zip(y, b["f1"], b["f1_ci_hi"]):
        ax.text(hi + 0.012, yi, f"{v:.2f}", va="center", fontsize=VALUE_PT)
    labels = [(f"Zero-shot {DISPLAY[s.removeprefix('zero-shot ')]}" if s.startswith("zero-shot")
               else DISPLAY[s]) for s in b["source"]]
    ax.set_yticks(y, labels, fontsize=TICK_PT)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Classification F1 with 95% CI", fontsize=LABEL_PT)
    ax.tick_params(axis="x", labelsize=TICK_PT)
    ax.grid(axis="x", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)


def legend_handles(include_zs, include_macro):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(color=REF_COLOR, label="Reference"), Patch(color=API_COLOR, label="Proprietary LLM"),
         Patch(color=LOCAL_COLOR, label="Locally-deployable LLM")]
    if include_zs:
        h.append(Patch(facecolor=ZS_COLOR, hatch="///", edgecolor="black",
                       label="Zero-shot on free text, threshold 0.5"))
    if include_macro:
        h.append(Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
                        markeredgecolor="black", markersize=5, label="Macro-F1"))
    return h


def _legend_below(ax, handles):
    ax.legend(handles=handles, fontsize=LEGEND_PT, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.13), frameon=False, handlelength=1.4, columnspacing=1.0)


def save(fig, base):
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, f"{base}.png")
    tif = os.path.join(OUT_DIR, f"{base}.tiff")
    fig.savefig(png, dpi=DPI, format="png")
    with Image.open(png) as im:
        im.convert("RGB").save(tif, format="TIFF", dpi=(DPI, DPI), compression="tiff_lzw")
    for p in (png, tif):
        with Image.open(p) as im:
            w, h = im.size
            print(f"  {os.path.basename(p)}: {w}x{h}px, {w / DPI:.2f}x{h / DPI:.2f} in, "
                  f"dpi={im.info.get('dpi')}, compression={im.info.get('compression', 'n/a')}")
    return png, tif


def main():
    a, b = panel_a_table(), panel_b_table()
    print("Panel (b) values to plot:")
    print(b[["source", "arm", "f1", "f1_ci_lo", "f1_ci_hi", "n", "protocol"]]
          .round(3).to_string(index=False))
    print("\nCheck against Section 4.3:")
    if not check_against_manuscript(b):
        sys.exit("STOP: plotted values do not match the manuscript; no image produced.")
    print("\nPanel (a) values to plot:")
    print(a[["model", "arm", "micro_f1", "macro_f1", "n_patients"]].round(3).to_string(index=False))
    a.to_csv(os.path.join(RES, "figure3a_values.csv"), index=False)
    b.to_csv(os.path.join(RES, "figure3b_values.csv"), index=False)

    # Full composite: (a) and (b) full width, (c)/(d) side by side.
    fig, axd = plt.subplot_mosaic([["a", "a"], ["b", "b"], ["c", "d"]],
                                  figsize=(WIDTH_IN, 13.0), constrained_layout=True,
                                  gridspec_kw={"height_ratios": [1.0, 0.95, 0.62]})
    draw_panel_a(axd["a"], a)
    add_panel_label(axd["a"], "a")
    _legend_below(axd["a"], legend_handles(False, True))
    draw_panel_b(axd["b"], b)
    add_panel_label(axd["b"], "b")
    _legend_below(axd["b"], legend_handles(True, False))
    plot_risk_kde_ax(axd["c"], "gpt-4o", "c", KDE_TITLE_PT, KDE_LABEL_PT, KDE_TICK_PT, KDE_LEGEND_PT)
    plot_risk_kde_ax(axd["d"], "Qwen2.5-7B-Instruct", "d", KDE_TITLE_PT, KDE_LABEL_PT,
                     KDE_TICK_PT, KDE_LEGEND_PT)
    # One shared legend under (c)/(d): inside either panel it covered the density peaks.
    handles, labels = axd["c"].get_legend_handles_labels()
    for k in ("c", "d"):
        axd[k].get_legend().remove()
        axd[k].set_xlabel("Risk score", fontsize=KDE_LABEL_PT)
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=KDE_LEGEND_PT,
               frameon=False)
    print("\nComposite:")
    save(fig, "figure3")
    plt.close(fig)

    # Panel (b) alone, same typography, for use if the old (a) is kept.
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 4.6), constrained_layout=True)
    draw_panel_b(ax, b)
    add_panel_label(ax, "b")
    _legend_below(ax, legend_handles(True, False))
    print("\nPanel (b) alone:")
    save(fig, "figure3b_panel")
    plt.close(fig)
    print(f"\nSmallest font: {min(LABEL_PT, TICK_PT, VALUE_PT, LEGEND_PT, KDE_LEGEND_PT)} pt")


if __name__ == "__main__":
    main()
