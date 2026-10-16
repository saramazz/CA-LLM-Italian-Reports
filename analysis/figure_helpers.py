"""Plotting helpers shared by the figure scripts."""
import os

import pandas as pd
import seaborn as sns

REPO = "."


def add_panel_label(ax, label, fontsize=16):
    # loc="left" is a layout-managed title slot (distinct from the centered
    # title some panels also set), so constrained_layout reserves space for
    # it instead of it floating over tick labels/legends.
    ax.set_title(f"({label})", loc="left", fontsize=fontsize, fontweight="bold")


def plot_risk_kde_ax(ax, model, panel_label, title_pt, label_pt, tick_pt, legend_pt):
    """Mirrors figure3cd_risk_distribution_300dpi.py (same cached predictions)."""
    input_file = os.path.join(
        REPO, f"saved_results/llm_zero_shot_prediction/amylo_predictions_{model}.csv")
    results_df = pd.read_csv(input_file)
    plot_df = results_df.dropna(subset=["risk", "amylo_gold"])

    sns.kdeplot(data=plot_df[plot_df["amylo_gold"] == 0], x="risk", fill=True,
                common_norm=False, alpha=0.5, label="Class 0 (No Amyloidosis)",
                color="gray", ax=ax)
    sns.kdeplot(data=plot_df[plot_df["amylo_gold"] == 1], x="risk", fill=True,
                common_norm=False, alpha=0.5, label="Class 1 (Amyloidosis)",
                color="blue", ax=ax)
    # Panel letter and model name share one loc="left" title instead of two
    # separate title slots: at this column width (~3.4 in) a separate
    # centered title collided with the loc="left" panel-label title next to it.
    # (d)'s title also sits against the figure's own right edge (rightmost
    # column, no margin to spill into), so a long model name is wrapped
    # rather than shrunk further -- it was still clipping at 8 pt.
    title_text = f"({panel_label}) {model}"
    if len(title_text) > 16:
        head, _, tail = title_text.rpartition("-")
        title_text = f"{head}-\n{tail}"
    ax.set_title(title_text, loc="left", fontsize=title_pt, fontweight="bold")
    ax.set_xlabel("Predicted Risk Score", fontsize=label_pt)
    ax.set_ylabel("Density", fontsize=label_pt)
    ax.tick_params(labelsize=tick_pt)
    ax.legend(fontsize=legend_pt)
