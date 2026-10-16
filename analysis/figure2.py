"""Figure 2: clinical prevalence vs documentation frequency, documented-only variables,
ejection fraction and NYHA class, on the 204 CA / 213 non-CA cohort.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _patient_ids import AMBIGUOUS_IDS  # noqa: E402

REPO = "."
CASI_XLSX = os.path.join(REPO, "database/unlabeled/amylo/Casi_new_noname.xlsx")
CONTROLLI_XLSX = os.path.join(REPO, "database/unlabeled/amylo/Controlli_new_noname.xlsx")
GOLD = os.path.join(REPO, "database/labeled/amylo_garf_gold_labelstudio.csv")
RES = os.path.join(REPO, "results")
OUT_DIR = os.path.join(RES, "figures")
DPI, WIDTH_IN = 300, 7.0
CA_COLOR, NC_COLOR = "salmon", "skyblue"
LABEL_PT, TICK_PT, LEGEND_PT, PANEL_PT = 10, 9, 9, 14

GOLD_VARS = {
    "HY": "Hypertension", "DY": "Dyslipidaemia", "DB": "Diabetes", "CODP": "COPD",
    "RS": "Sinus rhythm", "FA": "Atrial fibrillation", "FL": "Atrial flutter",
    "PM": "Pacemaker", "BBSx": "LBBB", "BBDx": "RBBB",
    "TNC": "Carpal tunnel syndrome", "RTS": "Spontaneous tendon rupture",
    "SCM": "Spinal canal stenosis", "PDA": "Hip prosthesis", "PDG": "Knee prosthesis",
    "IPC": "Family history of cardiac hypertrophy", "DSM": "Sensory/motor disturbance",
    "PAR": "Paraesthesia", "IPGMM": "Monoclonal gammopathy",
}
# Table 1 source columns: (Casi column, Controlli column)
TABLE1_COLS = {
    "HY": ("Ipertensione", "Ipertensione"), "DY": ("Dislipidemia", "Dislipidemia"),
    "DB": ("Diabete", "Diabetico"), "CODP": ("BPCO", "BPCO"), "FA": ("FA", "FA"),
    "RS": ("RS", "RS"), "PM": ("PM", "PM"), "BBSx": ("BBSx", "BBSx"), "BBDx": ("BBDx", "BBDx"),
}


def load():
    casi = pd.read_excel(CASI_XLSX, sheet_name="Casi")
    ctrl = pd.read_excel(CONTROLLI_XLSX, sheet_name="AMYLO Dati CONTROLLI")
    ctrl = ctrl[~ctrl["CODINT"].isin(AMBIGUOUS_IDS)]
    assert (len(casi), len(ctrl)) == (204, 213), (len(casi), len(ctrl))
    gold = pd.read_csv(GOLD).drop_duplicates(subset="ID", keep="first")
    ca_ids, nc_ids = set(casi["CODINT"]), set(ctrl["CODINT"])
    gold = gold[gold["ID"].isin(ca_ids | nc_ids)].copy()
    gold["group"] = np.where(gold["ID"].isin(ca_ids), "CA", "non-CA")
    return casi, ctrl, gold


def panel_a(gold, casi, ctrl):
    rows = []
    for v, label in GOLD_VARS.items():
        rec = {"variable": v, "label": label}
        for grp in ("CA", "non-CA"):
            g = gold[gold["group"] == grp]
            k = int((g[v] == 1).sum())
            rec[f"n_{grp}"], rec[f"mentioned_{grp}"] = len(g), k
            rec[f"pct_mentioned_{grp}"] = 100 * k / len(g)
        if v in TABLE1_COLS:
            for grp, df, col in (("CA", casi, TABLE1_COLS[v][0]), ("non-CA", ctrl, TABLE1_COLS[v][1])):
                x = pd.to_numeric(df[col], errors="coerce").fillna(0)
                rec[f"clinical_{grp}"] = int((x > 0).sum())
                rec[f"pct_clinical_{grp}"] = 100 * (x > 0).mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def panel_b(casi, ctrl):
    ca = pd.to_numeric(casi["FE"], errors="coerce")
    nc = pd.to_numeric(ctrl["EF"], errors="coerce")
    out = {}
    for grp, s in (("CA", ca), ("non-CA", nc)):
        m = s[s > 0]
        out[grp] = {"n_rows": len(s), "n_zero_coded_missing": int((s == 0).sum()),
                    "n_null": int(s.isna().sum()), "n": len(m),
                    "mean": m.mean(), "sd": m.std(), "median": m.median(),
                    "q1": m.quantile(0.25), "q3": m.quantile(0.75),
                    "mean_incl_zeros": s.mean(), "sd_incl_zeros": s.std()}
    a, b = ca[ca > 0], nc[nc > 0]
    tests = {"t_test_p": stats.ttest_ind(a, b).pvalue,
             "welch_p": stats.ttest_ind(a, b, equal_var=False).pvalue,
             "mann_whitney_p": stats.mannwhitneyu(a, b).pvalue,
             "t_test_p_incl_zeros": stats.ttest_ind(ca.dropna(), nc.dropna()).pvalue}
    return a, b, pd.DataFrame(out).T, tests


def panel_c(casi, ctrl):
    rows = []
    for grp, df in (("CA", casi), ("non-CA", ctrl)):
        s = pd.to_numeric(df["NYHA"], errors="coerce")
        s = s[s > 0].replace(2.5, 2.0)
        for cls in (1, 2, 3, 4):
            k = int((s == cls).sum())
            rows.append({"group": grp, "nyha_class": cls, "n": k, "denominator": len(s),
                         "pct": 100 * k / len(s)})
    return pd.DataFrame(rows)


def draw(a_df, ef_ca, ef_nc, c_df):
    fig, axd = plt.subplot_mosaic([["a", "a"], ["b", "b"], ["c", "d"]], figsize=(WIDTH_IN, 9.6),
                                  constrained_layout=True,
                                  gridspec_kw={"height_ratios": [1.0, 1.0, 0.95]})

    # (a) Table 1 variables: clinical prevalence (filled) vs documented mention (open)
    ax = axd["a"]
    t1 = a_df[a_df["variable"].isin(TABLE1_COLS)].copy()
    t1 = t1.sort_values("pct_clinical_CA", ascending=True).reset_index(drop=True)
    for i, r in t1.iterrows():
        for off, grp, col in ((0.17, "CA", CA_COLOR), (-0.17, "non-CA", NC_COLOR)):
            yv = i + off
            ax.plot([r[f"pct_mentioned_{grp}"], r[f"pct_clinical_{grp}"]], [yv, yv], color=col, lw=1.2, zorder=1)
            ax.scatter(r[f"pct_clinical_{grp}"], yv, s=26, color=col, edgecolor="black", lw=0.5, zorder=3)
            ax.scatter(r[f"pct_mentioned_{grp}"], yv, s=26, facecolor="white", edgecolor=col, lw=1.4, zorder=3)
    ax.set_yticks(range(len(t1)), t1["label"], fontsize=TICK_PT)
    ax.set_xlim(-2, 100); ax.set_xlabel("Patients (%)", fontsize=LABEL_PT)
    ax.tick_params(axis="x", labelsize=TICK_PT); ax.grid(axis="x", color="0.92", lw=0.6); ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", markerfacecolor=CA_COLOR, markeredgecolor="black", label="CA, clinical (Table 1)"),
         Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=CA_COLOR, mew=1.4, label="CA, mentioned in narrative"),
         Line2D([], [], marker="o", ls="", markerfacecolor=NC_COLOR, markeredgecolor="black", label="non-CA, clinical (Table 1)"),
         Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=NC_COLOR, mew=1.4, label="non-CA, mentioned in narrative")]
    ax.legend(handles=h, fontsize=7.5, loc="lower right", ncol=1, frameon=False)
    ax.set_title("(a) Clinical prevalence vs documentation", loc="left", fontsize=LABEL_PT, fontweight="bold")

    # (b) variables with no clinical source: documented mention only
    ax = axd["b"]
    rf = a_df[~a_df["variable"].isin(TABLE1_COLS)].copy()
    rf = rf.sort_values("pct_mentioned_CA", ascending=True).reset_index(drop=True)
    y = np.arange(len(rf)); w = 0.38
    ax.barh(y + w / 2, rf["pct_mentioned_CA"], w, color=CA_COLOR, label="CA (n=204)")
    ax.barh(y - w / 2, rf["pct_mentioned_non-CA"], w, color=NC_COLOR, label="non-CA (n=211)")
    ax.set_yticks(y, rf["label"], fontsize=TICK_PT)
    ax.set_xlim(0, 100); ax.set_xlabel("Patients whose narrative mentions it (%)", fontsize=LABEL_PT)
    ax.tick_params(axis="x", labelsize=TICK_PT); ax.grid(axis="x", color="0.92", lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.set_title("(b) Documented only (no clinical source)", loc="left", fontsize=LABEL_PT, fontweight="bold")

    # (c) ejection fraction
    ax = axd["c"]
    bp = ax.boxplot([ef_ca, ef_nc], widths=0.5, patch_artist=True, showfliers=True,
                    medianprops={"color": "black"})
    for patch, col in zip(bp["boxes"], (CA_COLOR, NC_COLOR)):
        patch.set_facecolor(col)
    rng = np.random.default_rng(82)
    for i, s in enumerate((ef_ca, ef_nc), start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(s)), s, s=4, color="0.25", alpha=0.45, zorder=3)
    ax.set_xticks([1, 2], [f"CA\n(n={len(ef_ca)})", f"non-CA\n(n={len(ef_nc)})"], fontsize=TICK_PT)
    ax.set_ylabel("Ejection fraction (%)", fontsize=LABEL_PT); ax.tick_params(axis="y", labelsize=TICK_PT)
    ax.grid(axis="y", color="0.92", lw=0.6); ax.set_axisbelow(True)
    ax.set_title("(c) Ejection fraction", loc="left", fontsize=LABEL_PT, fontweight="bold")

    # (d) NYHA
    ax = axd["d"]
    for j, (grp, col) in enumerate((("CA", CA_COLOR), ("non-CA", NC_COLOR))):
        g = c_df[c_df["group"] == grp]
        ax.bar(g["nyha_class"] + (j - 0.5) * 0.4, g["pct"], 0.4, color=col,
               label=f"{grp} (n={int(g['denominator'].iloc[0])})")
    ax.set_xticks([1, 2, 3, 4], ["I", "II", "III", "IV"], fontsize=TICK_PT)
    ax.set_xlabel("NYHA class", fontsize=LABEL_PT); ax.set_ylabel("Patients (%)", fontsize=LABEL_PT)
    ax.set_ylim(0, 56); ax.tick_params(axis="y", labelsize=TICK_PT)
    ax.grid(axis="y", color="0.92", lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("(d) NYHA class", loc="left", fontsize=LABEL_PT, fontweight="bold")
    return fig


def save(fig, base):
    png, tif = (os.path.join(OUT_DIR, f"{base}.{e}") for e in ("png", "tiff"))
    fig.savefig(png, dpi=DPI)
    with Image.open(png) as im:
        im.convert("RGB").save(tif, format="TIFF", dpi=(DPI, DPI), compression="tiff_lzw")
        print(f"  {base}: {im.size[0]}x{im.size[1]}px = {im.size[0] / DPI:.2f}x{im.size[1] / DPI:.2f} in")


def main():
    casi, ctrl, gold = load()
    print(f"gold coverage: CA {int((gold.group == 'CA').sum())}, non-CA {int((gold.group == 'non-CA').sum())}")
    a_df = panel_a(gold, casi, ctrl)
    ef_ca, ef_nc, b_df, tests = panel_b(casi, ctrl)
    c_df = panel_c(casi, ctrl)
    pd.set_option("display.width", 250)
    print("\n(a)\n", a_df.round(1).to_string(index=False))
    print("\n(b)\n", b_df.round(2).to_string()); print({k: f"{v:.3g}" for k, v in tests.items()})
    print("\n(c)\n", c_df.round(1).to_string(index=False))
    a_df.to_csv(os.path.join(RES, "figure2_panel_a.csv"), index=False)
    b_df.assign(**{k: v for k, v in tests.items()}).to_csv(os.path.join(RES, "figure2_panel_b.csv"))
    c_df.to_csv(os.path.join(RES, "figure2_panel_c.csv"), index=False)
    fig = draw(a_df, ef_ca, ef_nc, c_df)
    save(fig, "figure2")
    plt.close(fig)


if __name__ == "__main__":
    main()
