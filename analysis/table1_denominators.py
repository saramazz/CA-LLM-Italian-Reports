"""Variable-specific denominators and missingness for Table 1.
"""
import sys

import pandas as pd

CASI_XLSX = "database/unlabeled/amylo/Casi_new_noname.xlsx"
CONTROLLI_XLSX = "database/unlabeled/amylo/Controlli_new_noname.xlsx"
RAW_PATH = "database/unlabeled/amylo/casi_controlli.csv"
OUT = "results/table1_denominators.csv"

BINARY_VARS = [("sex", "sex"), ("Ipertensione", "Ipertensione"), ("Dislipidemia", "Dislipidemia"),
               ("Diabete", "Diabetico"), ("BPCO", "BPCO"), ("FA", "FA"), ("RS", "RS"),
               ("PM", "PM"), ("BBSx", "BBSx"), ("BBDx", "BBDx")]
LABELS = {"sex": "Male sex", "Ipertensione": "Hypertension", "Dislipidemia": "Dyslipidaemia",
          "Diabete": "Diabetes", "BPCO": "COPD", "FA": "Atrial fibrillation",
          "RS": "Sinus rhythm", "PM": "Pacemaker", "BBSx": "LBBB", "BBDx": "RBBB"}


def main():
    casi = pd.read_excel(CASI_XLSX, sheet_name="Casi")
    ctrl_full = pd.read_excel(CONTROLLI_XLSX, sheet_name="AMYLO Dati CONTROLLI")
    from _patient_ids import AMBIGUOUS_IDS
    ctrl = ctrl_full[~ctrl_full["CODINT"].isin(AMBIGUOUS_IDS)].copy()

    n_ca_header, n_noca_header = 204, 213
    rows = []

    def add_row(var, ca_n, ca_missing, noca_n, noca_missing, note=""):
        header_mismatch = (ca_n != n_ca_header) or (noca_n != n_noca_header)
        rows.append({
            "variable": var,
            "header_n_CA": n_ca_header, "row_denominator_CA": ca_n, "n_missing_CA": ca_missing,
            "header_n_nonCA": n_noca_header, "row_denominator_nonCA": noca_n, "n_missing_nonCA": noca_missing,
            "denominator_differs_from_header": header_mismatch,
            "pct_basis": "complete_cases" if header_mismatch else "stated_header_n",
            "note": note,
        })

    # --- Age ---
    ca_age_n = casi["age"].notna().sum()
    nc_age_n = ctrl["age"].notna().sum()
    add_row("Age (years)", ca_age_n, len(casi) - ca_age_n, nc_age_n, len(ctrl) - nc_age_n)

    # --- EF (casi_controlli.csv, zeros included per prior confirmed reconciliation) ---
    raw = pd.read_csv(RAW_PATH)
    raw = raw[~raw["CODINT"].isin(AMBIGUOUS_IDS)].drop_duplicates(subset="CODINT", keep="first")
    casi_ids = set(casi["CODINT"])
    ctrl_ids = set(ctrl["CODINT"])
    ef_ca = raw[raw["CODINT"].isin(casi_ids)]["EF"]
    ef_nc = raw[raw["CODINT"].isin(ctrl_ids)]["EF"]
    add_row("Ejection fraction (%)", ef_ca.notna().sum(), len(casi_ids) - ef_ca.notna().sum(),
            ef_nc.notna().sum(), len(ctrl_ids) - ef_nc.notna().sum(),
            note="source: casi_controlli.csv registry EF column, not the gold-annotation placeholder column")

    # --- NT-proBNP ---
    ca_bnp_n = casi["NT-proBNP"].notna().sum()
    nc_bnp_n = ctrl["NT-proBNP"].notna().sum()
    add_row("NT-proBNP (pg/mL)", ca_bnp_n, len(casi) - ca_bnp_n, nc_bnp_n, len(ctrl) - nc_bnp_n)

    # --- NYHA (0 = not recorded, denominators 199/208) ---
    ca_nyha = casi["NYHA"]
    nc_nyha = ctrl["NYHA"]
    ca_nyha_n = int(((ca_nyha.notna()) & (ca_nyha != 0)).sum())
    nc_nyha_n = int(((nc_nyha.notna()) & (nc_nyha != 0)).sum())
    ca_nyha_missing = len(casi) - ca_nyha_n
    nc_nyha_missing = len(ctrl) - nc_nyha_n
    add_row("NYHA class", ca_nyha_n, ca_nyha_missing, nc_nyha_n, nc_nyha_missing,
            note="0-coded rows treated as 'class not recorded', denominators 199/208")
    for cls in [1, 2, 3, 4]:
        ca_cls = int((ca_nyha == cls).sum())
        nc_cls = int((nc_nyha == cls).sum())
        add_row(f"NYHA class {cls} (% of {ca_nyha_n}/{nc_nyha_n} recorded)",
                ca_nyha_n, ca_nyha_missing, nc_nyha_n, nc_nyha_missing,
                note=f"n={ca_cls}/{ca_nyha_n} CA, {nc_cls}/{nc_nyha_n} non-CA")

    # --- 9 binary variables + male sex ---
    for casi_col, ctrl_col in BINARY_VARS:
        ca_v = casi[casi_col]
        nc_v = ctrl[ctrl_col]
        ca_n = ca_v.notna().sum()
        nc_n = nc_v.notna().sum()
        note = f"casi column '{casi_col}' / controlli column '{ctrl_col}'" if casi_col != ctrl_col else ""
        add_row(LABELS[casi_col], ca_n, len(casi) - ca_n, nc_n, len(ctrl) - nc_n, note=note)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\n[saved] -> {OUT}")

    flagged = out[out["denominator_differs_from_header"]]
    print(f"\n[flagged: row denominator != header n] {len(flagged)} rows:")
    print(flagged[["variable", "row_denominator_CA", "row_denominator_nonCA"]].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
