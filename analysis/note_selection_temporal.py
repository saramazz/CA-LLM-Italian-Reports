"""Canonical per-session note store and representative-note selection for the temporal arm.
"""
import sys

import numpy as np
import pandas as pd

from masking import mask_text, NEGATED_RE

CASI_XLSX = "database/unlabeled/amylo/Casi_new_noname.xlsx"
CONTROLLI_XLSX = "database/unlabeled/amylo/Controlli_new_noname.xlsx"
OUT_NOTE_LEVEL = "work/_temporal_note_selection.csv"  # gitignored
OUT_SUMMARY = "results/temporal_note_selection_summary.csv"

IT_MONTH = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12,
}
import re
DATE_RE = re.compile(r"^(\d{1,2})-([a-zA-Z]{3})-(\d{2})$")


def parse_it_date(x):
    if pd.isna(x):
        return pd.NaT
    m = DATE_RE.match(str(x).strip())
    if not m:
        return pd.NaT
    day, mon, yy = m.groups()
    month = IT_MONTH.get(mon.lower())
    if month is None:
        return pd.NaT
    year = 2000 + int(yy) if int(yy) <= 68 else 1900 + int(yy)
    try:
        return pd.Timestamp(year=year, month=month, day=int(day))
    except ValueError:
        return pd.NaT


def build_canonical_sessions():
    """Union of the two session sheets, de-duplicated per patient as described
    in this module's docstring. Returns columns: CODINT, DATAESAM (Timestamp), REFERTO."""
    ca_sheet = pd.read_excel(CASI_XLSX, sheet_name="anamnesi").dropna(subset=["CODINT", "DATAESAM"]).copy()
    ca_sheet["CODINT"] = ca_sheet["CODINT"].astype(int)
    ca_sheet["DATAESAM"] = ca_sheet["DATAESAM"].map(parse_it_date)
    ca_sheet = ca_sheet.dropna(subset=["DATAESAM"])
    ca_sheet = ca_sheet[["CODINT", "DATAESAM", "REFERTO"]]

    covered = set(ca_sheet["CODINT"])

    ctrl_sheet = pd.read_excel(CONTROLLI_XLSX, sheet_name="Anamnesi").dropna(subset=["codint", "DATAESAM"]).copy()
    ctrl_sheet["codint"] = ctrl_sheet["codint"].astype(int)
    ctrl_sheet = ctrl_sheet.rename(columns={"codint": "CODINT"})[["CODINT", "DATAESAM", "REFERTO"]]
    ctrl_supplement = ctrl_sheet[~ctrl_sheet["CODINT"].isin(covered)]

    canonical = pd.concat([ca_sheet, ctrl_supplement], ignore_index=True)
    return canonical


def load_labels():
    ca = pd.read_excel(CASI_XLSX, sheet_name="Casi")[["CODINT", "DataCaratterizzazione"]].dropna(subset=["CODINT"])
    ca["CODINT"] = ca["CODINT"].astype(int)
    ca["gold"] = 1
    ca["DataCaratterizzazione"] = pd.to_datetime(ca["DataCaratterizzazione"], errors="coerce")

    ctrl = pd.read_excel(CONTROLLI_XLSX, sheet_name="AMYLO Dati CONTROLLI")[["CODINT"]].dropna(subset=["CODINT"])
    ctrl["CODINT"] = ctrl["CODINT"].astype(int)
    ctrl["gold"] = 0
    ctrl["DataCaratterizzazione"] = pd.NaT

    labels = pd.concat([ca, ctrl], ignore_index=True).drop_duplicates(subset="CODINT", keep="first")
    return labels


def main():
    sessions = build_canonical_sessions()
    labels = load_labels()

    n_session_rows_raw = len(sessions)
    sessions = sessions.merge(labels, on="CODINT", how="inner")
    print(f"[coverage] canonical session rows: {n_session_rows_raw}, "
          f"after restricting to labelled patients: {len(sessions)}")
    print(f"[coverage] distinct patients with >=1 session: {sessions['CODINT'].nunique()} "
          f"(CA={sessions.loc[sessions['gold']==1,'CODINT'].nunique()}, "
          f"non-CA={sessions.loc[sessions['gold']==0,'CODINT'].nunique()})")

    ca_sessions = sessions[sessions["gold"] == 1].copy()
    ca_sessions["distance_days"] = (ca_sessions["DATAESAM"] - ca_sessions["DataCaratterizzazione"]).dt.days
    ca_sessions["abs_distance_days"] = ca_sessions["distance_days"].abs()
    ca_sessions["bucket"] = np.select(
        [ca_sessions["distance_days"] < 0, ca_sessions["distance_days"] == 0, ca_sessions["distance_days"] > 0],
        ["pre_diagnostic", "same_day", "post_diagnostic"],
    )

    # Closest note WITHIN each bucket, per patient (cost-bounding selection).
    idx = ca_sessions.groupby(["CODINT", "bucket"])["abs_distance_days"].idxmin()
    ca_selected = ca_sessions.loc[idx].copy()

    print("\n[selection] CA patients contributing a note to each temporal stratum:")
    print(ca_selected.groupby("bucket")["CODINT"].nunique().to_string())

    # Controls: one representative note per patient (median-dated session),
    # held fixed across all three temporal strata.
    ctrl_sessions = sessions[sessions["gold"] == 0].copy()
    ctrl_selected_rows = []
    for codint, grp in ctrl_sessions.groupby("CODINT"):
        grp_sorted = grp.sort_values("DATAESAM")
        median_row = grp_sorted.iloc[len(grp_sorted) // 2]
        ctrl_selected_rows.append(median_row)
    ctrl_selected = pd.DataFrame(ctrl_selected_rows)
    ctrl_selected["bucket"] = "control_fixed"
    print(f"\n[selection] control patients with a representative (median-dated) note: "
          f"{ctrl_selected['CODINT'].nunique()}/{ctrl_sessions['CODINT'].nunique()}")

    combined = pd.concat([
        ca_selected[["CODINT", "gold", "bucket", "REFERTO"]],
        ctrl_selected[["CODINT", "gold", "bucket", "REFERTO"]],
    ], ignore_index=True)

    masked_texts, family_counts_rows, negated_flags = [], [], []
    for _, row in combined.iterrows():
        result = mask_text(row["REFERTO"])
        masked_texts.append(result.masked_text)
        family_counts_rows.append(result.counts_by_family)
        negated_flags.append(bool(NEGATED_RE.search(str(row["REFERTO"]))))

    combined["masked_REFERTO"] = masked_texts
    combined["negated_present"] = negated_flags
    for family in ["explicit", "investigation", "negated"]:
        combined[f"n_{family}"] = [c[family] for c in family_counts_rows]

    combined[["CODINT", "gold", "bucket", "REFERTO", "masked_REFERTO",
              "negated_present", "n_explicit", "n_investigation", "n_negated"]].to_csv(
        OUT_NOTE_LEVEL, index=False
    )
    print(f"\n[saved] note-level temporal selection (gitignored) -> {OUT_NOTE_LEVEL}")

    rows = []
    for bucket in ["pre_diagnostic", "same_day", "post_diagnostic", "control_fixed"]:
        sub = combined[combined["bucket"] == bucket]
        if sub.empty:
            continue
        rows.append({
            "stratum": bucket,
            "n_patients": sub["CODINT"].nunique(),
            "n_notes": len(sub),
            "n_explicit_matches": sub["n_explicit"].sum(),
            "n_investigation_matches": sub["n_investigation"].sum(),
            "n_negated_matches": sub["n_negated"].sum(),
            "n_patients_with_negated_reference": int(sub["negated_present"].sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    print("\n=== TEMPORAL STRATUM SUMMARY (per-session REFERTO notes) ===")
    print(summary.to_string(index=False))
    print(f"\n[saved] aggregate-only summary -> {OUT_SUMMARY}")

    n_ctrl_negated = int(ctrl_selected_negated_count(combined))
    n_ctrl_total = int((combined["bucket"] == "control_fixed").sum())
    print(f"\n[finding] control notes (session-level, one per patient) with >=1 negated "
          f"diagnostic reference: {n_ctrl_negated}/{n_ctrl_total}")


def ctrl_selected_negated_count(combined):
    return combined.loc[combined["bucket"] == "control_fixed", "negated_present"].sum()


if __name__ == "__main__":
    sys.exit(main())
