"""Per-patient timing of the CA note relative to the diagnosis date (pre-diagnostic, same day,
post-diagnostic) and explicit-term flag.
"""
import re
import sys

import numpy as np
import pandas as pd

CASI_XLSX = "database/unlabeled/amylo/Casi_new_noname.xlsx"
OUT_AGGREGATE = "results/ca_temporal_stratification_summary.csv"
OUT_PATIENT_LEVEL = "work/_ca_temporal_patient_level.csv"  # gitignored

EXPLICIT_TERMS = [
    r"amiloidosi", r"amiloide", r"amyloid",
    r"\battr\b", r"transtiretina",
    r"\bal\b.{0,15}amiloid",
    r"catena leggera",
]
EXPLICIT_RE = re.compile("|".join(EXPLICIT_TERMS), flags=re.IGNORECASE)

# DATAESAM is stored as a string "DD-mon-YY" with an ITALIAN 3-letter month
# abbreviation, not English. pandas.to_datetime's default parser only recognises
# the 4 months whose Italian/English abbreviations happen to coincide (feb, mar,
# apr, nov) and silently turns the other 8 into NaT -- this was previously
# dropping 46/204 CA patients as "no resolvable note", which was a parsing bug,
# not missing data.
IT_MONTH = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12,
}
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


def main():
    casi = pd.ExcelFile(CASI_XLSX)
    patients = casi.parse("Casi")
    notes = casi.parse("anamnesi")

    print(f"[shape] Casi sheet: {patients.shape}")
    print(f"[shape] anamnesi sheet: {notes.shape}")
    print(f"[coverage] Casi.CODINT non-null: {patients['CODINT'].notna().sum()}/{len(patients)}")
    print(f"[coverage] Casi.DataCaratterizzazione non-null: {patients['DataCaratterizzazione'].notna().sum()}/{len(patients)}")
    print(f"[coverage] Casi.Diagnosi value counts (categories only):")
    print(patients["Diagnosi"].value_counts(dropna=False).to_string())

    patients = patients[["CODINT", "DataCaratterizzazione"]].dropna(subset=["CODINT"]).copy()
    patients["DataCaratterizzazione"] = pd.to_datetime(patients["DataCaratterizzazione"], errors="coerce")
    n_no_diag_date = patients["DataCaratterizzazione"].isna().sum()
    print(f"[coverage] patients with unparseable/missing diagnosis date: {n_no_diag_date}/{len(patients)}")

    notes = notes.dropna(subset=["CODINT", "DATAESAM"]).copy()
    n_raw_nonnull = len(notes)
    notes["DATAESAM"] = notes["DATAESAM"].map(parse_it_date)
    n_parse_failed = notes["DATAESAM"].isna().sum()
    print(f"[parsing] DATAESAM strings with a non-null raw value: {n_raw_nonnull}")
    print(f"[parsing] failed to parse as an Italian dd-mon-yy date: {n_parse_failed} ({100*n_parse_failed/n_raw_nonnull:.1f}%)")
    notes = notes.dropna(subset=["DATAESAM"])
    print(f"[coverage] note rows with valid CODINT+DATAESAM: {len(notes)}")
    print(f"[coverage] distinct CODINT with >=1 valid note: {notes['CODINT'].nunique()}")

    merged = notes.merge(patients, on="CODINT", how="inner")
    merged = merged.dropna(subset=["DataCaratterizzazione"])
    merged["distance_days"] = (merged["DATAESAM"] - merged["DataCaratterizzazione"]).dt.days
    merged["abs_distance_days"] = merged["distance_days"].abs()

    merged["explicit_term"] = merged["REFERTO"].fillna("").apply(lambda t: bool(EXPLICIT_RE.search(t)))

    idx_closest = merged.groupby("CODINT")["abs_distance_days"].idxmin()
    closest = merged.loc[idx_closest].copy()
    print(f"[result] patients with a resolvable closest note: {len(closest)}/{len(patients)}")

    closest["bucket"] = np.select(
        [closest["distance_days"] < 0, closest["distance_days"] == 0, closest["distance_days"] > 0],
        ["before", "same_day", "after"],
    )

    n = len(closest)
    pct_before = (closest["bucket"] == "before").mean() * 100
    pct_same = (closest["bucket"] == "same_day").mean() * 100
    pct_after = (closest["bucket"] == "after").mean() * 100
    median_dist = closest["abs_distance_days"].median()
    q1, q3 = closest["abs_distance_days"].quantile([0.25, 0.75])
    pct_explicit = closest["explicit_term"].mean() * 100

    print("\n=== AGGREGATE RESULT ===")
    print(f"n closest-notes resolved : {n}")
    print(f"% before diagnosis       : {pct_before:.1f}")
    print(f"% same day               : {pct_same:.1f}")
    print(f"% after diagnosis        : {pct_after:.1f}")
    print(f"median |distance| days   : {median_dist:.0f}")
    print(f"IQR |distance| days      : {q1:.0f}-{q3:.0f}")
    print(f"% explicit amyloid term  : {pct_explicit:.1f}")

    # Aggregate-only artifact: no CODINT, no dates, no text.
    pd.DataFrame([{
        "n": n,
        "pct_before": pct_before,
        "pct_same_day": pct_same,
        "pct_after": pct_after,
        "median_abs_distance_days": median_dist,
        "iqr_low": q1,
        "iqr_high": q3,
        "pct_explicit_term": pct_explicit,
        "n_patients_no_closest_note": len(patients) - n,
    }]).to_csv(OUT_AGGREGATE, index=False)
    print(f"\n[saved] aggregate-only summary -> {OUT_AGGREGATE}")

    # Patient-level artifact carries CODINT + dates + explicit-term flag (no REFERTO text).
    # Still gitignored/pre-commit-blocked by path, per the rules above.
    closest[["CODINT", "distance_days", "bucket", "explicit_term"]].to_csv(OUT_PATIENT_LEVEL, index=False)
    print(f"[saved] patient-level (gitignored) -> {OUT_PATIENT_LEVEL}")


if __name__ == "__main__":
    sys.exit(main())
