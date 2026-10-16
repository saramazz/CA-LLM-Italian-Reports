"""Note set for the primary endpoint: strictly pre-diagnostic CA notes plus one fixed note per
control, unmodified and masked.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # for leakage_spec
from leakage_spec import (  # noqa: E402
    mask_text, lexical_flags, RE_SPECIFIC, RE_NEGATED,
)
from note_selection_temporal import build_canonical_sessions, load_labels  # noqa: E402

OUT_NOTE_LEVEL = "work/_primary_endpoint_notes.csv"  # gitignored
OUT_SUMMARY = "results/primary_endpoint_masking_summary.csv"


def select_ca_pre_diagnostic(sessions: pd.DataFrame) -> pd.DataFrame:
    ca = sessions[sessions["gold"] == 1].copy()
    ca["distance_days"] = (ca["DATAESAM"] - ca["DataCaratterizzazione"]).dt.days
    ca["abs_distance_days"] = ca["distance_days"].abs()
    ca = ca.dropna(subset=["distance_days"])

    idx_closest = ca.groupby("CODINT")["abs_distance_days"].idxmin()
    closest = ca.loc[idx_closest].copy()
    closest["bucket"] = np.select(
        [closest["distance_days"] < 0, closest["distance_days"] == 0, closest["distance_days"] > 0],
        ["pre_diagnostic", "same_day", "post_diagnostic"],
    )
    return closest[closest["bucket"] == "pre_diagnostic"][["CODINT", "gold", "bucket", "REFERTO"]]


def select_controls(sessions: pd.DataFrame) -> pd.DataFrame:
    ctrl = sessions[sessions["gold"] == 0].copy()
    rows = []
    for codint, grp in ctrl.groupby("CODINT"):
        grp_sorted = grp.sort_values("DATAESAM")
        rows.append(grp_sorted.iloc[len(grp_sorted) // 2])
    out = pd.DataFrame(rows)
    out["bucket"] = "control_fixed"
    return out[["CODINT", "gold", "bucket", "REFERTO"]]


def main():
    sessions = build_canonical_sessions()
    labels = load_labels()
    sessions = sessions.merge(labels, on="CODINT", how="inner")

    ca_pre = select_ca_pre_diagnostic(sessions)
    ctrl = select_controls(sessions)
    combined = pd.concat([ca_pre, ctrl], ignore_index=True)

    print(f"[selection] CA pre-diagnostic patients: {ca_pre['CODINT'].nunique()} "
          f"(expected 91, per p17_temporal_recompute_summary.csv)")
    print(f"[selection] control patients with a representative note: "
          f"{ctrl['CODINT'].nunique()} (canonical non-CA cohort is 213; "
          f"gap, if any, is a session-coverage limit, not a filter applied here)")

    masked_texts, n_masked_spans = [], []
    n_explicit, n_specific, n_negated, has_negated = [], [], [], []
    for _, row in combined.iterrows():
        text = row["REFERTO"] if isinstance(row["REFERTO"], str) else ""
        mt, n = mask_text(text)
        masked_texts.append(mt)
        n_masked_spans.append(n)
        flags = lexical_flags(text)
        n_explicit.append(flags["n_explicit_hits"])
        n_specific.append(len(RE_SPECIFIC.findall(text)))
        neg_hits = len(RE_NEGATED.findall(text))
        n_negated.append(neg_hits)
        has_negated.append(neg_hits > 0)

    combined["masked_REFERTO"] = masked_texts
    combined["n_masked_spans"] = n_masked_spans
    combined["n_explicit"] = n_explicit
    combined["n_disease_specific"] = n_specific
    combined["n_negated"] = n_negated
    combined["has_negated"] = has_negated

    combined.to_csv(OUT_NOTE_LEVEL, index=False)
    print(f"\n[saved] patient-level primary-endpoint notes (gitignored) -> {OUT_NOTE_LEVEL}")

    rows = []
    for bucket in ["pre_diagnostic", "control_fixed"]:
        sub = combined[combined["bucket"] == bucket]
        rows.append({
            "stratum": bucket,
            "n_patients": sub["CODINT"].nunique(),
            "n_explicit_matches": int(sub["n_explicit"].sum()),
            "n_disease_specific_matches": int(sub["n_disease_specific"].sum()),
            "n_negated_matches": int(sub["n_negated"].sum()),
            "n_patients_with_negated_reference": int(sub["has_negated"].sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    print("\n=== PRIMARY ENDPOINT MASKING SUMMARY (spec patterns, leakage_spec.py) ===")
    print(summary.to_string(index=False))
    print(f"\n[saved] aggregate-only summary -> {OUT_SUMMARY}")

    n_ctrl = int((combined["bucket"] == "control_fixed").sum())
    n_ctrl_neg = int(combined.loc[combined["bucket"] == "control_fixed", "has_negated"].sum())
    print(f"\n[finding] control notes (session-level, one per patient) with >=1 negated "
          f"diagnostic reference under spec patterns: {n_ctrl_neg}/{n_ctrl}")


if __name__ == "__main__":
    sys.exit(main())
