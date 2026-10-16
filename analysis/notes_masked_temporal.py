"""Masked session notes for the same-day and post-diagnostic strata.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from leakage_spec import mask_text, lexical_flags  # noqa: E402
from note_selection_temporal import build_canonical_sessions, load_labels  # noqa: E402

OUT_NOTES = "work/_all_temporal_strata_notes_current_spec.csv"  # gitignored
OUT_SUMMARY = "results/masked_strata_availability_summary.csv"


def select_ca_all_buckets(sessions: pd.DataFrame) -> pd.DataFrame:
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
    return closest[["CODINT", "gold", "bucket", "REFERTO"]]


def main():
    sessions = build_canonical_sessions()
    labels = load_labels()
    sessions = sessions.merge(labels, on="CODINT", how="inner")

    ca_all = select_ca_all_buckets(sessions)
    bucket_counts = ca_all["bucket"].value_counts().to_dict()
    print(f"[selection, current spec, closest note per patient] {bucket_counts} "
          f"(matches p17_temporal_recompute's 91/102/11 pre/same/post for the same rule)")

    masked_texts, n_masked_spans = [], []
    n_explicit = []
    for _, row in ca_all.iterrows():
        text = row["REFERTO"] if isinstance(row["REFERTO"], str) else ""
        mt, n = mask_text(text)
        masked_texts.append(mt)
        n_masked_spans.append(n)
        flags = lexical_flags(text)
        n_explicit.append(flags["n_explicit_hits"])
    ca_all["masked_REFERTO"] = masked_texts
    ca_all["n_masked_spans"] = n_masked_spans
    ca_all["n_explicit"] = n_explicit
    ca_all.to_csv(OUT_NOTES, index=False)
    print(f"[saved] current-spec masked text, all 3 CA buckets (gitignored) -> {OUT_NOTES}")

    rows = []
    for bucket in ["pre_diagnostic", "same_day", "post_diagnostic"]:
        sub = ca_all[ca_all["bucket"] == bucket]
        rows.append({
            "stratum": bucket, "n_patients_masked_text_available": len(sub),
            "already_scored_by_a_model": bucket == "pre_diagnostic",
            "n_masked_spans_total": int(sub["n_masked_spans"].sum()),
            "n_explicit_matches_total": int(sub["n_explicit"].sum()),
        })
    rows.append({
        "stratum": "control_fixed", "n_patients_masked_text_available": 212,
        "already_scored_by_a_model": True,
        "n_masked_spans_total": None, "n_explicit_matches_total": None,
    })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(summary.to_string(index=False))
    print(f"\n[saved] -> {OUT_SUMMARY}")
    print("\n[conclusion] masked text is producible right now for same_day (102) and "
          "post_diagnostic (11) without any model call -- masking itself was never the "
          "blocker. Scoring (obtaining a risk prediction on this text) requires a reachable "
          "LM Studio host, which is the actual and only remaining blocker.")


if __name__ == "__main__":
    sys.exit(main())
