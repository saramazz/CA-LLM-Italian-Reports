"""Aggregate extraction F1 with and without ejection fraction and NYHA class.
"""
import sys

import pandas as pd

sys.path.insert(0, "analysis")
from extraction_metrics import (  # noqa: E402
    GOLD_PATH, MODELS, SPACY_FILE, load_model_file,
    per_label_binary_counts, macro_micro_from_counts,
)

OUT = "results/extraction_metrics_ef_nyha.csv"

# Values reported in the paper (two decimals), used as a consistency check. For the two
# locally deployable models run twice, the lower-case run (the one with a recorded
# parameter log) is the reported one; see qwen_duplication_comparison.csv.
REPORTED = {
    "spacy": {"micro": 0.79, "macro": 0.60},
    "gpt-4.1": {"micro": 0.76, "macro": 0.61},
    "gpt-4.1-mini": {"micro": 0.75, "macro": 0.60},
    "qwen2.5-7b-instruct": {"micro": 0.57, "macro": 0.37},
    "sauerkrautlm-gemma-2-9b-it-i1": {"micro": 0.66, "macro": 0.56},
}


def main():
    spacy = pd.read_csv(SPACY_FILE)
    spacy_ids = set(spacy["ID"].dropna().unique())
    gold = pd.read_csv(GOLD_PATH)
    all_labels = [c for c in gold.columns if c != "ID"]
    labels_excl = [l for l in all_labels if l not in ("EF", "NYHA")]
    print(f"[labels] included version: {len(all_labels)} labels; "
          f"excluded version: {len(labels_excl)} labels (EF, NYHA removed)")

    rows = []
    for model in MODELS:
        mdf = load_model_file(model)
        if mdf is None:
            continue
        counts_incl, n = per_label_binary_counts(gold, mdf, spacy_ids, all_labels)
        if n == 0:
            continue
        macro_incl, micro_incl = macro_micro_from_counts(counts_incl)

        counts_excl, _ = per_label_binary_counts(gold, mdf, spacy_ids, labels_excl)
        macro_excl, micro_excl = macro_micro_from_counts(counts_excl)

        pub = REPORTED.get(model)
        row = {
            "model": model, "n_patients": n,
            "macro_f1_EF_NYHA_included": round(macro_incl, 3),
            "micro_f1_EF_NYHA_included": round(micro_incl, 3),
            "macro_f1_EF_NYHA_excluded": round(macro_excl, 3),
            "micro_f1_EF_NYHA_excluded": round(micro_excl, 3),
            "reported_macro": pub["macro"] if pub else None,
            "reported_micro": pub["micro"] if pub else None,
        }
        if pub:
            row["macro_matches_included_2dp"] = round(macro_incl, 2) == pub["macro"]
            row["macro_matches_excluded_2dp"] = round(macro_excl, 2) == pub["macro"]
            row["micro_matches_included_2dp"] = round(micro_incl, 2) == pub["micro"]
            row["micro_matches_excluded_2dp"] = round(micro_excl, 2) == pub["micro"]
        rows.append(row)
        print(f"[{model}] n={n}  INCLUDED macro={macro_incl:.3f} micro={micro_incl:.3f}  "
              f"EXCLUDED macro={macro_excl:.3f} micro={micro_excl:.3f}" +
              (f"  reported macro={pub['macro']} micro={pub['micro']}" if pub else ""))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\n[saved] -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
