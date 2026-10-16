"""Extraction performance on the patients common to all systems: positive-class F1 per label,
macro-F1 and micro-F1.
"""
import sys

import pandas as pd

GOLD_PATH = "database/labeled/amylo_garf_gold_labelstudio.csv"
MODEL_DIR = "database/labeled/amylo_casi_controlli"
SPACY_FILE = f"{MODEL_DIR}/llm_annotations_spacy.csv"
TRAIN_IDS_PATH = "saved_results/classification_llm/train_ids.txt"
TEST_IDS_PATH = "saved_results/classification_llm/test_ids.txt"

OUT_COMMON = "results/common_cohort_macro_micro_f1.csv"
OUT_SUPPORT = "results/common_cohort_label_support.csv"
OUT_FULL = "results/full_cohort_macro_micro_f1.csv"
OUT_QWEN = "results/qwen_duplication_comparison.csv"

MODELS = [
    "BioMistral-7B", "DeepSeek-R1-Distill-Qwen-7B", "deepseek-r1-distill-qwen-7b",
    "Llama-3-8B-UltraMedical", "MMed-Llama-3-8B", "Meta-Llama-3-8B",
    "OpenBioLLM-Llama3-8B", "phi-4-mini-instruct", "phi-4-mini-reasoning",
    "Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct", "SauerkrautLM-gemma-2-9b-it-i1",
    "sauerkrautlm-gemma-2-9b-it-i1",
    "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o_iter", "medicine-LLM",
    "medgemma-27b-text-it", "meditron-70b", "meditron-7b", "spacy",
]


def binarize(series):
    return series.apply(lambda x: 0 if x == 0 or pd.isna(x) else 1).astype(int)


def load_model_file(model):
    path = f"{MODEL_DIR}/llm_annotations_{model}.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None
    if "ID" not in df.columns or len(df) == 0:
        return None
    return df


def cohort_check():
    spacy = pd.read_csv(SPACY_FILE)
    spacy_ids = set(spacy["ID"].dropna().unique())
    gold = pd.read_csv(GOLD_PATH)
    gold_ids = set(gold["ID"].dropna().unique())
    print(f"[cohort] spaCy file: {len(spacy_ids)} unique patients; "
          f"{len(spacy_ids - gold_ids)} not in the 430-patient gold cohort "
          f"(should be 0)")

    try:
        with open(TRAIN_IDS_PATH) as f:
            train_ids = set(line.strip() for line in f)
        with open(TEST_IDS_PATH) as f:
            test_ids = set(line.strip() for line in f)
        spacy_ids_str = set(str(x) for x in spacy_ids)
        n_train = len(spacy_ids_str & train_ids)
        n_test = len(spacy_ids_str & test_ids)
        print(f"[cohort] cached extraction/classification split: "
              f"{len(train_ids)} train / {len(test_ids)} test (total "
              f"{len(train_ids)+len(test_ids)})")
        print(f"[cohort] of spaCy's {len(spacy_ids)} patients: {n_train} fall in "
              f"the cached train partition, {n_test} in the cached test partition "
              f"-- spaCy is NOT confined to the test split, nor a clean subset of "
              f"it, under the currently cached split")
    except FileNotFoundError:
        print("[cohort] cached split files not found")
    return spacy_ids, gold_ids


def per_label_binary_counts(gold_df, model_df, common_ids, labels):
    """Returns dict label -> (tp, fp, fn, support) restricted to common_ids."""
    g = gold_df[gold_df["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)
    m = model_df[model_df["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)
    merged = pd.merge(g, m, on="ID", suffixes=("_gold", "_pred"))
    out = {}
    for label in labels:
        gcol, pcol = f"{label}_gold", f"{label}_pred"
        if gcol not in merged.columns or pcol not in merged.columns:
            continue
        y_true = binarize(merged[gcol])
        y_pred = binarize(merged[pcol])
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        support = int((y_true == 1).sum())
        out[label] = (tp, fp, fn, support)
    return out, len(merged)


def macro_micro_from_counts(counts):
    """Macro F1 excludes labels with zero gold-positive support in this cohort
    (RTS, on the 77-patient common cohort): positive-class F1 is undefined with
    no positive cases to recall, and scoring it as 0 for every model alike would
    silently deflate every model's macro F1 by the same amount without being able
    to distinguish a model that (correctly) never predicts it from one that does
    (both would otherwise score identically). Micro pooling is unaffected -- a
    false positive on a zero-support label still correctly penalizes precision
    in the pooled count."""
    f1s = []
    tp_sum = fp_sum = fn_sum = 0
    for label, (tp, fp, fn, support) in counts.items():
        if support > 0:
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            f1s.append(f1)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
    macro = sum(f1s) / len(f1s) if f1s else float("nan")
    micro = (2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum)
              if (2 * tp_sum + fp_sum + fn_sum) else float("nan"))
    return macro, micro


def main():
    spacy_ids, gold_ids = cohort_check()
    gold = pd.read_csv(GOLD_PATH)
    labels = [c for c in gold.columns if c != "ID"]
    print(f"\n[labels] n={len(labels)}: {labels}")

    # --- per-label support on the common (77-patient) cohort, once ---
    support_rows = []
    dummy_counts, n_common = per_label_binary_counts(gold, gold, spacy_ids, labels)
    # gold-vs-gold just to read off support cheaply (support doesn't depend on model)
    for label, (tp, fp, fn, support) in dummy_counts.items():
        support_rows.append({"label": label, "n_common_cohort": n_common, "n_positive_gold": support})
    pd.DataFrame(support_rows).to_csv(OUT_SUPPORT, index=False)
    print(f"[support] common cohort n={n_common}; saved -> {OUT_SUPPORT}")

    # --- common-cohort (77-patient) macro/micro per model ---
    common_rows = []
    full_rows = []
    for model in MODELS:
        mdf = load_model_file(model)
        if mdf is None:
            continue
        # common cohort (spaCy's 77)
        counts, n = per_label_binary_counts(gold, mdf, spacy_ids, labels)
        if n > 0:
            macro, micro = macro_micro_from_counts(counts)
            common_rows.append({"model": model, "n_patients": n, "macro_f1": macro, "micro_f1": micro})

        # full cohort: intersection of gold IDs and this model's own IDs
        model_ids = set(mdf["ID"].dropna().unique())
        full_common = gold_ids & model_ids
        counts_full, n_full = per_label_binary_counts(gold, mdf, full_common, labels)
        if n_full > 0:
            macro_f, micro_f = macro_micro_from_counts(counts_full)
            full_rows.append({"model": model, "n_patients": n_full, "macro_f1": macro_f, "micro_f1": micro_f})

    common_df = pd.DataFrame(common_rows).sort_values("micro_f1", ascending=False)
    full_df = pd.DataFrame(full_rows).sort_values("micro_f1", ascending=False)
    common_df.to_csv(OUT_COMMON, index=False)
    full_df.to_csv(OUT_FULL, index=False)

    print(f"\n[common cohort, n={len(spacy_ids)}] macro/micro F1 (positive class), saved -> {OUT_COMMON}")
    print(common_df.to_string(index=False))
    print(f"\n[full cohort, per-model own overlap with gold] macro/micro F1 (positive class), saved -> {OUT_FULL}")
    print(full_df.to_string(index=False))

    common_rank = list(common_df["model"])
    full_rank = list(full_df["model"])
    spacy_common_pos = common_rank.index("spacy") if "spacy" in common_rank else None
    spacy_full_pos = full_rank.index("spacy") if "spacy" in full_rank else None
    print(f"\n[ranking] spaCy rank by micro F1 -- common cohort: "
          f"{spacy_common_pos+1 if spacy_common_pos is not None else 'n/a'}/{len(common_rank)}; "
          f"full cohort (each model's own overlap with gold): "
          f"{spacy_full_pos+1 if spacy_full_pos is not None else 'n/a'}/{len(full_rank)}")

    # --- Qwen duplication ---
    qwen_rows = []
    for model in ["Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"]:
        mdf = load_model_file(model)
        if mdf is None:
            continue
        model_ids = set(mdf["ID"].dropna().unique())
        full_common = gold_ids & model_ids
        counts_full, n_full = per_label_binary_counts(gold, mdf, full_common, labels)
        macro_f, micro_f = macro_micro_from_counts(counts_full)
        counts_common, n_common_q = per_label_binary_counts(gold, mdf, spacy_ids, labels)
        macro_c, micro_c = macro_micro_from_counts(counts_common) if n_common_q else (float("nan"), float("nan"))
        qwen_rows.append({
            "file": f"llm_annotations_{model}.csv", "model_name_as_used": model,
            "n_rows": len(mdf), "n_overlap_with_gold": n_full,
            "full_cohort_macro_f1": macro_f, "full_cohort_micro_f1": micro_f,
            "common_cohort_n": n_common_q, "common_cohort_macro_f1": macro_c, "common_cohort_micro_f1": micro_c,
        })
    pd.DataFrame(qwen_rows).to_csv(OUT_QWEN, index=False)
    print(f"\n[Qwen duplication] saved -> {OUT_QWEN}")
    print(pd.DataFrame(qwen_rows).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
