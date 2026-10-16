#!/usr/bin/env python3
"""Rebuilds the gold standard for the 12 general variables from the annotation export, recovering
explicit negations, and rescores the model extractions against it.
"""
import collections
import os
import re

import pandas as pd

REPO = "."
FEATURES = ["HY", "DY", "DB", "CODP", "NYHA", "EF", "RS", "FA", "FL", "PM", "BBSx", "BBDx"]
FEATURES_9RF = ["TNC", "RTS", "SCM", "PDA", "PDG", "IPC", "DSM", "PAR", "IPGMM"]

GOLD_CONLL = os.path.join(
    REPO, "database/labeled/gold_standard_labelstudio_12cf",
    "project-18-at-2023-07-05-16-17-0f12dc92.json.conll")
SPACY_IOB_ID = os.path.join(
    REPO, "database/labeled/amylo_casi_controlli/old",
    "spacy_test_predict_dbmdz_output_labeled_by_spacy_ID.iob")

ORIG_GOLD_PATH = os.path.join(REPO, "database/labeled/amylo_garf_gold_labelstudio.csv")
MODEL_DIR = os.path.join(REPO, "database/labeled/amylo_casi_controlli")
GR_GOLD_XLSX = os.path.join(REPO, "saved_results/datasets_CA_GR/GR_12_9_gold.xlsx")

OUT_DIR = os.path.join(REPO, "work/_gold_rebuild")
RESULTS_DIR = os.path.join(REPO, "results")
NEW_GOLD_PATH = os.path.join(OUT_DIR, "amylo_garf_gold_labelstudio_rebuilt12cf.csv")
SPACY_3STATE_PATH = os.path.join(OUT_DIR, "spacy_12cf_3state.csv")

CONTROL_MODELS = ["gpt-4o", "gpt-4.1", "medgemma-27b-text-it"]
ALL_MODELS = [
    "BioMistral-7B", "DeepSeek-R1-Distill-Qwen-7B", "deepseek-r1-distill-qwen-7b",
    "Llama-3-8B-UltraMedical", "MMed-Llama-3-8B", "Meta-Llama-3-8B",
    "OpenBioLLM-Llama3-8B", "phi-4-mini-instruct", "phi-4-mini-reasoning",
    "Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct", "SauerkrautLM-gemma-2-9b-it-i1",
    "sauerkrautlm-gemma-2-9b-it-i1", "gpt-4.1", "gpt-4.1-mini", "gpt-4o",
    "gpt-4o_iter", "medicine-LLM", "medgemma-27b-text-it", "meditron-70b",
    "meditron-7b",
]

NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


# ---- three-state aggregation ----

def iter_entities(path):
    """Yield (doc_id, feature, is_negated, surface_tokens) from a -DOCSTART--delimited
    IOB file. doc_id is whatever token follows -DOCSTART- (str), never echoed."""
    doc, in_sent = None, False
    cur = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.rstrip("\n")
            st = s.strip()
            if st.startswith("-DOCSTART-"):
                if cur:
                    yield (doc, *cur)
                    cur = None
                parts = st.split()
                doc = parts[-1] if len(parts) > 1 else None
                in_sent = False
                continue
            if not st:
                if cur:
                    yield (doc, *cur)
                    cur = None
                in_sent = False
                continue
            parts = st.split()
            if len(parts) < 2:
                continue
            form, tag = parts[0], parts[-1]
            if tag.startswith("B-"):
                if cur:
                    yield (doc, *cur)
                t = tag[2:]
                neg = t.upper().startswith("NO") and t[2:] in FEATURES
                base = t[2:] if neg else t
                cur = [base, neg, [form]] if base in FEATURES else None
            elif tag.startswith("I-") and cur is not None:
                cur[2].append(form)
            else:
                if cur:
                    yield (doc, *cur)
                    cur = None
    if cur:
        yield (doc, *cur)


def read_docstart_ids(path):
    """All doc IDs seen at a -DOCSTART- marker, including documents with zero
    tagged spans -- without this, patients with no annotated feature at all
    silently vanish from the table instead of producing an all-NaN row
    (same fix rebuild_feature_tables.py applies via its `all_docs` parameter)."""
    ids = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            st = line.strip()
            if st.startswith("-DOCSTART-"):
                parts = st.split()
                if len(parts) > 1:
                    ids.append(parts[-1])
    return ids


def aggregate(path, ef_min=10.0, ef_max=80.0):
    """Return {doc_id: {feature: 1/0/None, 'EF_value': float or None}}."""
    pos = collections.defaultdict(lambda: collections.defaultdict(int))
    neg = collections.defaultdict(lambda: collections.defaultdict(int))
    ef_vals = collections.defaultdict(list)
    all_ids = read_docstart_ids(path)
    n_raw = len(all_ids)
    n_dupe = n_raw - len(set(all_ids))
    docs = set(all_ids)

    for doc, feat, negated, toks in iter_entities(path):
        if doc is None:
            continue
        docs.add(doc)
        (neg if negated else pos)[doc][feat] += 1
        if feat == "EF" and not negated:
            for raw in NUM_RE.findall(" ".join(toks)):
                try:
                    v = float(raw.replace(",", "."))
                except ValueError:
                    continue
                if ef_min <= v <= ef_max:
                    ef_vals[doc].append(v)

    table = {}
    for doc in sorted(docs):
        row = {}
        for f in FEATURES:
            p, n = pos[doc].get(f, 0), neg[doc].get(f, 0)
            if p:
                row[f] = 1
            elif n:
                row[f] = 0
            else:
                row[f] = None
        vals = ef_vals.get(doc, [])
        row["EF_value"] = round(sum(vals) / len(vals), 2) if vals else None
        table[doc] = row
    return table, n_raw, n_dupe


def id_key(x):
    """Normalize an ID to a bare-digit string for joining across float/int/str forms."""
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return None


def binarize(series):
    return series.apply(lambda x: 0 if x == 0 or pd.isna(x) else 1).astype(int)


def per_label_binary_counts(gold_df, model_df, common_ids, labels):
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


def f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def macro_micro(counts):
    f1s, tp_sum, fp_sum, fn_sum = [], 0, 0, 0
    for label, (tp, fp, fn, support) in counts.items():
        if support > 0:
            f1s.append(f1_from_counts(tp, fp, fn))
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
    macro = sum(f1s) / len(f1s) if f1s else float("nan")
    micro = (2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum)) if (2 * tp_sum + fp_sum + fn_sum) else float("nan")
    return macro, micro


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- 1. rebuild 12cf gold (CA cohort) from raw CoNLL, 3-state ----
    gold_table, gold_n_raw, gold_n_dupe = aggregate(GOLD_CONLL)
    print(f"[rebuild] gold CA-cohort: {gold_n_raw} -DOCSTART- lines, "
          f"{gold_n_dupe} duplicate-ID collisions, {len(gold_table)} distinct documents recovered")

    # ---- 2. rebuild spaCy's own 12cf predictions the same way ----
    spacy_table, spacy_n_raw, spacy_n_dupe = aggregate(SPACY_IOB_ID)
    print(f"[rebuild] spaCy CA-cohort: {spacy_n_raw} -DOCSTART- lines, "
          f"{spacy_n_dupe} duplicate-ID collisions, {len(spacy_table)} distinct documents recovered")

    # ---- 3. before/after counts per variable (full 430-row original gold) ----
    orig_gold = pd.read_csv(ORIG_GOLD_PATH)
    orig_gold["_key"] = orig_gold["ID"].apply(id_key)

    rebuilt_by_key = {id_key(k): v for k, v in gold_table.items() if id_key(k)}

    before_after_rows = []
    new_gold = orig_gold.copy()
    n_overwritten = 0
    for f in FEATURES:
        before_counts = collections.Counter()
        after_counts = collections.Counter()
        for idx, row in orig_gold.iterrows():
            key = row["_key"]
            before_val = row[f]
            before_bucket = "NaN" if pd.isna(before_val) else ("1" if before_val == 1 else ("0" if before_val == 0 else "other"))
            before_counts[before_bucket] += 1
            if key in rebuilt_by_key:
                new_val = rebuilt_by_key[key][f]
                after_bucket = "NaN" if new_val is None else str(new_val)
                new_gold.loc[idx, f] = new_val if new_val is not None else float("nan")
            else:
                after_bucket = before_bucket  # secondary-site record, not in the CoNLL export: keep original
            after_counts[after_bucket] += 1
        before_after_rows.append({
            "variable": f,
            "before_1": before_counts.get("1", 0), "before_0": before_counts.get("0", 0),
            "before_NaN": before_counts.get("NaN", 0),
            "after_1": after_counts.get("1", 0), "after_0": after_counts.get("0", 0),
            "after_NaN": after_counts.get("NaN", 0),
        })
    n_overwritten = sum(1 for k in orig_gold["_key"] if k in rebuilt_by_key)
    gr_xlsx = pd.read_excel(GR_GOLD_XLSX)
    gr_keys = set(gr_xlsx["ID"].apply(id_key).dropna())
    ca_gold_keys = set(orig_gold["_key"].dropna()) - gr_keys
    n_ca_not_in_conll = len(ca_gold_keys - set(rebuilt_by_key.keys()))
    print(f"[rebuild] rows overwritten with rebuilt 12cf values: {n_overwritten}/{len(orig_gold)} "
          f"(secondary-site records: {len(gr_keys)} rows kept as original, not sourced from this CoNLL; "
          f"{n_ca_not_in_conll}/{len(ca_gold_keys)} CA-cohort gold IDs have no matching document "
          f"in the raw CoNLL export at all -- kept as original, flagged as a coverage gap, not "
          f"chased further here)")

    new_gold = new_gold.drop(columns=["_key"])
    new_gold.to_csv(NEW_GOLD_PATH, index=False)
    print(f"[rebuild] wrote new gold file (original untouched) -> {NEW_GOLD_PATH}")

    ba_df = pd.DataFrame(before_after_rows)
    ba_out = os.path.join(RESULTS_DIR, "gold_12cf_rebuild_before_after_counts.csv")
    ba_df.to_csv(ba_out, index=False)
    print(f"[rebuild] before/after counts -> {ba_out}")
    print(ba_df.to_string(index=False))

    # write spaCy 3-state table (gitignored, patient-level)
    spacy_rows = []
    for doc, row in spacy_table.items():
        r = {"ID": doc}
        r.update({f: row[f] for f in FEATURES})
        spacy_rows.append(r)
    pd.DataFrame(spacy_rows).to_csv(SPACY_3STATE_PATH, index=False)
    print(f"[rebuild] spaCy 3-state table -> {SPACY_3STATE_PATH}")

    # spaCy negation capability, aggregate-only
    spacy_neg_counts = {f: sum(1 for r in spacy_table.values() if r[f] == 0) for f in FEATURES}
    print(f"[spacy-capability] explicit-negation (0) count per variable in spaCy's OWN "
          f"rebuilt predictions (proves spaCy CAN emit a negative span for that label "
          f"when the postprocessing bug is bypassed): {spacy_neg_counts}")

    # ---- 4. finding-13-style dropna agreement rate, new gold vs new spaCy-3-state and vs LLM controls ----
    new_gold_key = new_gold.copy()
    new_gold_key["ID"] = new_gold_key["ID"].apply(id_key)
    new_gold_key = new_gold_key.dropna(subset=["ID"]).drop_duplicates(subset="ID").set_index("ID")

    spacy_df = pd.DataFrame(spacy_rows)
    spacy_df["ID"] = spacy_df["ID"].apply(id_key)
    spacy_df = spacy_df.dropna(subset=["ID"]).drop_duplicates(subset="ID").set_index("ID")

    def agreement(gold_idx, other_idx, var):
        common = gold_idx.index.intersection(other_idx.index)
        sub = pd.DataFrame({"g": gold_idx.loc[common, var], "o": other_idx.loc[common, var]}).dropna()
        n = len(sub)
        if n == 0:
            return None, 0
        return (sub["g"] == sub["o"]).mean(), n

    control_dfs = {}
    for m in CONTROL_MODELS:
        df = pd.read_csv(os.path.join(MODEL_DIR, f"llm_annotations_{m}.csv"))
        df["ID"] = df["ID"].apply(id_key)
        df = df.dropna(subset=["ID"]).drop_duplicates(subset="ID").set_index("ID")
        control_dfs[m] = df

    agreement_rows = []
    for var in FEATURES:
        row = {"variable": var}
        rate, n = agreement(new_gold_key, spacy_df, var)
        row["spacy_rebuilt_agreement_rate"] = round(rate, 4) if rate is not None else None
        row["spacy_rebuilt_n_compared"] = n
        for m in CONTROL_MODELS:
            rate, n = agreement(new_gold_key, control_dfs[m], var)
            row[f"{m}_agreement_rate"] = round(rate, 4) if rate is not None else None
            row[f"{m}_n_compared"] = n
        agreement_rows.append(row)
    agr_df = pd.DataFrame(agreement_rows)
    agr_out = os.path.join(RESULTS_DIR, "gold_12cf_rebuilt_agreement.csv")
    agr_df.to_csv(agr_out, index=False)
    print(f"\n[agreement, rebuilt gold] -> {agr_out}")
    print(agr_df.to_string(index=False))

    # ---- 5. F1 delta: OLD gold vs NEW gold, binarize()-based, per label per model ----
    old_gold_bin = orig_gold.drop(columns=["_key"]).copy()
    old_gold_bin["ID"] = old_gold_bin["ID"].apply(id_key)
    old_gold_bin = old_gold_bin.dropna(subset=["ID"]).drop_duplicates(subset="ID")

    new_gold_bin = new_gold.copy()
    new_gold_bin["ID"] = new_gold_bin["ID"].apply(id_key)
    new_gold_bin = new_gold_bin.dropna(subset=["ID"]).drop_duplicates(subset="ID")

    f1_rows = []
    models_to_score = ALL_MODELS + ["spacy_rebuilt"]
    for model in sorted(set(models_to_score)):
        if model == "spacy_rebuilt":
            mdf = spacy_df.reset_index()
        else:
            p = os.path.join(MODEL_DIR, f"llm_annotations_{model}.csv")
            if not os.path.exists(p):
                continue
            mdf = pd.read_csv(p)
            mdf["ID"] = mdf["ID"].apply(id_key)
            mdf = mdf.dropna(subset=["ID"]).drop_duplicates(subset="ID")
        model_ids = set(mdf["ID"])
        common_old = set(old_gold_bin["ID"]) & model_ids
        common_new = set(new_gold_bin["ID"]) & model_ids
        counts_old, n_old = per_label_binary_counts(old_gold_bin, mdf, common_old, FEATURES)
        counts_new, n_new = per_label_binary_counts(new_gold_bin, mdf, common_new, FEATURES)
        if n_old == 0 and n_new == 0:
            continue
        macro_old, micro_old = macro_micro(counts_old)
        macro_new, micro_new = macro_micro(counts_new)
        for var in FEATURES:
            if var in counts_old and var in counts_new:
                f1_old = f1_from_counts(*counts_old[var][:3])
                f1_new = f1_from_counts(*counts_new[var][:3])
                f1_rows.append({
                    "model": model, "variable": var, "n_patients": n_new,
                    "f1_old_gold": round(f1_old, 4), "f1_new_gold": round(f1_new, 4),
                    "delta": round(f1_new - f1_old, 4),
                    "support_old": counts_old[var][3], "support_new": counts_new[var][3],
                })
        f1_rows.append({
            "model": model, "variable": "__macro__", "n_patients": n_new,
            "f1_old_gold": round(macro_old, 4), "f1_new_gold": round(macro_new, 4),
            "delta": round(macro_new - macro_old, 4), "support_old": "", "support_new": "",
        })
        f1_rows.append({
            "model": model, "variable": "__micro__", "n_patients": n_new,
            "f1_old_gold": round(micro_old, 4), "f1_new_gold": round(micro_new, 4),
            "delta": round(micro_new - micro_old, 4), "support_old": "", "support_new": "",
        })

    f1_df = pd.DataFrame(f1_rows)
    f1_out = os.path.join(RESULTS_DIR, "gold_12cf_rebuild_f1_delta.csv")
    f1_df.to_csv(f1_out, index=False)
    print(f"\n[F1 delta, old vs new gold] -> {f1_out}")
    max_abs_delta = f1_df["delta"].abs().max() if len(f1_df) else float("nan")
    print(f"[F1 delta] max |delta| across all models/variables/macro/micro: {max_abs_delta}")


if __name__ == "__main__":
    main()
