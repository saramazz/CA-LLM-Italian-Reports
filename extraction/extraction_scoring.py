"""Per-label scoring of model extractions against the manual gold standard.
"""
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report


def binarize(df):
    """Ensure all non-zero values are treated as 1 (int), rest as 0 (int)."""
    id_column = "ID"
    for col in df.columns:
        if col != id_column:
            df[col] = df[col].apply(lambda x: 0 if x == 0 or pd.isna(x) else 1)
    df.fillna(0, inplace=True)
    return df


def calculate_metrics(preds, golds, labels):
    report = classification_report(
        golds, preds, labels=labels, zero_division=0, output_dict=True
    )
    acc = accuracy_score(golds, preds)
    return acc, report


def create_metrics_table(metrics_dict, model_name):
    rows = []
    for label, data in metrics_dict.items():
        acc = data["Accuracy"]
        report = data["Report"]
        for value in report:
            if value not in ["accuracy", "macro avg", "weighted avg"]:
                rows.append(
                    {
                        "Label": label,
                        "Value": value,
                        "Precision": f"{report[value]['precision']*100:.2f}%",
                        "Recall": f"{report[value]['recall']*100:.2f}%",
                        "F1-Score": f"{report[value]['f1-score']*100:.2f}%",
                        "Accuracy": f"{acc*100:.2f}%",
                        "Model": model_name,
                    }
                )
        rows.append(
            {
                "Label": label,
                "Value": "Global",
                "Precision": f"{report['weighted avg']['precision']*100:.2f}%",
                "Recall": f"{report['weighted avg']['recall']*100:.2f}%",
                "F1-Score": f"{report['weighted avg']['f1-score']*100:.2f}%",
                "Accuracy": f"{acc*100:.2f}%",
                "Model": model_name,
            }
        )
    df = pd.DataFrame(rows)
    df = df[["Model", "Label", "Value", "F1-Score", "Precision", "Recall", "Accuracy"]]
    out_path = os.path.join(saved_path, f"gold_vs_{model_name}_metrics.csv")
    print(f"Saving results for {model_name} to {out_path}")
    df.to_csv(out_path, index=False)
    print(f"Saved results for {model_name} to {out_path}")
    return df  # <-- return the DataFrame


def evaluate_per_label(merged_df, common_cols):
    """
    Calcola metriche per ciascuna colonna target in common_cols, usando le colonne _pred e _gold.

    Parameters:
        merged_df (pd.DataFrame): dataframe contenente colonne f"{label}_pred" e f"{label}_gold"
        common_cols (list): lista delle etichette da valutare

    Returns:
        dict: dizionario con metrica per ogni etichetta
    """
    metrics = {}
    for col in common_cols:
        pred_col = f"{col}_pred"
        gold_col = f"{col}_gold"
        y_pred = merged_df[pred_col]
        y_true = merged_df[gold_col]
        labels = sorted(list(set(y_true.unique())))
        acc, report = calculate_metrics(y_pred, y_true, labels)
        metrics[col] = {"Accuracy": acc, "Report": report}
    return metrics


def create_global_metrics_table(metrics_dict, comparison_name, output_csv):
    """
    Creates a global metrics table from the metrics dictionary, including both macro and micro averages,
    saves it as a CSV, and prints it.
    """
    import pandas as pd
    from sklearn.metrics import classification_report, accuracy_score

    rows = []
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_accuracy = 0.0
    num_labels = len(metrics_dict)

    y_true_all, y_pred_all = [], []

    for column, data in metrics_dict.items():
        accuracy = data["Accuracy"]
        precision = data["Report"]["weighted avg"]["precision"]
        recall = data["Report"]["weighted avg"]["recall"]
        f1_score = data["Report"]["weighted avg"]["f1-score"]

        total_precision += precision
        total_recall += recall
        total_f1 += f1_score
        total_accuracy += accuracy

        y_true_all.extend(data["y_true"])
        y_pred_all.extend(data["y_pred"])

        rows.append(
            {
                "Label": column,
                "F1-Score": f"{f1_score * 100:.2f}%",
                "Precision": f"{precision * 100:.2f}%",
                "Recall": f"{recall * 100:.2f}%",
                "Accuracy": f"{accuracy * 100:.2f}%",
            }
        )

    # Macro average
    macro_metrics = {
        "Label": "Overall (Macro)",
        "F1-Score": f"{(total_f1 / num_labels) * 100:.2f}%",
        "Precision": f"{(total_precision / num_labels) * 100:.2f}%",
        "Recall": f"{(total_recall / num_labels) * 100:.2f}%",
        "Accuracy": f"{(total_accuracy / num_labels) * 100:.2f}%",
    }
    rows.append(macro_metrics)

    # Micro average
    if y_true_all and y_pred_all:
        # Get all unique labels used
        all_labels = sorted(list(set(y_true_all + y_pred_all)))
        report = classification_report(
            y_true_all, y_pred_all, labels=all_labels, output_dict=True, zero_division=0
        )
        if "micro avg" in report:
            micro_metrics = {
                "Label": "Overall (Micro)",
                "F1-Score": f"{report['micro avg']['f1-score'] * 100:.2f}%",
                "Precision": f"{report['micro avg']['precision'] * 100:.2f}%",
                "Recall": f"{report['micro avg']['recall'] * 100:.2f}%",
                "Accuracy": f"{accuracy_score(y_true_all, y_pred_all) * 100:.2f}%",
            }
            rows.append(micro_metrics)
        else:
            print(
                "Warning: 'micro avg' not found in classification report. Skipping micro average."
            )
    else:
        print("Warning: Empty predictions/labels for micro average calculation.")

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"\n--- {comparison_name} Global Metrics ---")
    print(df.to_string(index=False))

models = [
    "BioMistral-7B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1-distill-qwen-7b",
    "Llama-3-8B-UltraMedical",
    "MMed-Llama-3-8B",
    "Meta-Llama-3-8B",
    "OpenBioLLM-Llama3-8B",
    "phi-4-mini-instruct",
    "phi-4-mini-reasoning",
    "Qwen2.5-7B-Instruct",
    "qwen2.5-7b-instruct",
    "SauerkrautLM-gemma-2-9b-it-i1",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o_iter",
    "medicine-LLM",
    "medgemma-27b-text-it",
    "meditron-70b",
    "meditron-7b",
    "sauerkrautlm-gemma-2-9b-it-i1",
    "spacy",
]



# Initialize list to store performance tables of all models
all_model_results = []


# Paths
saving_path = "database/labeled/amylo_casi_controlli/"

saved_path = ()

gold_standard_file_path = (
    "database/labeled/amylo_garf_gold_labelstudio.csv"
)
# gold_standard_file_path = (   "saved_results/datasets_CA_GR/gold_combined.csv") it is the same of amylo_garf_gold_labelstudio.csv
saved_path = "saved_results/ner_result"


if not os.path.exists(saved_path):
    os.makedirs(saved_path)
    print(f"Created directory {saved_path}")
else:
    print(f"Directory {saved_path} already exists")

# Load and prepare the gold standard
gold_df = pd.read_csv(gold_standard_file_path)
gold_df.rename(
    columns={
        "Hypertension ": "HY",
        "Dyslipidaemia": "DY",
        "Diabetic": "DB",
        "BPCO": "CODP",
        "Flutter": "FL",
    },
    inplace=True,
)
gold_df.fillna(0, inplace=True)


gold_df = binarize(gold_df)
id_column = "ID"
common_cols = [col for col in gold_df.columns if col != id_column]


# Evaluate each model
for model in models:
    print(f"\nEvaluating model: {model}")
    model_file_path = os.path.join(saving_path, f"llm_annotations_{model}.csv")

    if not os.path.exists(model_file_path):
        print(f"Skipping {model}: file not found.")
        continue

    model_df = pd.read_csv(model_file_path)
    model_df = binarize(model_df)

    # Keep only common IDs
    common_ids = set(gold_df["ID"]).intersection(set(model_df["ID"]))
    if not common_ids:
        print(f"Skipping {model}: no common IDs with gold standard.")
        continue

    filtered_gold = gold_df[gold_df["ID"].isin(common_ids)].sort_values(by="ID")
    model_df = model_df[model_df["ID"].isin(common_ids)].sort_values(by="ID")

    # check if there are missing values in the model_df
    if model_df.isnull().values.any():
        print(f"Warning: Missing values found in {model}. Filling with 0.")
        model_df.fillna(0, inplace=True)

    # Merge on ID
    merged = pd.merge(model_df, filtered_gold, on="ID", suffixes=("_pred", "_gold"))

    # Evaluate
    metrics = {}
    print(f"Evaluating {model} on {len(common_ids)} common IDs.")
    # print the common colums
    print(f"Common columns: {common_cols}")
    # metrics = evaluate_per_label(merged, common_cols)

    for col in common_cols:
        pred_col = f"{col}_pred"
        gold_col = f"{col}_gold"
        y_pred = merged[pred_col]
        y_true = merged[gold_col]
        labels = sorted(list(set(y_true.unique())))
        acc, report = calculate_metrics(y_pred, y_true, labels)
        metrics[col] = {
            "Accuracy": acc,
            "Report": report,
            "y_true": list(y_true),
            "y_pred": list(y_pred),
        }

    model_df_result = create_metrics_table(metrics, model)
    # print the results
    # print("Model results:")
    # print(model_df_result)
    # Append the model results to the list
    all_model_results.append(model_df_result)

    # Generate and save global table
    global_table_path = os.path.join(saved_path, f"gold_vs_{model}_global_summary.csv")
    create_global_metrics_table(metrics, f"Performance {model}", global_table_path)


if all_model_results:
    combined_rows = []
    for model_df_result, model_name in zip(all_model_results, models):
        # Append all per-label rows
        combined_rows.append(model_df_result)
        # mantain only the rows with Value = "Global"

        # Calculate overall weighted avg metrics for this model
        # Extract metrics dict again for the model to compute overall
        # We do this by reading the saved CSV or reconstructing from the model_df_result

        # Safer to reconstruct from model_df_result:
        # The 'Global' rows have Label == label and Value == 'Global'
        global_rows = model_df_result[model_df_result["Value"] == "Global"]

        if not global_rows.empty:
            overall_row = global_rows.iloc[0].copy()
            overall_row["Label"] = "Overall"
            overall_row["Value"] = "Global"
            for metric in ["F1-Score", "Precision", "Recall", "Accuracy"]:
                overall_row[metric] = (
                    f"{global_rows[metric].str.rstrip('%').astype(float).mean():.2f}%"
                )
            combined_rows.append(pd.DataFrame([overall_row]))

    # Concatenate all
    combined_df = pd.concat(combined_rows, ignore_index=True)
    # mantain only the rows with Value = "Global"
    combined_df = combined_df[combined_df["Value"].isin(["Global"])]
    # remove the column Value
    # combined_df = combined_df.drop(columns=["Value"])

    combined_out_path = os.path.join(
        saved_path, "all_models_combined_metrics_with_overall.csv"
    )
    combined_df.to_csv(combined_out_path, index=False)
    print(f"\nCombined results (with overall rows) saved to {combined_out_path}")
    print("Combined results for all models (with overall rows):")
    print(combined_df)
