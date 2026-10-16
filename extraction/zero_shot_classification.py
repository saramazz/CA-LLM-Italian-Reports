"""Zero-shot cardiac amyloidosis classification on the free-text anamnesis: a binary prompt and a
continuous risk-score prompt (the risk score, thresholded at 0.5, is the one analysed).
"""
import pandas as pd
from openai import OpenAI
import numpy as np
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)
start_time = time.time()

file_path = "database/unlabeled/amylo/casi_controlli_merged_gold.csv"
saving_llm_path = "saved_results/llm_zero_shot_prediction/"

models = [
    "DeepSeek-R1-Distill-Qwen-7B",
    "Llama-3-8B-UltraMedical",
    "Phi-4-mini-instruct",
    "Phi-4-mini-reasoning",
    "Qwen2.5-7B-Instruct",
    "medgemma-27b-text-it",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "SauerkrautLM-gemma-2-9b-it-i1",
    "MMed-Llama-3-8B"]

#models=["gpt-4.1-mini"]

max_tokens = 2000

data = pd.read_csv(file_path)
print(f"Loaded data with shape: {data.shape}")
# data = data.sample(n=2, random_state=42).reset_index(drop=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_prediction(answer):
    try:
        if isinstance(answer, int) and answer in [0, 1]:
            return answer
        cleaned = str(answer).strip().strip("```json").strip("```").strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, int) and parsed in [0, 1]:
                return parsed
            if isinstance(parsed, dict):
                pred = parsed.get("prediction")
                if pred in [0, 1]:
                    return pred
        except json.JSONDecodeError:
            pass
        match = re.search(r"\b[01]\b", cleaned)
        if match:
            return int(match.group())
        if "0" in cleaned:
            return 0
        if "1" in cleaned:
            return 1
    except Exception as e:
        logger.error(f"Unexpected format: {answer} — Error: {e}")
    return np.nan


def parse_risk(answer):
    try:
        cleaned = str(answer).strip().strip("```json").strip("```").strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, float) and 0 <= parsed <= 1:
                return parsed
            if isinstance(parsed, dict):
                risk = parsed.get("risk")
                if isinstance(risk, (float, int)) and 0 <= risk <= 1:
                    return float(risk)
        except json.JSONDecodeError:
            pass
        match = re.search(r"0(\.\d+)?|1(\.0+)?", cleaned)
        if match:
            val = float(match.group())
            if 0 <= val <= 1:
                return val
    except Exception as e:
        logger.error(f"Unexpected risk format: {answer} — Error: {e}")
    return np.nan


def predict_amyloidosis(anamnesis, model, max_tokens):
    messages = [
        {
            "role": "system",
            "content": "You are a medical assistant. Based on the Italian anamnesis text, determine if the patient has cardiac amyloidosis. Reply with 1 if present, or 0 if absent. Provide only the number, without any explanations or additional text.",
        },
        {
            "role": "user",
            "content": f"Here is the patient's anamnesis:\n{anamnesis}\nJust answer with 0 if the patient has no cardiac amyloidosis, 1 if it has. No explanation or other text.",
        },
    ]
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens
            )
            answer = response.choices[0].message.content.strip()
            pred = parse_prediction(answer)
            if not np.isnan(pred):
                return pred
        except Exception as e:
            logger.error(f"Prediction error attempt {attempt+1}: {e}")
    return np.nan


def predict_risk(anamnesis, model, max_tokens):
    messages = [
        {
            "role": "system",
            "content": "You are a medical assistant. Based on the Italian anamnesis text, estimate the risk (probability) that the patient has cardiac amyloidosis. Reply with a number between 0 and 1, no explanations or additional text.",
        },
        {
            "role": "user",
            "content": f"Here is the patient's anamnesis:\n{anamnesis}\nAnswer with a number between 0 and 1 representing the probability of cardiac amyloidosis. No explanations.",
        },
    ]
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens
            )
            answer = response.choices[0].message.content.strip()
            risk_val = parse_risk(answer)
            if not np.isnan(risk_val):
                return risk_val
        except Exception as e:
            logger.error(f"Risk prediction error attempt {attempt+1}: {e}")
    return np.nan


def analyze_row_binary(row, model, max_tokens):
    return row["ID"], predict_amyloidosis(row["anamnesis"], model, max_tokens)


def analyze_row_risk(row, model, max_tokens):
    return row["ID"], predict_risk(row["anamnesis"], model, max_tokens)


for model in models:
    output_file = os.path.join(saving_llm_path, f"amylo_predictions_{model}.csv")
    print(f"Processing model: {model}")

    if os.path.exists(output_file):
        results_df = pd.read_csv(output_file)
        logger.info(
            f"Loaded existing predictions for {model}, shape: {results_df.shape}"
        )
    else:
        # If it's MedGemma, we expect the file to already exist
        if "medgemma" in model.lower():
            logger.warning(
                f"Predictions for {model} expected to be precomputed, but file not found."
            )
            continue  # skip to next model
        results_df = pd.DataFrame(columns=["ID", "amylo_predicted_llm", "risk"])
        logger.info(f"Starting fresh for model {model}")

        # Skip new predictions if it's MedGemma, just use existing results
        if "medgemma" not in model.lower():
            if results_df.empty:
                remaining_data_binary = data
            else:
                predicted_ids = results_df["ID"].tolist()
                remaining_data_binary = data[~data["ID"].isin(predicted_ids)]

            if not remaining_data_binary.empty:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(analyze_row_binary, row, model, max_tokens)
                        for _, row in remaining_data_binary.iterrows()
                    ]
                    binary_results = []
                    for future in as_completed(futures):
                        try:
                            binary_results.append(future.result())
                        except Exception as e:
                            logger.error(f"Binary thread error: {e}")
                binary_df = pd.DataFrame(
                    binary_results, columns=["ID", "amylo_predicted_llm"]
                )

                if results_df.empty:
                    results_df = binary_df
                    results_df["risk"] = np.nan
                else:
                    results_df = (
                        pd.concat([results_df, binary_df])
                        .drop_duplicates(subset="ID")
                        .reset_index(drop=True)
                    )

            if "risk" not in results_df.columns:
                results_df["risk"] = np.nan

            missing_risk_ids = results_df[results_df["risk"].isnull()]["ID"].tolist()
            if missing_risk_ids:
                risk_data = data[data["ID"].isin(missing_risk_ids)]
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(analyze_row_risk, row, model, max_tokens)
                        for _, row in risk_data.iterrows()
                    ]
                    risk_results = []
                    for future in as_completed(futures):
                        try:
                            risk_results.append(future.result())
                        except Exception as e:
                            logger.error(f"Risk thread error: {e}")
                risk_df = pd.DataFrame(risk_results, columns=["ID", "risk"])

                results_df = results_df.drop_duplicates(subset="ID").copy()
                risk_df = risk_df.drop_duplicates(subset="ID").copy()

                results_df.set_index("ID", inplace=True)
                risk_df.set_index("ID", inplace=True)

                risk_df_aligned = risk_df.loc[
                    risk_df.index.intersection(results_df.index)
                ]
                results_df.update(risk_df_aligned)
                results_df.reset_index(inplace=True)

        if results_df.empty:
            results_df = results_df.merge(
                data[["ID", "amyloidosis"]], on="ID", how="left"
            )
            results_df.rename(columns={"amyloidosis": "amylo_gold"}, inplace=True)
        else:
            # Se carichi un file esistente, controlla che amylo_gold sia correttamente allineata
            if "amylo_gold" not in results_df.columns:
                results_df = results_df.merge(
                    data[["ID", "amyloidosis"]], on="ID", how="left"
                )
                results_df.rename(columns={"amyloidosis": "amylo_gold"}, inplace=True)

        results_df.to_csv(output_file, index=False)
        logger.info(f"Completed processing for {model}. Results saved to {output_file}")

        elapsed_time = time.time() - start_time
        # create log folder if it doesn't exist
        log_path = os.path.join(saving_llm_path, "logs")
        os.makedirs(log_path, exist_ok=True)

        params_log_path = os.path.join(log_path, f"params_log_{model}.txt")
        with open(params_log_path, "w") as f:
            f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
            f.write(f"Input File: {file_path}\n")
            f.write(f"Output File: {output_file}\n")
            f.write(f"Model Used: {model}\n")
            f.write(f"Max Tokens: {max_tokens}\n")

performance_records = []


# Function to convert nested list/Series to a single int (0 or 1)
def extract_single_val(x):
    if isinstance(x, (list, pd.Series, np.ndarray)):
        return int(x[0]) if len(x) > 0 else None
    return int(x)


# Convert nested values in the original data column to int
# data["amyloidosis"] = data["amyloidosis"].apply(extract_single_val)


def to_binary(x):
    if isinstance(x, (list, np.ndarray, pd.Series)):
        return int(any(x))
    return int(x)


# Evaluate performance for each model

logger.info("Starting performance evaluation for all models...")
# print the models
logger.info(f"Models to evaluate: {models}")


for model in models:
    print("Evaluating performance for model:", model)
    output_file = os.path.join(saving_llm_path, f"amylo_predictions_{model}.csv")

    if os.path.exists(output_file):
        results_df = pd.read_csv(output_file)
        logger.info(
            f"Loaded existing predictions for {model}, shape: {results_df.shape}"
        )
    else:
        logger.info(
            f"No predictions file found for {model}, skipping performance evaluation."
        )
        continue

    # Merge gold labels from original data (already cleaned)
    # results_df = results_df.merge(data[["ID", "amyloidosis"]], on="ID", how="left")
    # results_df.rename(columns={"amyloidosis": "amylo_gold"}, inplace=True)

    # --- New code to plot the distribution ---
    if "risk" in results_df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(results_df["risk"].dropna(), kde=True, bins=30, color="blue")
        plt.title(f"Distribution of Risk Percentage - Model {model}")
        plt.xlabel("Risk Percentage")
        plt.ylabel("Frequency")
        plt.grid(True)
        # save it

        # create a risk_distribution folder if it doesn't exist
        saving_llm_path_r = os.path.join(saving_llm_path, "risk_distribution")
        os.makedirs(saving_llm_path_r, exist_ok=True)
        plt.savefig(os.path.join(saving_llm_path_r, f"risk_distribution_{model}.svg"))
        print(f"Saved risk distribution plot for model {model} to {saving_llm_path_r}")
    else:
        logger.warning(
            f"The column 'risk_percentage' does not exist in the data for model {model}"
        )

    # Ensure amylo_gold column is int (single values, no nested lists)
    # results_df["amylo_gold"] = results_df["amylo_gold"].apply(extract_single_val)

    # Filter rows with valid gold and prediction labels
    perf_df = results_df.dropna(subset=["amylo_gold", "amylo_predicted_llm"])

    # Convert predicted label to binary int
    perf_df["amylo_predicted_llm_bin"] = perf_df["amylo_predicted_llm"].apply(to_binary)

    if perf_df.empty:
        logger.warning(f"No valid predictions to evaluate for model {model}")
        continue

    # remove duplicates columns
    perf_df = perf_df.loc[:, ~perf_df.columns.duplicated()]

    y_true = perf_df["amylo_gold"].astype(int)

    y_pred = perf_df["amylo_predicted_llm_bin"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    print(f"AUC for model {model}: {auc:.4f}")

    logger.info(
        f"Performance for model {model}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
    )

    performance_records.append(
        {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
        }
    )

# Save all performances to a single CSV file
performance_df = pd.DataFrame(performance_records)
performance_csv_path = os.path.join(saving_llm_path, "performance_summary.csv")
performance_df.to_csv(performance_csv_path, index=False)

logger.info("Performance summary for all models:")
logger.info(performance_df)

logger.info(f"Saved performance summary for all models to {performance_csv_path}")


# Optional: set a clean seaborn style
sns.set(style="whitegrid")

for model in models:
    output_file = os.path.join(saving_llm_path, f"amylo_predictions_{model}.csv")

    if os.path.exists(output_file):
        results_df = pd.read_csv(output_file)
        logger.info(
            f"Loaded existing predictions for {model}, shape: {results_df.shape}"
        )
    else:
        logger.info(
            f"No predictions file found for {model}, skipping performance evaluation."
        )
        continue

    # Merge gold labels from original data (already cleaned)
    # results_df = results_df.merge(data[["ID", "amyloidosis"]], on="ID", how="left")

    # Drop NA in risk or amylo_gold just to be safe
    plot_df = results_df.dropna(subset=["risk", "amylo_gold"])

    # Plot
    plt.figure(figsize=(10, 6))

    # KDE plot for class 0 (non-amyloidosis)
    sns.kdeplot(
        data=plot_df[plot_df["amylo_gold"] == 0],
        x="risk",
        fill=True,
        common_norm=False,
        alpha=0.5,
        label="Class 0 (No Amyloidosis)",
        color="gray",
    )

    # KDE plot for class 1 (amyloidosis)
    sns.kdeplot(
        data=plot_df[plot_df["amylo_gold"] == 1],
        x="risk",
        fill=True,
        common_norm=False,
        alpha=0.5,
        label="Class 1 (Amyloidosis)",
        color="blue",
    )

    # Add plot decorations
    plt.title("Risk Score Distribution by True Class")
    plt.xlabel("Predicted Risk Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    # create a risk_distribution_by_class folder if it doesn't exist
    risk_distribution_path = os.path.join(saving_llm_path, "risk_distribution_by_class")
    os.makedirs(risk_distribution_path, exist_ok=True)

    # Save the plot
    plt.savefig(
        os.path.join(risk_distribution_path, f"risk_distribution_by_class_{model}.svg")
    )
    plt.savefig(
        os.path.join(risk_distribution_path, f"risk_distribution_by_class_{model}.png"),
        dpi=300,
    )


# List to store overall model performance
performance_records = []

# List to store detailed performance per class
detailed_class_performance = []

for model in models:
    print("Evaluating performance for model:", model)
    output_file = os.path.join(saving_llm_path, f"amylo_predictions_{model}.csv")

    if os.path.exists(output_file):
        results_df = pd.read_csv(output_file)
        logger.info(
            f"Loaded existing predictions for {model}, shape: {results_df.shape}"
        )
    else:
        logger.info(
            f"No predictions file found for {model}, skipping performance evaluation."
        )
        continue

    # Plot risk score distribution if present
    if "risk" in results_df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(results_df["risk"].dropna(), kde=True, bins=30, color="blue")
        plt.title(f"Distribution of Risk Percentage - Model {model}")
        plt.xlabel("Risk Percentage")
        plt.ylabel("Frequency")
        plt.grid(True)

        risk_plot_dir = os.path.join(saving_llm_path, "risk_distribution")
        os.makedirs(risk_plot_dir, exist_ok=True)
        plt.savefig(os.path.join(risk_plot_dir, f"risk_distribution_{model}.svg"))
        print(f"Saved risk distribution plot for model {model} to {risk_plot_dir}")
    else:
        logger.warning(
            f"The column 'risk' does not exist in the data for model {model}"
        )

    # Keep only rows with valid gold and predicted labels
    perf_df = results_df.dropna(subset=["amylo_gold", "amylo_predicted_llm"])
    perf_df["amylo_predicted_llm_bin"] = perf_df["amylo_predicted_llm"].apply(to_binary)

    if perf_df.empty:
        logger.warning(f"No valid predictions to evaluate for model {model}")
        continue

    # Remove duplicate columns, if any
    perf_df = perf_df.loc[:, ~perf_df.columns.duplicated()]

    y_true = perf_df["amylo_gold"].astype(int)
    y_pred = perf_df["amylo_predicted_llm_bin"]

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    print(f"AUC for model {model}: {auc:.4f}")
    logger.info(
        f"Performance for model {model}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
    )

    # Save overall performance
    performance_records.append(
        {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
        }
    )

    # Save per-class performance (class 0, class 1, weighted avg)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for label in ["0", "1", "weighted avg"]:
        detailed_class_performance.append({
            "model": model,
            "class": label,
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1_score": report[label]["f1-score"],
            "support": report[label]["support"],
        })

# Save overall performance summary
performance_df = pd.DataFrame(performance_records)
performance_csv_path = os.path.join(saving_llm_path, "performance_summary_2.csv")
performance_df.to_csv(performance_csv_path, index=False)
logger.info("Saved overall model performance to:")
logger.info(performance_csv_path)

# Save detailed per-class metrics (all models into a single file)
detailed_df = pd.DataFrame(detailed_class_performance)
detailed_csv_path = os.path.join(saving_llm_path, "performance_detailed_by_class.csv")
detailed_df.to_csv(detailed_csv_path, index=False)
logger.info("Saved detailed per-class performance to:")
logger.info(detailed_csv_path)