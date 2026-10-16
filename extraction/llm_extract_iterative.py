"""Iterative extraction arm: one GPT-4o call per variable rather than one call per patient.
"""

import os
import pandas as pd
from openai import OpenAI
import numpy as np
import time

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

# Paths
input_path = "database/unlabeled/amylo_garf_anamnesis.csv"
output_path = "database/labeled/amylo_casi_controlli/llm_annotations_gpt-4o_iter.csv"
log_path = "database/labeled/amylo_casi_controlli/params_log_gpt-4o_iter.txt"

# Load full dataset
data = pd.read_csv(input_path)
print(f"Original data shape: {data.shape}")

# Remove already processed IDs
if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
    processed = pd.read_csv(output_path)
    processed_ids = processed["ID"].dropna().unique()
    data = data[~data["ID"].isin(processed_ids)]
    print(f"Resuming: {len(processed_ids)} entries already processed.")
else:
    processed = pd.DataFrame()
    processed_ids = set()
    print("No previous results found. Starting fresh.")

print(f"Remaining entries to process: {len(data)}")

# Prepare
model = "gpt-4o"
max_tokens = 25
sys_content = "Sei un assistente utile per identificare entità mediche e loro sinonimi nelle anamnesi italiane."

clinical_entities = {
    "HY": "Ipertensione",
    "DY": "Dislipidemia",
    "DB": "Diabete",
    "CODP": "Malattia Polmonare Cronica Ostruttiva",
    "NYHA": "Classificazione NYHA",
    "EF": "Frazione di Eiezione",
    "RS": "Ritmo Senosiduale",
    "FA": "Fibrillazione Atriale",
    "FL": "Flutter Atriale",
    "PM": "Pacemaker",
    "BBSx": "Blocco di Branca Sinistra",
    "BBDx": "Blocco di Branca Destra",
    "TNC": "Sindrome del Tunnel Carpale",
    "RTS": "Rottura Tendinea Spontanea",
    "SCM": "Stenosi del Canale Spinale",
    "PDA": "Protesi dell'Anca",
    "PDG": "Protesi del Ginocchio",
    "IPC": "Storia Personale o Familiare di Ipertrofia Cardiaca",
    "DSM": "Disturbi Sensoriali o Motori",
    "PAR": "Parestesie",
    "IPGMM": "Gammopatia Monoclonale",
}


def roman_to_int(s):
    roman_numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev_value = 0
    for char in reversed(s):
        value = roman_numerals.get(char, 0)
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    return total


start_time = time.time()
results_list = []
prompts_used = []

for index, row in data.iterrows():
    anamnesis = row["anamnesis"]
    result_row = {"ID": row["ID"]}

    print(f"\nAnalyzing patient ID {row['ID']}")

    for key, entity in clinical_entities.items():
        if key == "NYHA":
            prompt = f"Il paziente presenta dispnea? Se si rispondi con 1 o con la classificazione NYHA (numero romano) dalla seguente anamnesi se presente: {anamnesis}. Rispondi con 'NaN' se non presente."
        elif key == "EF":
            prompt = f"Estrai il valore della Frazione di Eiezione dalla seguente anamnesi se presente: {anamnesis}. Rispondi con il valore numerico o 'NaN' se non presente."
        else:
            prompt = f"Il paziente presenta '{entity}' nella seguente anamnesi? Rispondi con 1 se presente, 0 se il paziente non ha {entity} o NaN se non è menzionato: {anamnesis}"

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content.strip()

            if key == "NYHA":
                result_row[key] = (
                    roman_to_int(answer)
                    if answer in ["I", "II", "III", "IV"]
                    else np.nan
                )
            elif key == "EF":
                try:
                    result_row[key] = (
                        float(answer) if answer.lower() != "nan" else np.nan
                    )
                except ValueError:
                    result_row[key] = np.nan
            else:
                result_row[key] = 1 if answer == "1" else 0 if answer == "0" else np.nan

        except Exception as e:
            print(f"Error for patient {row['ID']} and entity {key}: {e}")
            result_row[key] = np.nan

    result_row["anamnesis"] = anamnesis
    results_list.append(result_row)

    pd.DataFrame([result_row]).to_csv(
        output_path,
        mode="a",
        index=False,
        header=not os.path.exists(output_path) or os.path.getsize(output_path) == 0,
    )
    print(f"Processed patient {row['ID']}")

# Log
execution_time = time.strftime("%Y%m%d_%H%M%S")
elapsed_time = time.time() - start_time
with open(log_path, "a") as f:
    f.write(f"Execution Time: {execution_time}\n")
    f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
    f.write(f"Model Used: {model}\n")
    f.write(f"Max Tokens: {max_tokens}\n")
    f.write(f"System Content: {sys_content}\n")

print("\nDone.")
print(f"New results appended to: {output_path}")
