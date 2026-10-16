"""Extraction of the 21 clinical variables with locally served open models (LM Studio,
OpenAI-compatible endpoint). Same prompt and output format as llm_extract_proprietary.py.
"""
import requests
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import logging
from threading import Lock
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from datetime import datetime


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


# --------------------
# Extract answers from multi-entity JSON-style response or fallback to pattern matching


def extract_all_answers(response_text):

    parsed = {k: np.nan for k in clinical_entities}

    try:
        # Try to find and parse a JSON block
        match = re.search(r"\{[\s\S]*?\}", response_text)
        if match:
            json_str = match.group(0)
            json_str = re.sub(r"//.*", "", json_str)  # Remove JS-style comments
            result = json.loads(json_str)

            for key in clinical_entities:
                val = result.get(key, "NaN")
                if key == "NYHA" and isinstance(val, str):
                    parsed[key] = roman_to_int(val.strip().upper())
                elif key == "EF":
                    try:
                        parsed[key] = float(val)
                    except:
                        parsed[key] = np.nan
                else:
                    parsed[key] = int(val) if str(val) in ["0", "1"] else np.nan

            return parsed

        else:
            raise ValueError("No JSON found in response")

    except Exception as e:
        logger.warning(f"JSON parsing failed, trying fallback pattern extraction: {e}")

        try:
            # Fallback: look for lines like KEY:VALUE (comment)
            for line in response_text.splitlines():
                line = line.strip()
                match = re.match(
                    r"(\w+)\s*:\s*([^\s(]+)", line
                )  # Match KEY:VALUE pairs
                if match:
                    key, val = match.groups()
                    if key in clinical_entities:
                        if key == "NYHA":
                            parsed[key] = roman_to_int(val.strip().upper())
                        elif key == "EF":
                            try:
                                parsed[key] = float(val)
                            except:
                                parsed[key] = np.nan
                        else:
                            parsed[key] = int(val) if val in ["0", "1"] else np.nan
        except Exception as e2:
            logger.error(
                f"Fallback extraction failed: {e2}\nResponse was:\n{response_text}"
            )

        return parsed


# --------------------
# Unified API call with one prompt per note
def call_api(anamnesis, model, max_tokens=2000, temperature=0.2):
    url = f"{base_url}/v1/chat/completions"
    messages = [
        {
            "role": "system",
            "content": "You are a medical assistant. Extract structured clinical information from Italian anamnesis text.",
        },
        {
            "role": "user",
            "content": f"""Here is the patient's anamnesis: 
        {anamnesis}

        Respond with a JSON object like this:
        {{
        "HY": 1,  // Ipertensione (High Blood Pressure)
        "DY": 0,  // Dislipidemia (Dyslipidemia)
        "DB": 1,  // Diabete (Diabetes)
        "CODP": "NaN",  // Malattia Polmonare Cronica Ostruttiva (Chronic Obstructive Pulmonary Disease)
        "NYHA": "II",  // Classificazione NYHA (New York Heart Association class)
        "EF": 55,  // Frazione di Eiezione (Ejection Fraction)
        "RS": 1,  // Ritmo Sinusale (Sinus Rhythm)
        "FA": 0,  // Fibrillazione Atriale (Atrial Fibrillation)
        "FL": "NaN",  // Flutter Atriale (Atrial Flutter)
        "PM": 1,  // Pacemaker
        "BBSx": 0,  // Blocco di Branca Sinistra (Left Bundle Branch Block)
        "BBDx": "NaN",  // Blocco di Branca Destra (Right Bundle Branch Block)
        "TNC": 0,  // Sindrome del Tunnel Carpale (Carpal Tunnel Syndrome)
        "RTS": "NaN",  // Rottura Tendinea Spontanea (Spontaneous Tendon Rupture)
        "SCM": 1,  // Stenosi del Canale Spinale (Spinal Canal Stenosis)
        "PDA": 0,  // Protesi dell'Anca (Hip Prosthesis)
        "PDG": 1,  // Protesi del Ginocchio (Knee Prosthesis)
        "IPC": "NaN",  // Storia Personale o Familiare di Ipertrofia Cardiaca (Personal/Family History of Cardiac Hypertrophy)
        "DSM": 0,  // Disturbi Sensoriali o Motori (Sensory or Motor Disorders)
        "PAR": 1,  // Parestesie (Paresthesia)
        "IPGMM": "NaN"  // Gammopatia Monoclonale (Monoclonal Gammopathy)
        }}

        Instructions:
        - Use 1 if the condition is clearly present
        - Use 0 if it is clearly absent
        - Use "NaN" if the information is not mentioned or unclear
        - For NYHA, use Roman numeral I, II, III, or IV, or "NaN"
        - For EF, use a numeric value (e.g., 50), or "NaN"

        Only return the JSON object. Do not include any explanation, comments, Markdown, or tags. Just the JSON.
        """,
        },
    ]

    try:
        response = session.post(
            url,
            json={
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        logger.debug(f"Raw response text: {response.text}")

        if response.status_code == 200:
            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return extract_all_answers(content)
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return {k: np.nan for k in clinical_entities}
    except Exception as e:
        logger.error(f"API exception: {e}")
        return {k: np.nan for k in clinical_entities}


# --------------------
# Append results to CSV safely
def append_result(result_row):
    with save_lock:
        pd.DataFrame([result_row]).to_csv(
            saving_path, mode="a", header=False, index=False
        )


# --------------------
# Analyze a single row
def analyze_row(row, model, max_tokens, temperature):
    global processed_counter
    anamnesis = row["anamnesis"]
    result_row = {"ID": row["ID"]}
    result_row.update(call_api(anamnesis, model, max_tokens, temperature))
    append_result(result_row)

    with counter_lock:
        processed_counter += 1
        percent = (processed_counter / total_patients) * 100
        logger.info(
            f"Progress: {processed_counter}/{total_patients} patients processed ({percent:.2f}%)"
        )


# Setup logging for info and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lock to safely write to the output file from multiple threads
save_lock = Lock()

# Create a persistent HTTP session with retries to handle transient errors
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))

# Base URL of your local LM Studio server
base_url = "http://localhost:1234"

# --------------------
# CHANGE MODEL HERE to switch the LM used for inference


model = "deepseek-r1-distill-qwen-7b"
# model = "phi-4-mini-instruct"
# model = "sauerkrautlm-gemma-2-9b-it-i1"  # fast
# model = "qwen2.5-7b-instruct"
# model = "wasamikirua-samantha2.0-qwen2.5-14b-ita" #maybe too big
# model = "phi-4-mini-reasoning" #reasoning is too much confused and slow
# --------------------
# Parameters for the API call

max_tokens = 2500
temperature = 0.2

# Output file path depends on model name to keep separate results
saving_path = f"database/labeled/amylo_casi_controlli/llm_annotations_{model}.csv"
# --------------------
# Clinical entities to extract from notes (can be expanded)
# Define the clinical entities to check for
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

logger.info(f"Using model: {model}")
logger.info(f"Saving results to: {saving_path}")
logger.info(f"Clinical entities to extract: {list(clinical_entities.keys())}")
# print the number of clinical entities
logger.info(f"Number of clinical entities: {len(clinical_entities)}")
# --------------------
# Load input data
file_path = "database/unlabeled/amylo_garf_anamnesis.csv"
data = pd.read_csv(file_path)

total_patients = len(data)
logger.info(f"Total patients to process: {total_patients}")

# Subset for testing
# data = data.iloc[:5]

# Filter out already processed entries
if os.path.exists(saving_path) and os.path.getsize(saving_path) > 0:
    logger.info("Loading previously processed entries...")
    processed_ids = pd.read_csv(saving_path)["ID"]
    data = data[~data["ID"].isin(processed_ids)]
    already_processed = len(processed_ids)
else:
    processed_ids = set()
    already_processed = 0
    logger.info("No previously processed entries found.")

processed_counter = already_processed
counter_lock = Lock()
# Print the number of already processed entries
logger.info(f"Already processed entries: {already_processed}")
# Print the number of entries to process
logger.info(f"Entries to process: {len(data)}")


# Create output file if missing
if not os.path.exists(saving_path):
    pd.DataFrame(columns=["ID"] + list(clinical_entities.keys())).to_csv(
        saving_path, index=False
    )

# --------------------

start_time = time.time()
# --------------------
# Run multithreaded processing
with ThreadPoolExecutor(max_workers=50) as executor:
    logger.info(f"Processing {len(data)} notes...")
    futures = [
        executor.submit(analyze_row, row, model, max_tokens, temperature)
        for _, row in data.iterrows()
    ]

    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            logger.error(f"Thread error: {e}")

logger.info(f"Processing complete. Results saved to {saving_path}")


# --------------------
# Log Parameters Used
# --------------------


# Path for saving parameter log
params_log_path = f"database/labeled/amylo_casi_controlli/params_log_{model}.txt"

# Define what was used for logging
elapsed_time = time.time() - start_time if "start_time" in locals() else np.nan

sys_content = (
    "You are a medical assistant. Extract structured clinical information from Italian anamnesis text.",
)
prompt_example = f"""Here is the patient's anamnesis: 
        [ANAMNESIS TEXT] [omitted for brevity]\n\n

        Respond with a JSON object like this:
        {{
        "HY": 1,  // Ipertensione (High Blood Pressure)
        "DY": 0,  // Dislipidemia (Dyslipidemia)
        "DB": 1,  // Diabete (Diabetes)
        "CODP": "NaN",  // Malattia Polmonare Cronica Ostruttiva (Chronic Obstructive Pulmonary Disease)
        "NYHA": "II",  // Classificazione NYHA (New York Heart Association class)
        "EF": 55,  // Frazione di Eiezione (Ejection Fraction)
        "RS": 1,  // Ritmo Sinusale (Sinus Rhythm)
        "FA": 0,  // Fibrillazione Atriale (Atrial Fibrillation)
        "FL": "NaN",  // Flutter Atriale (Atrial Flutter)
        "PM": 1,  // Pacemaker
        "BBSx": 0,  // Blocco di Branca Sinistra (Left Bundle Branch Block)
        "BBDx": "NaN",  // Blocco di Branca Destra (Right Bundle Branch Block)
        "TNC": 0,  // Sindrome del Tunnel Carpale (Carpal Tunnel Syndrome)
        "RTS": "NaN",  // Rottura Tendinea Spontanea (Spontaneous Tendon Rupture)
        "SCM": 1,  // Stenosi del Canale Spinale (Spinal Canal Stenosis)
        "PDA": 0,  // Protesi dell'Anca (Hip Prosthesis)
        "PDG": 1,  // Protesi del Ginocchio (Knee Prosthesis)
        "IPC": "NaN",  // Storia Personale o Familiare di Ipertrofia Cardiaca (Personal/Family History of Cardiac Hypertrophy)
        "DSM": 0,  // Disturbi Sensoriali o Motori (Sensory or Motor Disorders)
        "PAR": 1,  // Parestesie (Paresthesia)
        "IPGMM": "NaN"  // Gammopatia Monoclonale (Monoclonal Gammopathy)
        }}

        Instructions:
        - Use 1 if the condition is clearly present
        - Use 0 if it is clearly absent
        - Use "NaN" if the information is not mentioned or unclear
        - For NYHA, use Roman numeral I, II, III, or IV, or "NaN"
        - For EF, use a numeric value (e.g., 50), or "NaN"

        Only return the JSON object. Do not include any explanation, comments, Markdown, or tags. Just the JSON.
        """
with open(params_log_path, "w") as f:
    f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
    f.write(f"Entities Extracted: {list(clinical_entities.keys())}\n")
    f.write(f"Input File: {file_path}\n")
    f.write(f"Output File: {saving_path}\n")
    f.write(f"Number of Records Processed: {len(data)}\n")
    f.write(f"Model Used: {model}\n")
    f.write(f"Max Tokens: {max_tokens}\n")
    f.write(f"System Prompt:\n{sys_content}\n")
    f.write(f"User Prompt Example:\n{prompt_example}\n")

logger.info(f"Parameters logged to: {params_log_path}")
