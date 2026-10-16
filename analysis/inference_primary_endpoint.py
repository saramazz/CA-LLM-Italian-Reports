"""Risk-score inference on the primary-endpoint notes, unmodified and masked, with a local model.
"""
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

NOTES_PATH = "work/_primary_endpoint_notes.csv"  # gitignored
OUT_DIR = "work"  # gitignored outputs (patient-level scores)
TEMPERATURE = 0.2
MAX_WORKERS = 1  # sequential: concurrent requests caused an LM Studio decode failure
# ("failed to decode, ret = 1") that left the model stuck (not responding at all,
# even to trivial prompts) until reloaded -- see run log and audit finding.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

session = requests.Session()
retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))

save_lock = Lock()
counter_lock = Lock()
processed = 0
total = 0


def parse_risk(answer) -> float:
    try:
        cleaned = str(answer).strip().strip("```json").strip("```").strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, (float, int)) and 0 <= parsed <= 1:
                return float(parsed)
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
        logger.error(f"Unexpected risk format (length {len(str(answer))} chars) -- Error: {e}")
    return np.nan


def call_risk(text: str, model: str, base_url: str, max_tokens: int) -> float:
    messages = [
        {
            "role": "system",
            "content": "You are a medical assistant. Based on the Italian anamnesis text, estimate the risk (probability) that the patient has cardiac amyloidosis. Reply with a number between 0 and 1, no explanations or additional text.",
        },
        {
            "role": "user",
            "content": f"Here is the patient's anamnesis:\n{text}\nAnswer with a number between 0 and 1 representing the probability of cardiac amyloidosis. No explanations.",
        },
    ]
    url = f"{base_url}/v1/chat/completions"
    for attempt in range(3):
        try:
            resp = session.post(
                url,
                json={"model": model, "messages": messages,
                      "max_tokens": max_tokens, "temperature": TEMPERATURE},
                timeout=180,
            )
            if resp.status_code != 200:
                logger.error(f"API error attempt {attempt+1}: status={resp.status_code}")
                continue
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            risk = parse_risk(content)
            if not np.isnan(risk):
                return risk
        except Exception as e:
            logger.error(f"Request exception attempt {attempt+1}: {e}")
    return np.nan


def process_one(codint, corpus, text, model, base_url, max_tokens, out_path):
    global processed
    risk = call_risk(text, model, base_url, max_tokens)
    with save_lock:
        pd.DataFrame([{"CODINT": codint, "corpus": corpus, "risk": risk}]).to_csv(
            out_path, mode="a", header=False, index=False
        )
    with counter_lock:
        processed += 1
        if processed % 20 == 0 or processed == total:
            logger.info(f"progress: {processed}/{total}")


def get_model_metadata(base_url: str, model: str) -> dict:
    try:
        resp = session.get(f"{base_url}/api/v0/models", timeout=10)
        for m in resp.json().get("data", []):
            if m.get("id") == model:
                return m
    except Exception as e:
        logger.warning(f"Could not fetch model metadata: {e}")
    return {}


def main():
    model, max_tokens, base_url = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    global total

    meta = get_model_metadata(base_url, model)
    logger.info(f"model={model} state={meta.get('state')} quantization={meta.get('quantization')} "
                f"loaded_context_length={meta.get('loaded_context_length')} "
                f"max_context_length={meta.get('max_context_length')}")
    if meta.get("state") != "loaded":
        logger.error(f"Model {model} is not reported as loaded in LM Studio -- aborting.")
        sys.exit(1)

    notes = pd.read_csv(NOTES_PATH)
    out_path = os.path.join(OUT_DIR, f"_primary_endpoint_risk_{model}.csv")
    if not os.path.exists(out_path):
        pd.DataFrame(columns=["CODINT", "corpus", "risk"]).to_csv(out_path, index=False)

    done = pd.read_csv(out_path) if os.path.getsize(out_path) > 0 else pd.DataFrame(columns=["CODINT", "corpus"])
    done_keys = set(zip(done["CODINT"], done["corpus"]))

    jobs = []
    for _, row in notes.iterrows():
        for corpus, col in (("raw", "REFERTO"), ("masked", "masked_REFERTO")):
            if (row["CODINT"], corpus) in done_keys:
                continue
            text = row[col] if isinstance(row[col], str) else ""
            jobs.append((row["CODINT"], corpus, text))

    total = len(jobs)
    logger.info(f"model={model}: {len(done_keys)} already done, {total} calls remaining "
                f"out of {2 * len(notes)} total (patients={len(notes)} x 2 corpora)")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one, c, corpus, text, model, base_url, max_tokens, out_path)
                   for c, corpus, text in jobs]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.error(f"Thread error: {e}")

    logger.info(f"model={model}: done in {time.time() - t0:.0f}s -> {out_path}")

    meta_after = get_model_metadata(base_url, model)
    logger.info(f"post-run check: state={meta_after.get('state')} "
                f"quantization={meta_after.get('quantization')} "
                f"loaded_context_length={meta_after.get('loaded_context_length')}")


if __name__ == "__main__":
    sys.exit(main())
