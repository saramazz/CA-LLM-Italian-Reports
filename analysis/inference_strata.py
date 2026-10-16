"""Risk-score inference with a local model (LM Studio) for the run-to-run reproducibility check and
the four further strata. Strictly sequential, temperature 0.2, max_tokens 16, threshold 0.5.
Usage: python analysis/inference_strata.py <stage>, with LMSTUDIO_MODEL and LMSTUDIO_QUANT set.
"""
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen2.5-7b-instruct")
QUANTIZATION = os.environ.get("LMSTUDIO_QUANT", "Q4_K_M")
BASE_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234")
TEMPERATURE = 0.2
MAX_TOKENS = 16
TIMEOUT = 120

PRIMARY_NOTES = "work/_primary_endpoint_notes.csv"
TEMPORAL_NOTES = "work/_all_temporal_strata_notes_current_spec.csv"
LEXICAL_NOTES = "work/_lexical_arm_masked_current_spec.csv"

OUT_DIR = "work"
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

session = requests.Session()
retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))


def parse_risk(answer):
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
    except Exception:
        pass
    return np.nan


def call_risk(text):
    messages = [
        {"role": "system", "content": "You are a medical assistant. Based on the Italian anamnesis text, estimate the risk (probability) that the patient has cardiac amyloidosis. Reply with a number between 0 and 1, no explanations or additional text."},
        {"role": "user", "content": f"Here is the patient's anamnesis:\n{text}\nAnswer with a number between 0 and 1 representing the probability of cardiac amyloidosis. No explanations."},
    ]
    url = f"{BASE_URL}/v1/chat/completions"
    for attempt in range(3):
        try:
            resp = session.post(url, json={"model": MODEL, "messages": messages,
                                            "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE},
                                 timeout=TIMEOUT)
            if resp.status_code != 200:
                logger.error(f"API error attempt {attempt+1}: status={resp.status_code}")
                continue
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            risk = parse_risk(content)
            if not np.isnan(risk):
                return risk
        except Exception as e:
            logger.error(f"Request exception attempt {attempt+1}: {type(e).__name__}")
    return np.nan


def score_series(texts, label):
    n = len(texts)
    scores = []
    n_unparseable = 0
    t0 = time.time()
    for i, text in enumerate(texts):
        t = text if isinstance(text, str) else ""
        r = call_risk(t)
        if np.isnan(r):
            n_unparseable += 1
        scores.append(r)
        if (i + 1) % 25 == 0 or (i + 1) == n:
            logger.info(f"[{label}] progress: {i+1}/{n}")
    elapsed = time.time() - t0
    logger.info(f"[{label}] done: {n} notes, {n_unparseable} unparseable, {elapsed:.0f}s "
                f"({elapsed/max(n,1):.1f}s/note)")
    return scores, n_unparseable, elapsed


def main():
    stage = sys.argv[1]
    start_ts = datetime.now(timezone.utc).isoformat()

    if stage == "repro":
        notes = pd.read_csv(PRIMARY_NOTES)[["CODINT", "gold", "bucket", "masked_REFERTO"]]
        logger.info(f"[repro check] n={len(notes)} (pre-diagnostic CA + fixed controls, masked corpus)")
        scores, n_bad, elapsed = score_series(notes["masked_REFERTO"].tolist(), "repro_masked_pre_diagnostic")
        out = pd.DataFrame({"CODINT": notes["CODINT"], "gold": notes["gold"],
                             "corpus": "masked", "stratum": "pre_diagnostic_REPRO", "risk": scores})
        out_path = f"{OUT_DIR}/_strata_repro_{MODEL}.csv"
        out.to_csv(out_path, index=False)
        end_ts = datetime.now(timezone.utc).isoformat()
        logger.info(f"[saved] -> {out_path}")
        write_params_log("repro", start_ts, end_ts, elapsed, n_bad, len(notes))

    elif stage in ("strata", "lexical"):
        results = []
        temporal = pd.read_csv(TEMPORAL_NOTES)
        # stage=lexical skips the two temporal strata (resume after an interrupted run)
        temporal_strata = [("same_day", "same_day"), ("post_diagnostic", "post_diagnostic")] if stage == "strata" else []
        for bucket, label in temporal_strata:
            sub = temporal[temporal["bucket"] == bucket]
            logger.info(f"[stratum={label}] n={len(sub)}")
            scores, n_bad, elapsed = score_series(sub["masked_REFERTO"].tolist(), label)
            out = pd.DataFrame({"CODINT": sub["CODINT"], "gold": sub["gold"],
                                 "corpus": "masked", "stratum": label, "risk": scores})
            out_path = f"{OUT_DIR}/_strata_stratum_{label}_{MODEL}.csv"
            out.to_csv(out_path, index=False)
            logger.info(f"[saved] -> {out_path}")
            end_ts = datetime.now(timezone.utc).isoformat()
            write_params_log(label, start_ts, end_ts, elapsed, n_bad, len(sub))
            start_ts = end_ts

        lexical = pd.read_csv(LEXICAL_NOTES)
        for flag, label in [(True, "explicit_present"), (False, "explicit_absent")]:
            sub = lexical[lexical["explicit_term"] == flag]
            logger.info(f"[stratum={label}] n={len(sub)}")
            scores, n_bad, elapsed = score_series(sub["masked_anamnesi"].tolist(), label)
            out = pd.DataFrame({"ID": sub["ID"], "gold": sub["gold"],
                                 "corpus": "masked", "stratum": label, "risk": scores})
            out_path = f"{OUT_DIR}/_strata_stratum_{label}_{MODEL}.csv"
            out.to_csv(out_path, index=False)
            logger.info(f"[saved] -> {out_path}")
            end_ts = datetime.now(timezone.utc).isoformat()
            write_params_log(label, start_ts, end_ts, elapsed, n_bad, len(sub))
            start_ts = end_ts
    elif stage == "referto_raw":
        # Phase 3: UNMASKED REFERTO for the same-day / post-diagnostic CA notes, so the
        # raw-vs-masked comparison uses the same text object on both sides. Controls are
        # not re-scored: their unmasked-REFERTO scores already exist from the primary endpoint.
        temporal = pd.read_csv(TEMPORAL_NOTES)
        for bucket, label in [("same_day", "same_day"), ("post_diagnostic", "post_diagnostic")]:
            sub = temporal[(temporal["bucket"] == bucket) & (temporal["gold"] == 1)]
            logger.info(f"[stratum={label} raw_referto] n={len(sub)}")
            scores, n_bad, elapsed = score_series(sub["REFERTO"].tolist(), f"{label}_raw_referto")
            out = pd.DataFrame({"CODINT": sub["CODINT"], "gold": sub["gold"],
                                 "corpus": "raw_referto", "stratum": label, "risk": scores})
            out_path = f"{OUT_DIR}/_strata_stratum_{label}_raw_referto_{MODEL}.csv"
            out.to_csv(out_path, index=False)
            logger.info(f"[saved] -> {out_path}")
            end_ts = datetime.now(timezone.utc).isoformat()
            write_params_log(f"{label}_raw_referto", start_ts, end_ts, elapsed, n_bad, len(sub))
            start_ts = end_ts
    else:
        raise SystemExit(f"unknown stage {stage}")


def write_params_log(stage_name, start_ts, end_ts, elapsed_s, n_unparseable, n_total):
    log_path = f"results/inference_params_log_{MODEL}_{stage_name}.txt"
    with open(log_path, "w") as f:
        f.write(f"Stage: {stage_name}\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Quantization: {QUANTIZATION}\n")
        f.write(f"Loaded context length: 8192\n")
        f.write(f"Parallel slots: 1 (strictly sequential)\n")
        f.write(f"Temperature: {TEMPERATURE}\n")
        f.write(f"Max tokens: {MAX_TOKENS}\n")
        f.write(f"Threshold (binary decision on continuous risk): 0.5\n")
        f.write(f"Sampling seed: not fixed (none available via this API)\n")
        f.write(f"Request timeout: {TIMEOUT}s\n")
        f.write(f"Start (UTC): {start_ts}\n")
        f.write(f"End (UTC): {end_ts}\n")
        f.write(f"Elapsed: {elapsed_s:.0f}s\n")
        f.write(f"n_total: {n_total}\n")
        f.write(f"n_unparseable_score: {n_unparseable}\n")
    logger.info(f"[params log] -> {log_path}")


if __name__ == "__main__":
    sys.exit(main())
