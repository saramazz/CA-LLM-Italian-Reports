# CA-LLM-Italian-Reports

Code and aggregate results for **"Detecting Cardiac Amyloidosis in Italian Cardiology Reports:
Structured Variable Extraction versus Direct Free-Text Analysis"**.

The repository contains the scripts behind the results, tables and figures of the paper, and
the aggregate outputs they produce. **It contains no patient data**: the clinical records are
protected under the GDPR and are not distributed.

## Structure

| Folder | Content |
|---|---|
| `extraction/` | Prompt-based extraction of the 21 clinical variables (proprietary and local models), iterative single-variable arm, zero-shot classification, extraction scoring |
| `analysis/` | Every analysis reported in the paper, one script per analysis (listed below) |
| `results/` | Aggregate outputs (metrics, counts, confidence intervals, run parameter logs) |
| `results/figures/` | Figures 2–4 and the supplementary scatter figure |

## Analyses

| Topic | Script(s) in `analysis/` |
|---|---|
| Table 1 denominators | `table1_denominators.py` |
| Extraction performance | `extraction_metrics.py`, `extraction_metrics_ef_nyha.py`, `gold_rebuild.py` |
| Classification from extracted variables | `classification_harmonised.py` |
| Baselines B1–B5b | `redflag_dictionary.py`, `baselines.py`, `baselines_full_metrics.py` |
| Zero-shot at the 0.5 threshold | `zero_shot_threshold.py`, `zero_shot_threshold_table.py` |
| Paired comparisons, calibration, decision curves, predictive values | `paired_comparisons.py`, `paired_comparisons_harmonised.py`, `calibration.py`, `decision_curve.py`, `predictive_values.py`, `confusion_auprc.py` |
| Matched analysis | `matched_analysis.py` |
| Leakage-controlled evaluation | `leakage_spec.py` (masking term lists and strata), `cohort_temporal.py`, `explicit_term_flag.py`, `note_selection_temporal.py`, `notes_primary_endpoint.py`, `notes_masked_temporal.py`, `notes_masked_lexical.py`, `inference_primary_endpoint.py`, `inference_strata.py`, `metrics_primary_endpoint.py`, `metrics_strata.py` |
| Figures | `figure2.py`, `figure3.py`, `figure4.py` |

Each script states in its header what it computes. Its outputs are written to `results/`
(aggregate) or `work/` (patient-level, not versioned).

## Running

- Python 3.9; packages in `requirements.txt`. Run from the repository root.
- The original data are expected in `database/` and the cached model outputs in
  `saved_results/`. Neither is distributed.
- Copy `analysis/_patient_ids.example.py` to `analysis/_patient_ids.py`.
- Proprietary models read the API key from the `OPENAI_API_KEY` environment variable.
- Local models are served with LM Studio (OpenAI-compatible endpoint, default
  `http://localhost:1234`); quantisation and settings of every run are in
  `results/inference_params_log_*.txt`.

## Licence

MIT, see `LICENSE`.
