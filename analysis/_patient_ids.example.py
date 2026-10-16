"""Template for the local module `_patient_ids.py`, which is not distributed.

Two patients appear in both the case and the control spreadsheet and are excluded. Their
identifiers are patient data. Copy this file to `analysis/_patient_ids.py` (ignored by git)
and fill in the set when running on the original data; leave it empty otherwise.
"""

AMBIGUOUS_IDS = set()
