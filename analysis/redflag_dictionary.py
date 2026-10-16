"""Dictionary of surface forms for the nine red-flag term families (carpal tunnel syndrome,
spontaneous tendon rupture, spinal canal stenosis, hip and knee prosthesis, family history of
cardiac hypertrophy, sensory/motor disturbance, paraesthesia, monoclonal gammopathy), built
from the gold annotations. Used by the red-flag baselines.
"""
import json
import sys
from collections import defaultdict

SRC = "database/labeled/all_clinical_features/9_risk_factors_20240618_corrected.json.conll"
OUT = "work/_redflag_term_dictionary.json"

FAMILIES = ["TNC", "RTS", "SCM", "PDA", "PDG", "IPC", "DSM", "PAR", "IPGMM"]


def main():
    dictionary = defaultdict(set)
    tags = []

    def flush():
        if tags:
            tag0 = tags[0][1][2:]  # strip "B-"
            if tag0 in FAMILIES:
                phrase = " ".join(t[0].lower() for t in tags)
                dictionary[tag0].add(phrase)

    with open(SRC, encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                flush()
                tags = []
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            form, tag = parts[0], parts[1]
            if tag == "O":
                flush()
                tags = []
            elif tag.startswith("B-"):
                flush()
                tags = [(form, tag)]
            else:  # I- continuation
                tags.append((form, tag))
        flush()

    out = {k: sorted(v) for k, v in dictionary.items()}
    for fam in FAMILIES:
        out.setdefault(fam, [])
        print(f"[{fam}] {len(out[fam])} distinct surface forms")

    with open(OUT, "w", encoding="utf-8") as fout:
        json.dump(out, fout, ensure_ascii=False, indent=2)
    print(f"\n[saved] -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
