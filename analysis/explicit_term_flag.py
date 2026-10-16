"""Explicit amyloidosis-term flag on the per-patient anamnesis used by zero-shot classification.
"""
import re
import sys

import pandas as pd

CASI_XLSX = "database/unlabeled/amylo/Casi_new_noname.xlsx"
CONTROLLI_XLSX = "database/unlabeled/amylo/Controlli_new_noname.xlsx"
OUT = "work/_explicit_flag_main_text.csv"  # gitignored

EXPLICIT_TERMS = [
    r"amiloidosi", r"amiloide", r"amyloid",
    r"\battr\b", r"transtiretina",
    r"\bal\b.{0,15}amiloid",
    r"catena leggera",
]
EXPLICIT_RE = re.compile("|".join(EXPLICIT_TERMS), flags=re.IGNORECASE)


def main():
    ca = pd.read_excel(CASI_XLSX, sheet_name="Casi")[["CODINT", "anamnesi"]]
    ca["gold"] = 1
    ctrl = pd.read_excel(CONTROLLI_XLSX, sheet_name="AMYLO Dati CONTROLLI")[["CODINT", "anamnesi"]]
    ctrl["gold"] = 0

    df = pd.concat([ca, ctrl], ignore_index=True).dropna(subset=["CODINT", "anamnesi"])
    df["ID"] = df["CODINT"].astype(int)
    df["explicit_term"] = df["anamnesi"].apply(lambda t: bool(EXPLICIT_RE.search(str(t))))

    out = df[["ID", "gold", "explicit_term"]]
    out.to_csv(OUT, index=False)
    print(f"[shape] {out.shape}")
    print(f"[prevalence] explicit term overall: {out['explicit_term'].mean()*100:.1f}%")
    print(f"[prevalence] explicit term, CA: {out.loc[out['gold']==1,'explicit_term'].mean()*100:.1f}%")
    print(f"[prevalence] explicit term, non-CA: {out.loc[out['gold']==0,'explicit_term'].mean()*100:.1f}%")
    print(f"[saved] -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
