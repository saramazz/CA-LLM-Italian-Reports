"""Specification of the leakage-controlled evaluation: masking term lists (explicit diagnostic
terms, disease-specific investigations and treatments, negated references), temporal strata,
metrics with bootstrap confidence intervals and paired tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

RNG = np.random.default_rng(20260817)
N_BOOT = 2000


# =============================================================================
# 0. Data access: WIRE THESE UP
# =============================================================================

def load_notes() -> pd.DataFrame:
    """
    Returns one row per patient:
        patient_id : str
        text       : str    raw Italian anamnesis
        label      : int    1 = CA, 0 = non-CA
        note_date  : datetime64
        dx_date    : datetime64 or NaT   (NaT for controls)
        age        : float
        sex        : str    'M' / 'F'
        subtype    : str    ATTRwt / AL / ATTRv / Mixed / NA
        split      : str    'train' / 'test'   (patient-level stratified 80/20)

    CHECK BEFORE RUNNING: confirm with the data provider that `dx_date` is the
    confirmatory-test date, not the coding or registry-entry date. The entire
    temporal stratification rests on this.
    """
    raise NotImplementedError


def load_zeroshot_binary() -> pd.DataFrame:
    """patient_id x model -> binary prediction (Prompt 2). Long format:
    columns [patient_id, model, corpus, pred] where corpus in {raw, masked}."""
    raise NotImplementedError


def load_zeroshot_score() -> pd.DataFrame:
    """patient_id x model -> continuous risk score in [0,1] (Prompt 3).
    Same long format as above."""
    raise NotImplementedError


def load_extracted_features() -> pd.DataFrame:
    """patient_id x 21 variables, one frame per extraction source
    (gold, spacy, gpt-4.1-mini, ...). Values in {0, 1, NaN} plus EF, NYHA."""
    raise NotImplementedError


# =============================================================================
# 1. WP-A2: lexical flags
# =============================================================================

# Explicit diagnosis terms. `\bAL\b` is deliberately excluded from the
# automatic pass: in Italian notes it collides with the preposition/article
# "al". Flag candidates and adjudicate manually.
EXPLICIT_DX_TERMS = [
    r"amiloidos\w*", r"amiloide\w*", r"amyloid\w*",
    r"\bATTR\b", r"\bATTRwt\b", r"\bATTRv\b", r"\bTTR\b",
    r"transtiretin\w*", r"tra?nstiretin\w*",
    r"\bAL\s+(?:amiloidosi|cardiaca|sistemica)\b",
    r"catene\s+leggere\s+libere", r"\bFLC\b",
]

# Disease-specific investigations and treatments. Omitting these makes the
# masking cosmetic: "in terapia con tafamidis" leaks as completely as a
# named diagnosis.
DISEASE_SPECIFIC_TERMS = [
    r"tafamidis", r"patisiran", r"vutrisiran", r"inotersen", r"eplontersen",
    r"diflunisal", r"daratumumab", r"acoramidis",
    r"perugini", r"\bDPD\b", r"\bPYP\b", r"\bHMDP\b",
    r"scintigrafia\s+ossea", r"rosso\s+congo", r"congo\s+red",
    r"immunofissazione", r"biopsia\s+endomiocardica",
    r"grasso\s+periombelicale", r"\bBNP\s+sproporzionat\w*",
]

# Reverse-direction leakage in the control arm. Count these separately and
# report the number: it is a finding, not just a cleaning step.
NEGATED_DX_PATTERNS = [
    r"amiloidos\w*\s+esclus\w*", r"esclus\w*\s+amiloidos\w*",
    r"non\s+\w{0,12}\s*amiloidos\w*",
    r"scintigrafia\s+\w*\s*negativ\w*", r"perugini\s+(?:grado\s+)?0",
    r"assenza\s+di\s+amiloidos\w*",
]

# Amyloidosis red flags (WP-B2 baseline). NOT masked: these are the
# legitimate pre-diagnostic signal the pipeline is supposed to detect.
RED_FLAG_TERMS = {
    "TNC":   [r"tunnel\s+carpale", r"\bSTC\b", r"carpal\w*"],
    "RTS":   [r"rottura\s+\w*\s*tendin\w*", r"lesione\s+del\s+tendine"],
    "SCM":   [r"stenosi\s+(?:del\s+)?canale", r"stenosi\s+\w*\s*midollar\w*",
              r"stenosi\s+lombar\w*"],
    "PDA":   [r"protesi\s+d['\s]anca", r"artroprotesi\s+\w*\s*anca"],
    "PDG":   [r"protesi\s+\w*\s*ginocchio"],
    "DSM":   [r"neuropati\w*", r"polineuropati\w*",
              r"disturb\w*\s+sensitiv\w*", r"deficit\s+\w*\s*motori\w*"],
    "PAR":   [r"parestesi\w*", r"formicoli\w*", r"ipoestesi\w*"],
    "IPGMM": [r"gammopati\w*\s+monoclonal\w*", r"\bMGUS\b",
              r"picco\s+monoclonal\w*", r"component\w*\s+monoclonal\w*"],
    "IPC":   [r"ipertrofi\w*\s+\w*\s*ventricolar\w*", r"cardiomiopati\w*\s+ipertrofic\w*"],
}

MASK_TOKEN = "[TERMINE RIMOSSO]"


def _compile(terms: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{t})" for t in terms), re.IGNORECASE)


RE_EXPLICIT = _compile(EXPLICIT_DX_TERMS)
RE_SPECIFIC = _compile(DISEASE_SPECIFIC_TERMS)
RE_NEGATED = _compile(NEGATED_DX_PATTERNS)
RE_RED_FLAGS = {k: _compile(v) for k, v in RED_FLAG_TERMS.items()}


def lexical_flags(text: str) -> dict:
    hits = {k: bool(rx.search(text)) for k, rx in RE_RED_FLAGS.items()}
    return {
        "has_explicit_dx": bool(RE_EXPLICIT.search(text)),
        "has_disease_specific": bool(RE_SPECIFIC.search(text)),
        "has_negated_dx": bool(RE_NEGATED.search(text)),
        "n_explicit_hits": len(RE_EXPLICIT.findall(text)),
        "n_red_flag_types": sum(hits.values()),
        **{f"rf_{k}": v for k, v in hits.items()},
    }


def mask_text(text: str) -> tuple[str, int]:
    """Mask explicit, disease-specific and negated mentions. Both arms."""
    n = 0
    for rx in (RE_NEGATED, RE_EXPLICIT, RE_SPECIFIC):  # negated first: longer spans
        text, k = rx.subn(MASK_TOKEN, text)
        n += k
    return text, n


# =============================================================================
# 2. WP-A1: temporal strata
# =============================================================================

def temporal_stratum(row) -> str:
    """CA notes -> pre / same-day / post. Controls -> 'control'.

    Controls have no diagnosis date, so the control set is held CONSTANT
    across strata. Sensitivity is reported within each CA stratum;
    specificity is by construction unchanged.
    """
    if row["label"] == 0:
        return "control"
    if pd.isna(row["dx_date"]):
        return "unknown"
    d = (row["note_date"] - row["dx_date"]).days
    return "pre" if d < 0 else ("same-day" if d == 0 else "post")


def build_strata(notes: pd.DataFrame) -> pd.DataFrame:
    out = notes.copy()
    out = out.join(out["text"].apply(lambda t: pd.Series(lexical_flags(t))))
    out["temporal"] = out.apply(temporal_stratum, axis=1)
    masked = out["text"].apply(mask_text)
    out["text_masked"] = masked.apply(lambda x: x[0])
    out["n_masked_spans"] = masked.apply(lambda x: x[1])
    return out


# Approximate expected counts, printed as a sanity check.
EXPECTED = {"pre": 80, "same-day": 47, "post": 78, "explicit": 134}


def sanity_check(strata: pd.DataFrame) -> None:
    ca = strata[strata.label == 1]
    print("CA temporal:", ca.temporal.value_counts().to_dict(),
          "| expected ~", EXPECTED)
    print("CA explicit-dx notes:", int(ca.has_explicit_dx.sum()),
          f"({100 * ca.has_explicit_dx.mean():.1f}%)")
    ctrl = strata[strata.label == 0]
    print("Controls with NEGATED dx mention:", int(ctrl.has_negated_dx.sum()),
          "(reported separately: reverse-direction leakage)")
    print("Controls with explicit dx mention:", int(ctrl.has_explicit_dx.sum()))

    # Red flags without any explicit mention.
    n_rf_no_explicit = int(((ca.n_red_flag_types >= 1) & ~ca.has_explicit_dx).sum())
    print(f"CA notes with >=1 red flag AND no explicit mention: {n_rf_no_explicit}"
          f"  (of {len(ca) - int(ca.has_explicit_dx.sum())} CA notes without an explicit mention)")
    print(f"CA notes with >=1 red flag (any):"
          f" {int((ca.n_red_flag_types >= 1).sum())}")


# =============================================================================
# 3. WP-C: metrics with bootstrap CIs, paired tests
# =============================================================================

@dataclass
class Result:
    name: str
    metrics: dict = field(default_factory=dict)

    def __str__(self):
        return self.name + " | " + "  ".join(
            f"{k}={v[0]:.3f} [{v[1]:.3f}-{v[2]:.3f}]" if isinstance(v, tuple)
            else f"{k}={v:.3f}" for k, v in self.metrics.items()
        )


def _point_metrics(y, p, s=None) -> dict:
    m = {
        "F1": f1_score(y, p, zero_division=0),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
    }
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    m["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    if s is not None and len(np.unique(y)) > 1:
        m["AUROC"] = roc_auc_score(y, s)
        m["AUPRC"] = average_precision_score(y, s)
        m["Brier"] = brier_score_loss(y, s)
    return m


def evaluate(y, p, s=None, name="", n_boot=N_BOOT) -> Result:
    """Point estimates with stratified bootstrap 95% CIs (WP-C)."""
    y, p = np.asarray(y), np.asarray(p)
    s = None if s is None else np.asarray(s)
    point = _point_metrics(y, p, s)

    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boot = {k: [] for k in point}
    for _ in range(n_boot):
        idx = np.concatenate([
            RNG.choice(pos, len(pos), replace=True),
            RNG.choice(neg, len(neg), replace=True),
        ])
        try:
            for k, v in _point_metrics(y[idx], p[idx],
                                       None if s is None else s[idx]).items():
                boot[k].append(v)
        except ValueError:
            continue

    return Result(name, {
        k: (v, float(np.nanpercentile(boot[k], 2.5)),
            float(np.nanpercentile(boot[k], 97.5))) if boot[k] else v
        for k, v in point.items()
    })


def mcnemar(y, p_a, p_b) -> tuple[int, int, float]:
    """Exact McNemar for paired classifiers (WP-C)."""
    from scipy.stats import binomtest
    a_ok, b_ok = (np.asarray(p_a) == y), (np.asarray(p_b) == y)
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    pval = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, pval


def bootstrap_f1_diff(y, p_a, p_b, n_boot=N_BOOT) -> tuple[float, float, float]:
    """95% CI on F1(a) - F1(b), paired resampling (WP-C)."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    d0 = f1_score(y, p_a, zero_division=0) - f1_score(y, p_b, zero_division=0)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    ds = []
    for _ in range(n_boot):
        idx = np.concatenate([RNG.choice(pos, len(pos), replace=True),
                              RNG.choice(neg, len(neg), replace=True)])
        ds.append(f1_score(y[idx], p_a[idx], zero_division=0)
                  - f1_score(y[idx], p_b[idx], zero_division=0))
    return d0, float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni over the PRE-SPECIFIED primary comparisons only."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        prev = out[k] = min(1.0, max(prev, (n - i) * p))
    return out


# Declare primaries BEFORE running. Do not add to this list afterwards.
PRIMARY_COMPARISONS = [
    ("zeroshot_masked_prediagnostic", "features_21"),
    ("zeroshot_raw", "zeroshot_masked"),
    ("zeroshot_masked", "baseline_B2_redflag_keywords"),
]


# =============================================================================
# 4. WP-A: the stratified evaluation itself
# =============================================================================

STRATA_SPEC = [
    # (label, row filter). Report all five strata,
    # including unflattering ones.
    ("pre-diagnostic notes only",   lambda d: d.temporal.isin(["pre", "control"])),
    ("same-day notes",              lambda d: d.temporal.isin(["same-day", "control"])),
    ("post-diagnostic notes",       lambda d: d.temporal.isin(["post", "control"])),
    ("explicit amyloidosis terms",  lambda d: (d.label == 0) | d.has_explicit_dx),
    ("no explicit amyloidosis terms", lambda d: (d.label == 0) | ~d.has_explicit_dx),
]


def run_wp_a(strata: pd.DataFrame, preds: pd.DataFrame,
             scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Table 2: models x strata x {raw, masked}."""
    rows = []
    for corpus in ("raw", "masked"):
        pc = preds[preds.corpus == corpus]
        sc = None if scores is None else scores[scores.corpus == corpus]
        for model, g in pc.groupby("model"):
            d = strata.merge(g, on="patient_id")
            sm = None if sc is None else strata.merge(
                sc[sc.model == model], on="patient_id")
            for name, filt in STRATA_SPEC:
                sub = d[filt(d)]
                if sub.label.nunique() < 2 or len(sub) < 20:
                    continue
                s = None
                if sm is not None:
                    s = sm[filt(sm)].set_index("patient_id").reindex(
                        sub.patient_id)["score"].to_numpy()
                r = evaluate(sub.label.to_numpy(), sub.pred.to_numpy(), s,
                             name=f"{model} | {corpus} | {name}")
                rows.append({"model": model, "corpus": corpus, "stratum": name,
                             "n": len(sub), "n_ca": int(sub.label.sum()),
                             **r.metrics})
    return pd.DataFrame(rows)


# WP-A4: the primary de-leaked endpoint. Declared in advance.
PRIMARY_ENDPOINT = ("masked", "pre-diagnostic notes only")


# =============================================================================
# 5. WP-B: baseline ladder
# =============================================================================

def baseline_B1_explicit(strata: pd.DataFrame) -> np.ndarray:
    """Pure lexical ceiling. Expect near-perfect on the raw corpus; that is
    the comparison the keyword baseline is designed for."""
    return strata.has_explicit_dx.astype(int).to_numpy()


def baseline_B2_redflags(strata: pd.DataFrame, k: int = 1) -> np.ndarray:
    """Naive red-flag matching. Run at k=1 and k=2."""
    return (strata.n_red_flag_types >= k).astype(int).to_numpy()


def baseline_B3_B4_B5(strata, features, feature_cols):
    """B3 age+sex; B4 age+sex+12 routine; B5 gold 21 (+/- explicit flag).

    Uses the harmonised WP-C protocol: fit on train, tune with 5x repeated
    stratified 10-fold, evaluate once on the fixed held-out test set.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    df = strata.merge(features, on="patient_id")
    df["sex_M"] = (df.sex == "M").astype(int)
    tr, te = df[df.split == "train"], df[df.split == "test"]

    specs = {
        "B3_age_sex": (["age", "sex_M"], LogisticRegression(max_iter=2000)),
        "B4_age_sex_routine": (["age", "sex_M"] + feature_cols["routine_12"],
                               RandomForestClassifier(random_state=42)),
        "B5_gold_21": (feature_cols["all_21"],
                       RandomForestClassifier(random_state=42)),
        "B5b_gold_21_plus_dx_flag": (feature_cols["all_21"] + ["has_explicit_dx"],
                                     RandomForestClassifier(random_state=42)),
    }
    grids = {
        "LogisticRegression": {"clf__C": [0.01, 0.1, 1, 10]},
        "RandomForestClassifier": {"clf__n_estimators": [200, 500],
                                   "clf__max_depth": [3, 5, None],
                                   "clf__min_samples_leaf": [1, 3, 5]},
    }

    out = {}
    for name, (cols, est) in specs.items():
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()), ("clf", est)])
        gs = GridSearchCV(pipe, grids[type(est).__name__], cv=cv,
                          scoring="f1", n_jobs=-1).fit(tr[cols], tr.label)
        p = gs.best_estimator_.predict(te[cols])
        s = gs.best_estimator_.predict_proba(te[cols])[:, 1]
        out[name] = evaluate(te.label.to_numpy(), p, s, name=name)
    return out


# =============================================================================
# 6. WP-D: realistic-prevalence metrics, DCA, number needed to scan
# =============================================================================

# Verify the 0.13 anchor against Gonzalez-Lopez et al., Eur Heart J 2015
# before citing it. The lower values represent unselected HF referrals.
PREVALENCES = [0.01, 0.05, 0.10, 0.13]


def transported_ppv_npv(sens: float, spec: float,
                        prevalences=PREVALENCES) -> pd.DataFrame:
    """Sens/spec transport across prevalence; PPV/NPV do not.

    The cohort's 47.2% CA prevalence is an enriched case-control design, so
    the observed PPV is not interpretable as a screening PPV.
    """
    rows = []
    for pi in prevalences:
        tp, fn = sens * pi, (1 - sens) * pi
        fp, tn = (1 - spec) * (1 - pi), spec * (1 - pi)
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        rows.append({
            "prevalence": pi, "PPV": ppv, "NPV": npv,
            "flagged_per_1000": 1000 * (tp + fp),
            "true_CA_detected_per_1000": 1000 * tp,
            "CA_missed_per_1000": 1000 * fn,
            # The number a cardiology department will actually ask for.
            "number_needed_to_scan": (1 / ppv) if ppv else np.nan,
        })
    return pd.DataFrame(rows)


def decision_curve(y, score, thresholds=np.arange(0.01, 0.21, 0.01),
                   prevalence: float | None = None) -> pd.DataFrame:
    """Net benefit vs screen-all / screen-none.

    If `prevalence` is given, sensitivity and specificity are transported to
    that prevalence before computing net benefit, so the curve reflects a
    realistic screening population rather than the enriched cohort.
    """
    y, score = np.asarray(y), np.asarray(score)
    n = len(y)
    obs_prev = y.mean()
    rows = []
    for t in thresholds:
        pred = (score >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        if prevalence is None:
            nb = tp / n - (fp / n) * (t / (1 - t))
            nb_all = obs_prev - (1 - obs_prev) * (t / (1 - t))
        else:
            sens = tp / max(int((y == 1).sum()), 1)
            spec = 1 - fp / max(int((y == 0).sum()), 1)
            pi = prevalence
            nb = sens * pi - (1 - spec) * (1 - pi) * (t / (1 - t))
            nb_all = pi - (1 - pi) * (t / (1 - t))
        rows.append({"threshold": t, "net_benefit_model": nb,
                     "net_benefit_screen_all": nb_all,
                     "net_benefit_screen_none": 0.0})
    return pd.DataFrame(rows)


def calibration(y, score, bins=10) -> pd.DataFrame:
    """Calibration curve + intercept/slope (WP-C). Report with Brier score."""
    import statsmodels.api as sm
    y, score = np.asarray(y), np.asarray(np.clip(score, 1e-6, 1 - 1e-6))
    logit = np.log(score / (1 - score))
    slope = sm.Logit(y, sm.add_constant(logit)).fit(disp=0).params[1]
    intercept = sm.Logit(y, np.ones_like(y), offset=logit).fit(disp=0).params[0]
    q = pd.qcut(score, bins, duplicates="drop")
    tbl = (pd.DataFrame({"y": y, "s": score, "bin": q})
           .groupby("bin", observed=True)
           .agg(mean_pred=("s", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
           .reset_index(drop=True))
    tbl.attrs.update(calibration_slope=float(slope),
                     calibration_intercept=float(intercept),
                     brier=float(brier_score_loss(y, score)))
    return tbl


# =============================================================================
# 7. WP-E: matching and adjusted incremental value
# =============================================================================

def match_age_sex(strata: pd.DataFrame, caliper: float = 3.0) -> pd.DataFrame:
    """1:1 exact match on sex + age within +/- caliper years.
    Greedy nearest-neighbour, deterministic order. Report matched-set size."""
    ca = strata[strata.label == 1].sort_values("age").copy()
    ctrl = strata[strata.label == 0].sort_values("age").copy()
    used, pairs = set(), []
    for _, c in ca.iterrows():
        pool = ctrl[(ctrl.sex == c.sex) & (~ctrl.patient_id.isin(used))
                    & ((ctrl.age - c.age).abs() <= caliper)]
        if pool.empty:
            continue
        m = pool.iloc[(pool.age - c.age).abs().argsort().iloc[0]]
        used.add(m.patient_id)
        pairs += [c.patient_id, m.patient_id]
    return strata[strata.patient_id.isin(pairs)]


def incremental_auc(strata: pd.DataFrame, score: np.ndarray) -> dict:
    """Adjusted OR for the zero-shot score and incremental AUC over age+sex.
    Answers 'does the model add anything beyond obvious cohort differences?'"""
    import statsmodels.api as sm
    d = strata.copy()
    d["sex_M"] = (d.sex == "M").astype(int)
    d["score"] = np.clip(score, 1e-6, 1 - 1e-6)
    d["logit_score"] = np.log(d.score / (1 - d.score))

    base = sm.Logit(d.label, sm.add_constant(d[["age", "sex_M"]])).fit(disp=0)
    full = sm.Logit(d.label,
                    sm.add_constant(d[["age", "sex_M", "logit_score"]])).fit(disp=0)
    auc_base = roc_auc_score(d.label, base.predict())
    auc_full = roc_auc_score(d.label, full.predict())
    return {
        "adjusted_OR_score": float(np.exp(full.params["logit_score"])),
        "p_score": float(full.pvalues["logit_score"]),
        "AUC_age_sex": auc_base,
        "AUC_age_sex_plus_score": auc_full,
        "delta_AUC": auc_full - auc_base,
        "LRT_p": float(full.compare_lr_test(base)[1]),
        # Report a DeLong test on delta_AUC as well.
    }


# =============================================================================
# 8. Entry point
# =============================================================================

def main():
    notes = load_notes()
    strata = build_strata(notes)
    sanity_check(strata)

    table2 = run_wp_a(strata, load_zeroshot_binary(), load_zeroshot_score())
    table2.to_csv("results/table2_leakage_strata.csv", index=False)

    primary = table2[(table2.corpus == PRIMARY_ENDPOINT[0])
                     & (table2.stratum == PRIMARY_ENDPOINT[1])]
    print("\n=== PRIMARY DE-LEAKED ENDPOINT (WP-A4) ===")
    print(primary.to_string(index=False))
    print("\nReport this number in the abstract regardless of what it shows.")


if __name__ == "__main__":
    main()
