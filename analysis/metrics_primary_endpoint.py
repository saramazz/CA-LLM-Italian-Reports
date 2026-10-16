"""Primary-endpoint metrics: F1, sensitivity, specificity, AUROC with bootstrap CIs, paired F1
difference, exact McNemar and paired DeLong tests (unmodified vs masked).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from leakage_spec import evaluate, mcnemar, bootstrap_f1_diff  # noqa: E402

NOTES_PATH = "work/_primary_endpoint_notes.csv"


def delong_paired_auc_test(y, score_a, score_b):
    """Paired DeLong test for AUC(a) vs AUC(b) on the SAME cases (same y).
    Standard DeLong (1988) structural-components method, O(n^2), fine for
    n in the hundreds."""
    y = np.asarray(y)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    m, n = len(pos), len(neg)

    def structural_components(score):
        score = np.asarray(score)
        sp, sn = score[pos], score[neg]
        # psi(a,b) = 1 if a>b, 0.5 if a==b, 0 if a<b
        diff = sp[:, None] - sn[None, :]
        psi = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
        v10 = psi.mean(axis=1)  # per positive case
        v01 = psi.mean(axis=0)  # per negative case
        auc = psi.mean()
        return v10, v01, auc

    v10_a, v01_a, auc_a = structural_components(score_a)
    v10_b, v01_b, auc_b = structural_components(score_b)

    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1) / m
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1) / n
    s = s10 + s01
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    diff = auc_a - auc_b
    if var_diff <= 0:
        return auc_a, auc_b, diff, float("nan"), float("nan")
    z = diff / np.sqrt(var_diff)
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc_a, auc_b, diff, z, p


def main():
    model = sys.argv[1]
    risk_path = f"work/_primary_endpoint_risk_{model}.csv"

    notes = pd.read_csv(NOTES_PATH)[["CODINT", "gold", "bucket"]]
    risk = pd.read_csv(risk_path)
    wide = risk.pivot(index="CODINT", columns="corpus", values="risk")
    d = wide.merge(notes, on="CODINT", how="inner").dropna(subset=["raw", "masked"])

    print(f"=== PRIMARY ENDPOINT: {model} ===")
    print(f"n={len(d)} (CA pre-diagnostic={int((d.gold==1).sum())}, "
          f"controls={int((d.gold==0).sum())})")
    n_dropped = len(notes) - len(d)
    if n_dropped:
        print(f"[note] {n_dropped} patient(s) missing a valid score in one or both "
              f"arms, excluded from this paired analysis")

    y = d["gold"].to_numpy()
    s_raw = d["raw"].to_numpy()
    s_masked = d["masked"].to_numpy()
    p_raw = (s_raw >= 0.5).astype(int)
    p_masked = (s_masked >= 0.5).astype(int)

    for name, p, s in (("raw", p_raw, s_raw), ("masked", p_masked, s_masked)):
        r = evaluate(y, p, s, name=name)
        print(r)

    b, c, p_mcnemar = mcnemar(y, p_raw, p_masked)
    print(f"\nExact McNemar (raw vs masked, 0.5 threshold): "
          f"raw-correct/masked-wrong={b}, masked-correct/raw-wrong={c}, p={p_mcnemar:.4g}")

    d0, lo, hi = bootstrap_f1_diff(y, p_raw, p_masked)
    print(f"F1(raw) - F1(masked) = {d0:.3f} [95% CI {lo:.3f}, {hi:.3f}]")

    auc_raw, auc_masked, auc_diff, z, p_delong = delong_paired_auc_test(y, s_raw, s_masked)
    print(f"DeLong (paired): AUROC(raw)={auc_raw:.3f}, AUROC(masked)={auc_masked:.3f}, "
          f"diff={auc_diff:.3f}, z={z:.3f}, p={p_delong:.4g}")

    out = pd.DataFrame([{
        "model": model, "n": len(d),
        "n_ca_pre_diagnostic": int((d.gold == 1).sum()),
        "n_controls": int((d.gold == 0).sum()),
        "f1_raw": f1_score(y, p_raw, zero_division=0),
        "f1_masked": f1_score(y, p_masked, zero_division=0),
        "f1_diff": d0, "f1_diff_ci_lo": lo, "f1_diff_ci_hi": hi,
        "sens_raw": recall_score(y, p_raw, zero_division=0),
        "sens_masked": recall_score(y, p_masked, zero_division=0),
        "auroc_raw": auc_raw, "auroc_masked": auc_masked,
        "auroc_diff": auc_diff, "delong_z": z, "delong_p": p_delong,
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p_mcnemar,
    }])
    out_path = f"results/primary_endpoint_{model}_stats.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[saved] -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
