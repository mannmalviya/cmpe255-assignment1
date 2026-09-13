"""Phase 15c — Is the fine-tuned model's margin over the SVM real?

SPLIT USED: test predictions ALREADY PRODUCED by Phase 12 (linear SVM) and
Phase 15 (DistilBERT, seed 42). Nothing is fitted, tuned or re-selected. This is
inference ABOUT the test result, not a new use of the test set for any choice.

The Phase 1 decision rule has a threshold: the LLM must win by >= 3 macro F1
points. The observed margin is 3.99. A single test set of 2,000 rows has sampling
error, so the question is not "is 3.99 > 3.00" but "how confident are we that the
TRUE margin exceeds 3.00".

Two methods:
  1. Paired bootstrap on macro F1 (10,000 resamples of the 2,000 test rows, both
     models scored on the SAME resampled rows each time). Gives a confidence
     interval on the margin, and P(margin > 0) and P(margin >= 0.03).
  2. McNemar's exact test on per-row correctness - the standard paired test for
     two classifiers on one test set.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from config import DATA_PROCESSED, LABELS, SEED, ensure_dirs, set_all_seeds
from hardware import save_metrics

N_BOOT = 10_000
THRESHOLD = 0.03


def macro_f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def main():
    set_all_seeds()
    ensure_dirs()
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    y = test["label"].to_numpy()

    svm = joblib.load(DATA_PROCESSED / "models" / "linear_svm.joblib")
    p_svm = np.asarray(svm.predict(test["text"].tolist()))
    bert = pd.read_parquet(DATA_PROCESSED / "models" / "distilbert_finetuned" /
                           "test_predictions_gpu.parquet")
    assert (bert["row_id"].to_numpy() == test["row_id"].to_numpy()).all()
    p_bert = bert["pred"].to_numpy()

    f_svm, f_bert = macro_f1(y, p_svm), macro_f1(y, p_bert)
    observed = f_bert - f_svm
    print(f"observed: SVM {f_svm:.4f}  DistilBERT {f_bert:.4f}  margin {observed:+.4f}")

    # --- 1. paired bootstrap -------------------------------------------------
    # Label -> int codes once, then a fast confusion-based macro F1 per resample.
    code = {l: i for i, l in enumerate(LABELS)}
    yc = np.array([code[v] for v in y])
    sc = np.array([code[v] for v in p_svm])
    bc = np.array([code[v] for v in p_bert])
    K, n = len(LABELS), len(y)

    def fast_f1(t, p):
        cm = np.bincount(t * K + p, minlength=K * K).reshape(K, K)
        tp = np.diag(cm).astype(float)
        prec = np.divide(tp, cm.sum(0), out=np.zeros(K), where=cm.sum(0) > 0)
        rec = np.divide(tp, cm.sum(1), out=np.zeros(K), where=cm.sum(1) > 0)
        f = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K),
                      where=(prec + rec) > 0)
        return f.mean()

    assert abs(fast_f1(yc, bc) - f_bert) < 1e-9, "fast F1 disagrees with sklearn"
    rng = np.random.default_rng(SEED)
    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        deltas[b] = fast_f1(yc[idx], bc[idx]) - fast_f1(yc[idx], sc[idx])

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    boot = {
        "n_resamples": N_BOOT, "paired": True, "seed": SEED,
        "observed_margin": round(observed, 4),
        "bootstrap_mean_margin": round(float(deltas.mean()), 4),
        "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
        "p_margin_gt_0": round(float((deltas > 0).mean()), 4),
        "p_margin_ge_threshold": round(float((deltas >= THRESHOLD).mean()), 4),
        "threshold": THRESHOLD,
        "ci_lower_bound_clears_threshold": bool(lo >= THRESHOLD),
    }

    # --- 2. McNemar exact ------------------------------------------------------
    svm_ok, bert_ok = p_svm == y, p_bert == y
    b01 = int((~svm_ok & bert_ok).sum())    # SVM wrong, BERT right
    b10 = int((svm_ok & ~bert_ok).sum())    # SVM right, BERT wrong
    mc = binomtest(b01, b01 + b10, 0.5, alternative="two-sided")
    mcnemar = {
        "svm_wrong_bert_right": b01, "svm_right_bert_wrong": b10,
        "both_right": int((svm_ok & bert_ok).sum()),
        "both_wrong": int((~svm_ok & ~bert_ok).sum()),
        "exact_p_value": float(f"{mc.pvalue:.3g}"),
        "accuracy_svm": round(float(svm_ok.mean()), 4),
        "accuracy_bert": round(float(bert_ok.mean()), 4),
    }

    save_metrics("phase15_significance", {
        "phase": "15c_significance",
        "splits_used": ["test predictions already produced by Phases 12 and 15"],
        "policy": "inference about an existing result; nothing fitted, tuned or "
                  "selected",
        "paired_bootstrap_macro_f1": boot, "mcnemar_exact": mcnemar}, device="cpu")

    print(json.dumps(boot, indent=2))
    print(json.dumps(mcnemar, indent=2))


if __name__ == "__main__":
    main()
