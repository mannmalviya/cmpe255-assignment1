"""Phase 11 — Naive baselines. The first models in the study.

SPLIT USED: fitted on train_clean (15,923 rows); scored ONCE on test (2,000 rows).
val is not used — these models have no hyper-parameters to tune.

HARD RULE: no other model may be reported before this phase. Without a baseline,
a macro F1 of 0.60 has no meaning. Two baselines are fitted, because they fail in
opposite ways:

  1. MAJORITY  - always predict the most frequent training class. Maximises plain
     accuracy among constant predictors and has the worst possible macro F1,
     which is exactly the failure mode Phase 1 chose macro F1 to expose.
  2. STRATIFIED - sample a label at random from the training class distribution.
     Predicts every class, so macro F1 is non-zero, but is uninformative by
     construction.

Together they bracket "no skill": a model that does not beat BOTH on BOTH metrics
has learned nothing.

This phase also sets the LATENCY FLOOR. A baseline does no work, so its timing is
the irreducible overhead of the harness itself, and every later model's latency
should be read as a cost above this floor.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

import evaluate as ev
from config import (DATA_PROCESSED, LABELS, METRICS, SEED, ensure_dirs,
                    set_all_seeds)
from hardware import save_metrics

MODELS_DIR = DATA_PROCESSED / "models"


def build(strategy: str, train: pd.DataFrame):
    """Fit a DummyClassifier on train_clean and wrap it as a text->label callable."""
    clf = DummyClassifier(strategy=strategy, random_state=SEED)
    # A dummy ignores X, but it must receive one row per training label.
    clf.fit(np.zeros((len(train), 1)), train["label"].to_numpy())

    def predict_batch(texts):
        return clf.predict(np.zeros((len(texts), 1))).tolist()

    def predict_one(text):
        return predict_batch([text])[0]

    return clf, predict_one, predict_batch


def main() -> None:
    set_all_seeds()
    ensure_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    assert len(train) == 15923 and len(test) == 2000

    texts = test["text"].tolist()
    y_true = test["label"].to_numpy()

    results = {}
    for name, strategy in (("baseline_majority", "most_frequent"),
                           ("baseline_stratified", "stratified")):
        clf, predict_one, predict_batch = build(strategy, train)

        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(clf, path, compress=3)

        y_pred = np.array(predict_batch(texts))
        row = ev.evaluate_model(
            name=name,
            family="baseline",
            y_true=y_true, y_pred=y_pred,
            predict_one=predict_one, predict_batch=predict_batch,
            texts=texts, device="cpu",
            model_size_mb=ev.artifact_size_mb(path),
            peak_memory_mb=ev.peak_rss_mb(),
            notes={
                "strategy": strategy,
                "fitted_on": "train_clean (15,923 rows)",
                "scored_on": "test (2,000 rows), once",
                "uses_the_text": False,
                "description": ("always predicts the most frequent training class"
                                if strategy == "most_frequent" else
                                "samples a label from the training class distribution"),
            })
        results[name] = row

        q = row["quality"]
        print(f"=== {name} ===")
        print(f"  accuracy   {q['accuracy']:.4f}")
        print(f"  macro F1   {q['macro_f1']:.4f}   "
              f"(target {q['target_macro_f1']}, met: {q['meets_target_macro_f1']})")
        print(f"  acc - macroF1 gap  {q['accuracy_minus_macro_f1']:+.4f}")
        print(f"  never predicted: {q['labels_never_predicted'] or 'none'}")
        print("  per-class F1: " + "  ".join(
            f"{lab}={q['per_class'][lab]['f1']:.3f}" for lab in LABELS))
        s, b, c = row["latency_single_row"], row["throughput_batched"], row["cost"]
        print(f"  latency p50 {s['p50_ms']:.4f} ms  p95 {s['p95_ms']:.4f} ms  "
              f"({s['n_runs']} runs, {s['warmup_runs_discarded']} warm-up discarded)")
        print(f"  best throughput {b['best_rows_per_second']:,.0f} rows/s "
              f"at batch {b['best_batch_size']}")
        print(f"  cost ${c['usd_per_1000_predictions']:.3g} / 1,000 predictions "
              f"({c['pricing_tier']} tier)")
        print(f"  model on disk {row['model_size_on_disk_mb']} MB\n")

    # The floor every later model is read against.
    floor = {
        "accuracy_floor": max(r["quality"]["accuracy"] for r in results.values()),
        "macro_f1_floor": max(r["quality"]["macro_f1"] for r in results.values()),
        "p95_latency_floor_ms": min(r["latency_single_row"]["p95_ms"]
                                    for r in results.values()),
        "throughput_ceiling_rows_per_second": max(
            r["throughput_batched"]["best_rows_per_second"] for r in results.values()),
        "cost_floor_usd_per_1000": min(r["cost"]["usd_per_1000_predictions"]
                                       for r in results.values()),
        "interpretation": (
            "A model must beat BOTH baselines on BOTH accuracy and macro F1 to have "
            "learned anything. The latency and cost figures are the harness overhead "
            "with no work done, so every later model's cost should be read as an "
            "increment above this floor, not as an absolute."),
    }

    payload = {
        "phase": "11_baseline",
        "splits_used": ["train_clean (fit)", "test (scored once)"],
        "split_policy": "fitted on train_clean; scored once on test. val unused - "
                        "these models have no hyper-parameters.",
        "why_two_baselines": (
            "They fail in opposite ways. Majority maximises plain accuracy among "
            "constant predictors and minimises macro F1. Stratified predicts every "
            "class so macro F1 is non-zero, but is uninformative. Together they "
            "bracket 'no skill'."),
        "models": results,
        "floor": floor,
    }
    path = save_metrics("phase11_baseline", payload, device="cpu")

    print("=== THE FLOOR FOR EVERY LATER MODEL ===")
    print(json.dumps({k: v for k, v in floor.items() if k != "interpretation"},
                     indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
