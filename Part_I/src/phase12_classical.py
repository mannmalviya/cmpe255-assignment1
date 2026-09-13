"""Phase 12 — Classical models: logistic regression, linear SVM, gradient boosting.

SPLIT USED:
  Tuning  : train_clean + val (17,923 rows), 5-fold stratified cross-validation.
            The test split is NOT touched during any search.
  Scoring : each tuned model is refitted on all of train_clean + val, then scored
            ONCE on test (2,000 rows).

Selection criterion is macro F1, matching Phase 1 §1.3. Selecting on accuracy
would reward exactly the head-class-for-tail trade that Phase 11 showed the
majority baseline making.

The grid encodes three questions raised by earlier phases, so each is decided by
cross-validation rather than by assertion:
  - min_df 1 vs 2        (Phase 3: 51% of types are hapax; Phase 7: min_df=2 halves
                          the feature space for 0.5 non-zeros per row)
  - unigram vs bigram    (Phase 10 PREDICTED bigrams will not help)
  - class_weight balanced vs none  (Phase 11: the tail needs pushing)
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import evaluate as ev
from config import (DATA_PROCESSED, LABELS, SEED, ensure_dirs, set_all_seeds)
from hardware import save_metrics

MODELS_DIR = DATA_PROCESSED / "models"
CKPT_DIR = DATA_PROCESSED / "phase12_checkpoints"
N_FOLDS = 5

# Parallelism is capped deliberately. The first attempt used n_jobs=-1 for both
# GridSearchCV and LightGBM on a 16-core machine, which means 16 worker processes
# each spawning 16 LightGBM threads - 256 threads all allocating histogram
# buffers. The run was SIGTERMed with no output. 4 x 4 keeps the machine busy
# without oversubscribing it.
SEARCH_JOBS = 4
MODEL_JOBS = 4

# Shared representation grid. Kept identical across the linear models so the
# comparison between them is not confounded by different feature spaces.
TFIDF_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
}


def base_vectorizer() -> TfidfVectorizer:
    # lowercase=False and token_pattern=r"\S+" because Phase 5 proved the corpus
    # is already lowercase with no punctuation: the defaults would be dead work.
    return TfidfVectorizer(sublinear_tf=True, lowercase=False, token_pattern=r"\S+")


def search(key: str, name: str, estimator, grid: dict, X, y, cv) -> dict:
    """Run one grid search, checkpointing the fitted result to disk.

    Checkpointing exists because the first attempt at this phase was killed
    part-way through and lost all three searches. A completed search is now
    reloaded rather than repeated.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"{key}.joblib"
    if ckpt.exists():
        saved = joblib.load(ckpt)
        print(f"  {name}: loaded checkpoint, CV macro F1 "
              f"{saved['grid_search'].best_score_:.4f}", flush=True)
        return saved

    pipe = Pipeline([("tfidf", base_vectorizer()), ("clf", estimator)])
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=SEARCH_JOBS,
                      refit=True, verbose=1, pre_dispatch="2*n_jobs")
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    print(f"  {name}: {n_combos} combos x {N_FOLDS} folds = "
          f"{n_combos * N_FOLDS} fits, {SEARCH_JOBS} workers ...", flush=True)

    t0 = time.perf_counter()
    gs.fit(X, y)
    secs = time.perf_counter() - t0

    cvres = pd.DataFrame(gs.cv_results_)
    out = {"grid_search": gs, "seconds": secs, "cv_results": cvres}
    joblib.dump(out, ckpt, compress=3)
    print(f"  {name}: best CV macro F1 {gs.best_score_:.4f}  "
          f"({len(cvres)} combos x {N_FOLDS} folds in {secs:.0f}s)", flush=True)
    print(f"      best params: {gs.best_params_}", flush=True)
    return out


def representation_effects(cvres: pd.DataFrame) -> dict:
    """Marginal effect of each representation choice, averaged over the rest.

    Reported because the three questions above deserve an answer that is not just
    "whatever the winner happened to use".
    """
    out = {}
    for col, pretty in (("param_tfidf__ngram_range", "ngram_range"),
                        ("param_tfidf__min_df", "min_df"),
                        ("param_clf__class_weight", "class_weight")):
        if col not in cvres:
            continue
        g = cvres.groupby(cvres[col].astype(str))["mean_test_score"]
        out[pretty] = {k: round(float(v), 4) for k, v in g.mean().items()}
    return out


def main() -> None:
    set_all_seeds()
    ensure_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
    val = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")

    dev = pd.concat([train, val], ignore_index=True)
    X, y = dev["text"].to_numpy(), dev["label"].to_numpy()
    assert len(dev) == 17923 and len(test) == 2000
    print(f"development set for CV: {len(dev):,} rows "
          f"(train_clean {len(train):,} + val {len(val):,}). test untouched.\n",
          flush=True)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    texts, y_true = test["text"].tolist(), test["label"].to_numpy()

    searches = {}
    print("=== CROSS-VALIDATED SEARCH (train_clean + val only) ===", flush=True)

    searches["logreg"] = search(
        "logreg", "logistic regression",
        LogisticRegression(max_iter=2000, random_state=SEED),
        {**TFIDF_GRID, "clf__C": [1.0, 5.0, 10.0],
         "clf__class_weight": [None, "balanced"]},
        X, y, cv)

    searches["linear_svm"] = search(
        "linear_svm", "linear SVM",
        LinearSVC(random_state=SEED, max_iter=5000),
        {**TFIDF_GRID, "clf__C": [0.1, 0.5, 1.0],
         "clf__class_weight": [None, "balanced"]},
        X, y, cv)

    # Gradient boosting gets a narrower grid: trees on high-dimensional sparse
    # text are far slower to fit than linear models, which is itself part of the
    # cost story this study is measuring.
    searches["gradient_boosting"] = search(
        "gradient_boosting", "gradient boosting (LightGBM)",
        LGBMClassifier(n_estimators=300, random_state=SEED, n_jobs=MODEL_JOBS,
                       verbose=-1),
        {"tfidf__ngram_range": [(1, 1)], "tfidf__min_df": [2],
         "clf__num_leaves": [31, 63], "clf__learning_rate": [0.1, 0.2],
         "clf__class_weight": [None, "balanced"]},
        X, y, cv)

    print("\n=== FINAL SCORING ON TEST (once per model) ===", flush=True)
    results = {}
    for key, s in searches.items():
        gs = s["grid_search"]
        model = gs.best_estimator_          # already refitted on all of dev
        path = MODELS_DIR / f"{key}.joblib"
        joblib.dump(model, path, compress=3)

        def predict_batch(batch, _m=model):
            return list(_m.predict(batch))

        def predict_one(text, _m=model):
            return _m.predict([text])[0]

        y_pred = np.array(predict_batch(texts))
        vec = model.named_steps["tfidf"]
        row = ev.evaluate_model(
            name=key, family="classical",
            y_true=y_true, y_pred=y_pred,
            predict_one=predict_one, predict_batch=predict_batch,
            texts=texts, device="cpu",
            model_size_mb=ev.artifact_size_mb(path),
            peak_memory_mb=ev.peak_rss_mb(),
            notes={
                "tuned_on": "train_clean + val (17,923 rows), "
                            f"{N_FOLDS}-fold stratified CV",
                "selection_metric": "macro F1",
                "scored_on": "test (2,000 rows), once",
                "best_params": {k: str(v) for k, v in gs.best_params_.items()},
                "best_cv_macro_f1": round(float(gs.best_score_), 4),
                "cv_search_seconds": round(s["seconds"], 1),
                "n_param_combinations": int(len(s["cv_results"])),
                "vocabulary_size": int(len(vec.vocabulary_)),
                "representation_effects_on_cv_macro_f1":
                    representation_effects(s["cv_results"]),
            })
        results[key] = row

        q = row["quality"]
        gap = q["macro_f1"] - float(gs.best_score_)
        print(f"\n  {key}")
        print(f"    CV macro F1 {gs.best_score_:.4f} -> TEST macro F1 "
              f"{q['macro_f1']:.4f}  ({gap:+.4f})")
        print(f"    accuracy {q['accuracy']:.4f}   "
              f"meets 0.85 target: {q['meets_target_macro_f1']}")
        print("    per-class F1: " + "  ".join(
            f"{lab}={q['per_class'][lab]['f1']:.3f}" for lab in LABELS))
        s_, b_, c_ = (row["latency_single_row"], row["throughput_batched"],
                      row["cost"])
        print(f"    p50 {s_['p50_ms']:.3f} ms  p95 {s_['p95_ms']:.3f} ms  "
              f"(budget 50 ms: {s_['within_p95_budget']})")
        print(f"    throughput {b_['best_rows_per_second']:,.0f} rows/s  "
              f"cost ${c_['usd_per_1000_predictions']:.3g}/1k  "
              f"disk {row['model_size_on_disk_mb']} MB")

    # ------------------------------------------------------------- figure
    import matplotlib.pyplot as plt
    import viz
    from config import FIGURES
    viz.apply_style()

    best_key_fig = max(results, key=lambda k: results[k]["quality"]["macro_f1"])
    order = ["joy", "sadness", "anger", "fear", "love", "surprise"]
    train_counts = {lab: int((train["label"] == lab).sum()) for lab in order}
    f1s = [results[best_key_fig]["quality"]["per_class"][lab]["f1"] for lab in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    xs = [train_counts[lab] for lab in order]
    ax1.scatter(xs, [f * 100 for f in f1s], s=46, color=viz.PRIMARY, zorder=4,
                edgecolors="none")
    for lab, x, f in zip(order, xs, f1s):
        ax1.annotate(f"  {lab}", (x, f * 100), fontsize=8, color=viz.INK_2,
                     va="center")
    rho = float(pd.Series(xs).corr(pd.Series(f1s), method="spearman"))
    ax1.set_xscale("log")
    ax1.set_xlim(400, 11000)
    ax1.set_ylim(60, 100)
    ax1.set_xlabel("training rows for that class (log)")
    ax1.set_ylabel("test F1 for that class (%)")
    viz.value_grid(ax1, "y")
    viz.titles(ax1, "a.  Per-class F1 is ordered by class size",
               f"{best_key_fig}, test split. Spearman rank correlation "
               f"rho = {rho:+.2f} over six classes.")

    # The three representation questions, as marginal CV means.
    eff = results["linear_svm"]["notes"]["representation_effects_on_cv_macro_f1"]
    questions = [("ngram_range", "(1, 1)", "(1, 2)", "unigram", "+ bigram"),
                 ("min_df", "1", "2", "min_df 1", "min_df 2"),
                 ("class_weight", "None", "balanced", "no weights", "balanced")]
    x = np.arange(len(questions))
    w = 0.34
    va = [eff[q[0]][q[1]] * 100 for q in questions]
    vb = [eff[q[0]][q[2]] * 100 for q in questions]
    b1 = ax2.bar(x - w / 2, va, width=w * 0.92, color=viz.SERIES[0], zorder=3)
    b2 = ax2.bar(x + w / 2, vb, width=w * 0.92, color=viz.SERIES[1], zorder=3)
    viz.rounded_bars(ax2, b1, "v")
    viz.rounded_bars(ax2, b2, "v")
    for xi, v, q in zip(x - w / 2, va, questions):
        ax2.text(xi, v + 0.25, f"{q[3]}\n{v:.1f}", ha="center", va="bottom",
                 fontsize=7, color=viz.INK_2)
    for xi, v, q in zip(x + w / 2, vb, questions):
        ax2.text(xi, v + 0.25, f"{q[4]}\n{v:.1f}", ha="center", va="bottom",
                 fontsize=7, color=viz.INK_2)
    ax2.set_xticks(x, [q[0] for q in questions])
    ax2.set_ylim(78, 90)
    ax2.set_ylabel("mean CV macro F1 (%), averaged over the rest of the grid")
    viz.value_grid(ax2, "y")
    viz.titles(ax2, "b.  Three predictions, tested on train+val",
               "bigrams HURT (Phase 10 predicted no help); balanced weights help "
               "(Phase 11);\nmin_df=2 is free (Phase 3). Linear SVM grid.")

    viz.caption(fig, "Panel a: test split, scored once. Panel b: 5-fold CV on "
                     "train_clean + val only; test never used for tuning.")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig12_classical.png")
    plt.close(fig)

    baseline = json.loads((DATA_PROCESSED.parent.parent / "reports" / "metrics" /
                           "phase11_baseline.json").read_text())
    floor = baseline["floor"]
    best_key = max(results, key=lambda k: results[k]["quality"]["macro_f1"])

    payload = {
        "phase": "12_classical",
        "splits_used": ["train_clean + val (5-fold CV tuning)",
                        "test (scored once per model)"],
        "split_policy": "all tuning on train_clean + val; test read once per model "
                        "to produce its final row. No test row influenced any "
                        "hyper-parameter.",
        "selection_metric": "macro F1",
        "cv": {"folds": N_FOLDS, "stratified": True, "shuffle": True, "seed": SEED,
               "development_rows": int(len(dev))},
        "models": results,
        "best_model": best_key,
        "figures": ["fig12_classical.png"],
        "vs_baseline": {
            "baseline_macro_f1_floor": floor["macro_f1_floor"],
            "baseline_accuracy_floor": floor["accuracy_floor"],
            "best_classical_macro_f1": results[best_key]["quality"]["macro_f1"],
            "macro_f1_improvement_over_floor": round(
                results[best_key]["quality"]["macro_f1"] - floor["macro_f1_floor"], 4),
            "baseline_p95_latency_ms": floor["p95_latency_floor_ms"],
            "best_classical_p95_latency_ms":
                results[best_key]["latency_single_row"]["p95_ms"],
        },
    }
    path = save_metrics("phase12_classical", payload, device="cpu")

    print("\n=== REPRESENTATION QUESTIONS, ANSWERED BY CV ===")
    for key in ("logreg", "linear_svm"):
        eff = results[key]["notes"]["representation_effects_on_cv_macro_f1"]
        print(f"  {key}:")
        for q, vals in eff.items():
            print(f"      {q:<14} {vals}")
    print(f"\nbest classical model: {best_key} "
          f"(test macro F1 {results[best_key]['quality']['macro_f1']:.4f})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
