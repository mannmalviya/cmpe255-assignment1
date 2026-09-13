"""Shared evaluation and latency harness.

Every model in this study — baseline, classical, zero-shot, few-shot, fine-tuned —
is scored through this module, so the Phase 16 table compares like with like.

Two design decisions worth stating, because they change the numbers:

1. Latency is measured END TO END, from a raw text string to a predicted label
   string. For a TF-IDF model that includes vectorisation; for an LLM it includes
   tokenisation, generation and label parsing. A number that excludes
   pre-processing measures a component nobody deploys.

2. Timing runs cycle through REAL held-out rows rather than repeating one string,
   so a cache-friendly single input cannot flatter the result.
"""

import json
import resource
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from config import (LABELS, LATENCY_MIN_RUNS, LATENCY_WARMUP_RUNS,
                    LATENCY_BUDGET_P95_MS, TARGET_MACRO_F1)

# Reference on-demand hourly rates, fixed in Phase 1 §1.5. These are assumptions,
# not measurements: the ratios between models are robust, the absolute dollars
# are not.
HOURLY_RATE_USD = {"cpu": 0.192, "gpu": 0.60}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ----------------------------------------------------------------- quality

def quality_metrics(y_true, y_pred, labels: list = None) -> dict:
    """Accuracy, macro F1 and the full per-class breakdown.

    Macro F1 is primary (Phase 1 §1.3): it weights `surprise` (3.3% of test) the
    same as `joy` (34.75%), so a model cannot buy a good score by abandoning the
    tail. Accuracy is reported alongside because the GAP between them is itself a
    diagnostic — a wide gap means the head classes are being bought with the tail.
    """
    labels = labels or LABELS
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro",
                              zero_division=0))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n": int(len(y_true)),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels,
                                            average="weighted", zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "accuracy_minus_macro_f1": round(acc - macro_f1, 4),
        "per_class": {lab: {"precision": round(float(p[i]), 4),
                            "recall": round(float(r[i]), 4),
                            "f1": round(float(f[i]), 4),
                            "support": int(s[i])}
                      for i, lab in enumerate(labels)},
        "labels_never_predicted": [lab for lab in labels if lab not in set(y_pred)],
        "confusion_matrix": {"labels": labels, "rows_true_cols_pred": cm.tolist()},
        "meets_target_macro_f1": bool(macro_f1 >= TARGET_MACRO_F1),
        "target_macro_f1": TARGET_MACRO_F1,
    }


# ----------------------------------------------------------------- latency

def _percentiles(samples_ms: list) -> dict:
    a = np.asarray(samples_ms)
    return {
        "n_runs": int(a.size),
        "p50_ms": round(float(np.percentile(a, 50)), 4),
        "p95_ms": round(float(np.percentile(a, 95)), 4),
        "p99_ms": round(float(np.percentile(a, 99)), 4),
        "mean_ms": round(float(a.mean()), 4),
        "min_ms": round(float(a.min()), 4),
        "max_ms": round(float(a.max()), 4),
        "stdev_ms": round(float(a.std(ddof=1)), 4) if a.size > 1 else 0.0,
    }


def measure_single_row(predict_one, texts: list,
                       warmup: int = LATENCY_WARMUP_RUNS,
                       runs: int = LATENCY_MIN_RUNS) -> dict:
    """Single-row latency: the wait a user experiences for one message.

    Warm-up runs are executed and DISCARDED, so lazy imports, JIT, allocator
    growth and cold caches do not land in the reported distribution.
    """
    n = len(texts)
    for i in range(warmup):
        predict_one(texts[i % n])

    samples = []
    for i in range(runs):
        t = texts[(warmup + i) % n]
        t0 = time.perf_counter()
        predict_one(t)
        samples.append((time.perf_counter() - t0) * 1000.0)

    out = _percentiles(samples)
    out["warmup_runs_discarded"] = warmup
    out["within_p95_budget"] = bool(out["p95_ms"] <= LATENCY_BUDGET_P95_MS)
    out["p95_budget_ms"] = LATENCY_BUDGET_P95_MS
    return out


def measure_batch(predict_batch, texts: list, batch_sizes: list = (1, 8, 32, 128),
                  warmup: int = 3, repeats: int = 5) -> dict:
    """Batched throughput: what bulk re-scoring costs, which is a different question.

    Reported as rows/second at each batch size, plus the best observed rate. A
    model can be slow per row and still cheap in bulk, and the business cares
    about both.
    """
    n = len(texts)
    results = {}
    for bs in batch_sizes:
        if bs > n:
            continue
        try:
            for _ in range(warmup):
                predict_batch(texts[:bs])
            timings = []
            for r in range(repeats):
                start = (r * bs) % max(1, n - bs)
                chunk = texts[start:start + bs]
                t0 = time.perf_counter()
                predict_batch(chunk)
                timings.append(time.perf_counter() - t0)
        except RuntimeError as exc:
            # A batch size that does not fit in accelerator memory is a RESULT,
            # not a crash: it bounds how far this model can be batched on this
            # hardware. (torch.cuda.OutOfMemoryError subclasses RuntimeError.)
            if "out of memory" not in str(exc).lower():
                raise
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            results[str(bs)] = {"batch_size": bs, "out_of_memory": True}
            continue
        med = statistics.median(timings)
        results[str(bs)] = {
            "batch_size": bs,
            "median_seconds": round(med, 6),
            "rows_per_second": round(bs / med, 2),
            "ms_per_row": round(med / bs * 1000, 4),
            "repeats": repeats,
        }
    fitted = [r for r in results.values() if not r.get("out_of_memory")]
    best = max(fitted, key=lambda r: r["rows_per_second"])
    return {"by_batch_size": results,
            "best_rows_per_second": best["rows_per_second"],
            "best_batch_size": best["batch_size"],
            "batch_sizes_out_of_memory": [r["batch_size"] for r in results.values()
                                          if r.get("out_of_memory")]}


# -------------------------------------------------------------------- cost

def cost_per_1000(rows_per_second: float, device: str) -> dict:
    """USD per 1,000 predictions, from the Phase 1 §1.5 model.

        cost = (1000 / throughput) / 3600 * hourly_rate

    Pure compute: excludes engineering time, storage and serving overhead, all of
    which favour the cheaper model further. For an LLM this is a LOWER BOUND.
    """
    tier = "gpu" if str(device).startswith("cuda") else "cpu"
    rate = HOURLY_RATE_USD[tier]
    seconds = 1000.0 / rows_per_second
    usd = seconds / 3600.0 * rate
    return {
        # Significant figures, not fixed decimals: costs in this study span many
        # orders of magnitude, and fixed rounding reports the cheap models as $0.
        "usd_per_1000_predictions": float(f"{usd:.4g}"),
        "usd_per_1m_predictions": float(f"{usd * 1000:.4g}"),
        "seconds_per_1000": round(seconds, 4),
        "pricing_tier": tier,
        "hourly_rate_usd": rate,
        "formula": "(1000 / rows_per_second) / 3600 * hourly_rate_usd",
        "caveat": "compute only; excludes engineering, storage and serving "
                  "overhead. A lower bound for large models.",
    }


def artifact_size_mb(paths) -> float:
    total = 0
    for p in ([paths] if isinstance(paths, (str, Path)) else paths):
        p = Path(p)
        if p.is_dir():
            # skip symlinks: caches such as the HF hub link snapshots to blobs,
            # and following the links double-counts every file
            total += sum(f.stat().st_size for f in p.rglob("*")
                         if f.is_file() and not f.is_symlink())
        elif p.exists():
            total += p.stat().st_size
    return round(total / 1024**2, 3)


# --------------------------------------------------------------- full run

def evaluate_model(name: str, family: str, y_true, y_pred, predict_one,
                   predict_batch, texts: list, device: str,
                   model_size_mb: float, peak_memory_mb: float,
                   notes: dict = None) -> dict:
    """Assemble one model's complete row for the Phase 16 comparison table."""
    quality = quality_metrics(y_true, y_pred)
    single = measure_single_row(predict_one, texts)
    batch = measure_batch(predict_batch, texts)
    cost = cost_per_1000(batch["best_rows_per_second"], device)

    return {
        "model": name,
        "family": family,
        "device_used": device,
        "quality": quality,
        "latency_single_row": single,
        "throughput_batched": batch,
        "cost": cost,
        "model_size_on_disk_mb": model_size_mb,
        "peak_memory_mb": round(peak_memory_mb, 1),
        "notes": notes or {},
    }
