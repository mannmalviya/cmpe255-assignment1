"""Phase 13 — Zero-shot LLM classification.

SPLIT USED: test only (2,000 rows), scored ONCE. Nothing is fitted and nothing is
tuned — that is what zero-shot means. The training split is never read, not even
for a class prior.

FIXED BEFORE ANY RESULT WAS SEEN (in src/llm_common.py):
  - the model:   Qwen/Qwen2.5-1.5B-Instruct, bf16, greedy decoding
  - the prompt:  one system message naming the six labels and demanding one word
  - the parser:  strict exact match on the whole output; unparseable = ERROR

The prompt is not revised after looking at scores. Revising it would be
prompt-tuning on the test split, the same violation as tuning a hyper-parameter
on it. Every unparseable output is logged and reported.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

import evaluate as ev
import llm_common as L
from config import DATA_PROCESSED, LABELS, METRICS, SEED, ensure_dirs, set_all_seeds
from hardware import save_metrics

BATCH_SIZE = 32


def model_disk_mb(model_id: str) -> float:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_id.replace("/", "--")
    d = cache / slug
    if not d.exists():
        return 0.0
    # The HF cache stores real files in blobs/ and SYMLINKS to them in
    # snapshots/. Following the symlinks double-counts every weight file; the
    # first run of this phase reported 5,910.8 MB for a 2,955.4 MB model.
    return round(sum(f.stat().st_size for f in d.rglob("*")
                     if f.is_file() and not f.is_symlink()) / 1024**2, 1)


def main() -> None:
    set_all_seeds()
    ensure_dirs()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Phase 13 must run on the GPU: a CPU decoder "
            "latency measurement would describe a system nobody deploys. Install "
            "the CUDA build of torch and re-run.")

    device = "cuda:0"
    torch.cuda.reset_peak_memory_stats()

    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    assert len(test) == 2000
    texts, y_true = test["text"].tolist(), test["label"].to_numpy()

    print(f"loading {L.MODEL_ID} on {device} ...", flush=True)
    t0 = time.perf_counter()
    runner = L.LLMRunner(device=device).load()
    load_s = time.perf_counter() - t0
    print(f"  loaded in {load_s:.1f}s, {runner.parameters_billions()}B params",
          flush=True)

    # A visible sanity check on three rows before committing to 2,000.
    print("\n  sanity check (3 rows):", flush=True)
    for t in texts[:3]:
        gen = runner.generate([t])[0]
        p = L.parse_label(gen)
        print(f"    {t[:58]!r}\n      raw={gen!r}  strict={p.strict}", flush=True)

    print(f"\n=== SCORING test (2,000 rows, batch {BATCH_SIZE}) ===", flush=True)
    labels, raws, gen_s = L.run_full_split(runner, texts, batch_size=BATCH_SIZE)
    y_pred = np.array(labels)

    parse = L.parse_report(raws)
    print(f"\n  strict parse rate  {parse['strict_parse_rate']:.4f}   "
          f"lenient {parse['lenient_parse_rate']:.4f}")
    print(f"  unparseable: {parse['n_unparseable_strict']} "
          f"({parse['n_unparseable_strict']/len(raws):.2%}) - all counted as errors")
    if parse["most_common_unparseable_outputs"]:
        print("  most common unparseable outputs:")
        for shape, n in parse["most_common_unparseable_outputs"][:6]:
            print(f"      {n:>4}x  {shape!r}")

    # The scored result uses STRICT parsing. The lenient variant is computed too,
    # as a diagnostic of how much of the error is formatting rather than judgement.
    y_pred_lenient = np.array([r["lenient"] if r["lenient"] else "UNPARSEABLE"
                               for r in raws])

    print("\n=== LATENCY (GPU, full protocol) ===", flush=True)
    row = ev.evaluate_model(
        name="zero_shot_qwen2.5_1.5b", family="llm_zero_shot",
        y_true=y_true, y_pred=y_pred,
        predict_one=runner.predict_one, predict_batch=runner.predict_batch,
        texts=texts, device=device,
        model_size_mb=model_disk_mb(L.MODEL_ID),
        peak_memory_mb=ev.peak_rss_mb(),
        notes={
            "model_id": L.MODEL_ID,
            "parameters_billions": runner.parameters_billions(),
            "dtype": "bfloat16",
            "decoding": "greedy, deterministic (do_sample=False)",
            "max_new_tokens": L.MAX_NEW_TOKENS,
            "shots": 0,
            "split": "test only (2,000 rows), scored once",
            "trained_or_tuned": False,
            "system_prompt": L.SYSTEM_PROMPT,
            "prompt_fixed_before_seeing_results": True,
            "model_load_seconds": round(load_s, 1),
            "full_split_generation_seconds": round(gen_s, 1),
            "full_split_rows_per_second": round(len(texts) / gen_s, 2),
            "batch_size_used_for_split": BATCH_SIZE,
            "gpu_memory": runner.gpu_memory_mb(),
        })
    # Recorded AFTER the latency protocol: the batch-128 throughput test is what
    # sets the true peak. The first run captured it beforehand (3,360 MB vs 4,540).
    row["notes"]["gpu_memory"] = runner.gpu_memory_mb()
    row["peak_memory_mb"] = row["notes"]["gpu_memory"]["peak_allocated_mb"]
    row["peak_memory_kind"] = "GPU peak allocated (torch.cuda.max_memory_allocated)"
    row["parsing"] = {k: v for k, v in parse.items() if k != "all_unparseable"}
    row["quality_if_lenient_parsing"] = ev.quality_metrics(y_true, y_pred_lenient)

    q = row["quality"]
    ql = row["quality_if_lenient_parsing"]
    print(f"\n  STRICT  accuracy {q['accuracy']:.4f}  macro F1 {q['macro_f1']:.4f}")
    print(f"  LENIENT accuracy {ql['accuracy']:.4f}  macro F1 {ql['macro_f1']:.4f}"
          f"   (diagnostic only)")
    print("  per-class F1: " + "  ".join(
        f"{lab}={q['per_class'][lab]['f1']:.3f}" for lab in LABELS))
    s, b, c = row["latency_single_row"], row["throughput_batched"], row["cost"]
    print(f"  p50 {s['p50_ms']:.1f} ms  p95 {s['p95_ms']:.1f} ms  "
          f"(50 ms budget: {s['within_p95_budget']})")
    print(f"  throughput {b['best_rows_per_second']:.1f} rows/s at batch "
          f"{b['best_batch_size']}")
    print(f"  cost ${c['usd_per_1000_predictions']:.4g}/1k ({c['pricing_tier']} tier)")
    print(f"  model on disk {row['model_size_on_disk_mb']} MB   "
          f"GPU peak {runner.gpu_memory_mb()}")

    # Comparison against the bar Phase 12 set.
    p12 = json.loads((METRICS / "phase12_classical.json").read_text())
    best_classical = p12["models"][p12["best_model"]]
    bc_q, bc_s, bc_c = (best_classical["quality"], best_classical["latency_single_row"],
                        best_classical["cost"])
    delta = q["macro_f1"] - bc_q["macro_f1"]

    verdict = {
        "best_classical_model": p12["best_model"],
        "best_classical_macro_f1": bc_q["macro_f1"],
        "zero_shot_macro_f1": q["macro_f1"],
        "macro_f1_delta": round(delta, 4),
        "decision_rule": "LLM judged worth it only if it beats the best classical "
                         "model by >= 3 macro F1 points AND stays inside the 50 ms "
                         "p95 budget (Phase 1 §1.3)",
        "required_macro_f1_to_pass": round(bc_q["macro_f1"] + 0.03, 4),
        "passes_accuracy_bar": bool(delta >= 0.03),
        "passes_latency_budget": bool(s["within_p95_budget"]),
        "worth_it": bool(delta >= 0.03 and s["within_p95_budget"]),
        "latency_ratio_vs_classical": round(s["p95_ms"] / bc_s["p95_ms"], 1),
        "cost_ratio_vs_classical": round(
            c["usd_per_1000_predictions"] / bc_c["usd_per_1000_predictions"], 1),
        "disk_ratio_vs_classical": round(
            row["model_size_on_disk_mb"] / best_classical["model_size_on_disk_mb"], 1),
        "note": "device tiers differ (GPU vs CPU); the cost ratio already uses each "
                "tier's own hourly rate. Latency figures are NOT from the same "
                "device and are compared as a deployment fact, with both devices "
                "named.",
    }

    payload = {
        "phase": "13_zero_shot_llm",
        "splits_used": ["test (scored once)"],
        "split_policy": "test only. Nothing fitted, nothing tuned, training split "
                        "never read - not even for a class prior.",
        "model": row,
        "unparseable_outputs_full_log": parse["all_unparseable"],
        "verdict_vs_classical": verdict,
    }
    save_metrics("phase13_zeroshot", payload, device=device)

    print("\n=== VERDICT vs THE PHASE 12 BAR ===")
    print(json.dumps({k: v for k, v in verdict.items() if k != "note"}, indent=2))


if __name__ == "__main__":
    main()
