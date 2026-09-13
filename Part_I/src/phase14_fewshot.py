"""Phase 14 — Few-shot LLM classification.

SPLITS USED:
  Shots     : drawn from train_clean only.
  Selection : the number of shots per class is chosen on VAL (2,000 rows).
  Scoring   : the chosen configuration is scored ONCE on test (2,000 rows).
Test never influences any choice. Trying several shot counts on test and keeping
the best would be tuning on test.

ONE CHANGE FROM PHASE 13: labelled examples are placed in the prompt as prior
user/assistant turns. Model, system prompt, decoding, and strict parser are
imported unchanged from src/llm_common.py, so any difference from Phase 13 is
attributable to the examples alone.

DESIGN FIXED BEFORE ANY RESULT WAS SEEN:
  - candidates: train_clean rows with NO Phase 6 outlier flag (not short, not
    long, not off-template, not rare-vocabulary, not cue-free) and <= 25 words.
    Shots should show the typical convention, not an edge case, and short shots
    keep the prompt - and so the latency cost - bounded.
  - selection: stratified random, seed 42, k shots PER CLASS, so every class and
    in particular the joy/love boundary is shown.
  - order: round-robin across classes in a seeded shuffled class order, so the
    final example is not always the same class (recency bias).
  - k grid on val: {1, 2, 4} per class = {6, 12, 24} shots. Ties go to the smaller
    k, because more shots cost latency.
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
from config import (DATA_PROCESSED, LABELS, METRICS, SEED, ensure_dirs,
                    set_all_seeds)
from hardware import save_metrics

K_GRID = [1, 2, 4]
MAX_SHOT_WORDS = 25
BATCH_SIZE = 32


def model_disk_mb(model_id: str) -> float:
    d = (Path.home() / ".cache" / "huggingface" / "hub" /
         ("models--" + model_id.replace("/", "--")))
    return round(sum(f.stat().st_size for f in d.rglob("*")
                     if f.is_file() and not f.is_symlink()) / 1024**2, 1)


def candidate_pool() -> pd.DataFrame:
    train = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
    flags = pd.read_parquet(DATA_PROCESSED / "train_outlier_flags.parquet")
    df = train.merge(flags[["row_id", "n_flags"]], on="row_id", how="left")
    df = df[(df["n_flags"] == 0) &
            (df["text"].str.split().str.len() <= MAX_SHOT_WORDS)]
    return df.reset_index(drop=True)


def select_shots(pool: pd.DataFrame, k: int, seed: int = SEED) -> list:
    rng = np.random.default_rng(seed)
    per_class = {}
    for lab in LABELS:
        sub = pool[pool["label"] == lab]
        idx = rng.choice(len(sub), size=k, replace=False)
        per_class[lab] = sub.iloc[idx][["row_id", "text", "label"]].to_dict("records")
    class_order = list(rng.permutation(LABELS))
    shots = []
    for i in range(k):
        for lab in class_order:
            shots.append(per_class[lab][i])
    return shots


def prompt_tokens(runner, text: str) -> int:
    rendered = runner._render([text])[0]
    return len(runner.tokenizer(rendered, add_special_tokens=False)["input_ids"])


def score_split(runner, df: pd.DataFrame) -> dict:
    texts, y_true = df["text"].tolist(), df["label"].to_numpy()
    bs = BATCH_SIZE
    while True:
        try:
            labels, raws, secs = L.run_full_split(runner, texts, batch_size=bs,
                                                  log_every=20)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs //= 2
            print(f"    OOM - retrying with batch {bs}", flush=True)
            if bs < 1:
                raise
    q = ev.quality_metrics(y_true, np.array(labels))
    return {"quality": q, "parse": L.parse_report(raws), "raws": raws,
            "labels": labels, "seconds": secs, "batch_size": bs}


def main() -> None:
    set_all_seeds()
    ensure_dirs()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 14 must run on the GPU, as Phase 13 did.")
    device = "cuda:0"
    torch.cuda.reset_peak_memory_stats()

    val = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    pool = candidate_pool()
    test_texts = set(test["text"])
    print(f"shot candidate pool: {len(pool):,} train_clean rows "
          f"(no outlier flag, <= {MAX_SHOT_WORDS} words)", flush=True)

    runner = L.LLMRunner(device=device).load()
    print(f"loaded {L.MODEL_ID}", flush=True)

    # ---------------------------------------------- choose k on VAL only
    print("\n=== SELECTING SHOTS PER CLASS ON VAL (test not touched) ===", flush=True)
    # Each val result is checkpointed as it completes. The first run of this phase
    # finished the whole sweep and then lost it to an OOM in the latency harness,
    # because results were only written at the very end.
    ckpt_path = DATA_PROCESSED / "phase14_val_sweep.json"
    done = {r["k_per_class"]: r for r in
            (json.loads(ckpt_path.read_text()) if ckpt_path.exists() else [])}
    sweep = []
    for k in K_GRID:
        if k in done:
            sweep.append(done[k])
            print(f"  k={k}: loaded checkpoint, val macro F1 "
                  f"{done[k]['val_macro_f1']:.4f}", flush=True)
            continue
        shots = select_shots(pool, k)
        assert not any(s["text"] in test_texts for s in shots), "shot leaks into test"
        runner.shots = [(s["text"], s["label"]) for s in shots]
        ptoks = prompt_tokens(runner, val["text"].iloc[0])
        print(f"  k={k} per class ({len(shots)} shots, ~{ptoks} prompt tokens)",
              flush=True)
        res = score_split(runner, val)
        q = res["quality"]
        sweep.append({
            "k_per_class": k, "n_shots": len(shots), "prompt_tokens_example": ptoks,
            "val_macro_f1": q["macro_f1"], "val_accuracy": q["accuracy"],
            "val_strict_parse_rate": res["parse"]["strict_parse_rate"],
            "val_n_unparseable": res["parse"]["n_unparseable_strict"],
            "val_rows_per_second": round(len(val) / res["seconds"], 2),
            "shots": shots,
        })
        ckpt_path.write_text(json.dumps(sweep, indent=2, default=str))
        print(f"    val macro F1 {q['macro_f1']:.4f}  acc {q['accuracy']:.4f}  "
              f"unparseable {res['parse']['n_unparseable_strict']}  "
              f"{len(val)/res['seconds']:.1f} rows/s", flush=True)

    best_f1 = max(r["val_macro_f1"] for r in sweep)
    chosen = min((r for r in sweep if r["val_macro_f1"] == best_f1),
                 key=lambda r: r["k_per_class"])
    k = chosen["k_per_class"]
    print(f"\n  chosen on val: k={k} per class ({chosen['n_shots']} shots)",
          flush=True)

    # --------------------------------------------- score ONCE on test
    runner.shots = [(s["text"], s["label"]) for s in chosen["shots"]]
    runner.unparseable = []
    print(f"\n=== SCORING test ONCE with k={k} ===", flush=True)
    res = score_split(runner, test)
    y_true = test["label"].to_numpy()
    y_pred = np.array(res["labels"])
    y_pred_len = np.array([r["lenient"] or "UNPARSEABLE" for r in res["raws"]])

    print("\n=== LATENCY (GPU, full protocol) ===", flush=True)
    row = ev.evaluate_model(
        name=f"few_shot_qwen2.5_1.5b_k{k}", family="llm_few_shot",
        y_true=y_true, y_pred=y_pred,
        predict_one=runner.predict_one, predict_batch=runner.predict_batch,
        texts=test["text"].tolist(), device=device,
        model_size_mb=model_disk_mb(L.MODEL_ID),
        peak_memory_mb=0.0,
        notes={
            "model_id": L.MODEL_ID,
            "parameters_billions": runner.parameters_billions(),
            "dtype": "bfloat16", "decoding": "greedy, deterministic",
            "max_new_tokens": L.MAX_NEW_TOKENS,
            "system_prompt": L.SYSTEM_PROMPT,
            "shots_per_class": k, "n_shots": chosen["n_shots"],
            "shots": chosen["shots"],
            "prompt_tokens_example": chosen["prompt_tokens_example"],
            "shot_source": "train_clean, no Phase 6 outlier flag, "
                           f"<= {MAX_SHOT_WORDS} words, stratified, seed {SEED}",
            "k_selected_on": "val",
            "full_split_rows_per_second": round(len(test) / res["seconds"], 2),
            "batch_size_used_for_split": res["batch_size"],
        })
    row["notes"]["gpu_memory"] = runner.gpu_memory_mb()
    row["peak_memory_mb"] = row["notes"]["gpu_memory"]["peak_allocated_mb"]
    row["peak_memory_kind"] = "GPU peak allocated (torch.cuda.max_memory_allocated)"
    row["parsing"] = {k_: v for k_, v in res["parse"].items() if k_ != "all_unparseable"}
    row["quality_if_lenient_parsing"] = ev.quality_metrics(y_true, y_pred_len)

    # ---------------------------------------------------- comparisons
    p12 = json.loads((METRICS / "phase12_classical.json").read_text())
    p13 = json.loads((METRICS / "phase13_zeroshot.json").read_text())
    svm = p12["models"][p12["best_model"]]
    zs = p13["model"]
    q, s, c = row["quality"], row["latency_single_row"], row["cost"]
    delta = q["macro_f1"] - svm["quality"]["macro_f1"]
    verdict = {
        "best_classical_macro_f1": svm["quality"]["macro_f1"],
        "zero_shot_macro_f1": zs["quality"]["macro_f1"],
        "few_shot_macro_f1": q["macro_f1"],
        "gain_over_zero_shot": round(q["macro_f1"] - zs["quality"]["macro_f1"], 4),
        "delta_vs_best_classical": round(delta, 4),
        "required_macro_f1_to_pass": round(svm["quality"]["macro_f1"] + 0.03, 4),
        "passes_accuracy_bar": bool(delta >= 0.03),
        "passes_latency_budget": bool(s["within_p95_budget"]),
        "worth_it": bool(delta >= 0.03 and s["within_p95_budget"]),
        "p95_ms_few_shot": s["p95_ms"], "p95_ms_zero_shot": zs["latency_single_row"]["p95_ms"],
        "p95_ms_classical": svm["latency_single_row"]["p95_ms"],
        "latency_ratio_vs_zero_shot": round(
            s["p95_ms"] / zs["latency_single_row"]["p95_ms"], 2),
        "latency_ratio_vs_classical": round(
            s["p95_ms"] / svm["latency_single_row"]["p95_ms"], 1),
        "cost_ratio_vs_classical": round(
            c["usd_per_1000_predictions"] / svm["cost"]["usd_per_1000_predictions"], 1),
    }

    payload = {
        "phase": "14_few_shot_llm",
        "splits_used": ["train_clean (shot source)", "val (choice of k)",
                        "test (scored once)"],
        "split_policy": "shots from train_clean; k chosen on val; test scored once "
                        "with the chosen k. No test row influenced any choice.",
        "one_change_from_phase_13": "labelled examples in the prompt; model, system "
                                    "prompt, decoding and parser unchanged",
        "val_sweep": sweep,
        "chosen_k_per_class": k,
        "model": row,
        "unparseable_outputs_full_log": res["parse"]["all_unparseable"],
        "verdict": verdict,
    }
    save_metrics("phase14_fewshot", payload, device=device)

    ql = row["quality_if_lenient_parsing"]
    print(f"\n  STRICT  accuracy {q['accuracy']:.4f}  macro F1 {q['macro_f1']:.4f}")
    print(f"  LENIENT accuracy {ql['accuracy']:.4f}  macro F1 {ql['macro_f1']:.4f}")
    print(f"  unparseable {res['parse']['n_unparseable_strict']}")
    print("  per-class F1: " + "  ".join(
        f"{lab}={q['per_class'][lab]['f1']:.3f}" for lab in LABELS))
    print(f"  p50 {s['p50_ms']:.1f} ms  p95 {s['p95_ms']:.1f} ms  "
          f"throughput {row['throughput_batched']['best_rows_per_second']:.1f} rows/s  "
          f"cost ${c['usd_per_1000_predictions']:.4g}/1k  gpu {row['notes']['gpu_memory']}")
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
