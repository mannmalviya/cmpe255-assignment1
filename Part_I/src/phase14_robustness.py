"""Phase 14b — Robustness check for the few-shot result. VAL ONLY.

SPLIT USED: val (2,000 rows) only. Test is NOT scored again and NO choice is
re-made. The Phase 14 test result stands as reported; this script asks whether
it can be trusted.

Two questions the Phase 14 design left open, found after the result:
  1. k = 0 (zero-shot) was not in the val grid, so val could not have chosen it.
     Would it have?
  2. The test score rests on ONE seeded draw of 12 examples. How much does the
     val score move under a different draw of the same size?

If the seed-to-seed spread is large, "few-shot hurts" is a statement about one
draw, and must be reported as such.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

import llm_common as L
from config import DATA_PROCESSED, LABELS, METRICS, ensure_dirs, set_all_seeds
from hardware import save_metrics
from phase14_fewshot import candidate_pool, score_split, select_shots

EXTRA_SEEDS = [7, 123, 2024]


def main() -> None:
    set_all_seeds()
    ensure_dirs()
    device = "cuda:0"
    val = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    pool = candidate_pool()
    runner = L.LLMRunner(device=device).load()

    runs = []

    def record(name, k, seed, shots):
        runner.shots = [(s["text"], s["label"]) for s in shots]
        res = score_split(runner, val)
        q = res["quality"]
        pred = pd.Series(res["labels"]).value_counts().to_dict()
        row = {"config": name, "k_per_class": k, "seed": seed,
               "val_macro_f1": q["macro_f1"], "val_accuracy": q["accuracy"],
               "val_n_unparseable": res["parse"]["n_unparseable_strict"],
               "per_class_f1": {lab: q["per_class"][lab]["f1"] for lab in LABELS},
               "predicted_counts": {lab: int(pred.get(lab, 0)) for lab in LABELS},
               "shots": shots}
        runs.append(row)
        print(f"  {name:<22} macro F1 {q['macro_f1']:.4f}  acc {q['accuracy']:.4f}  "
              f"unparseable {row['val_n_unparseable']:>3}  "
              f"fear F1 {row['per_class_f1']['fear']:.3f}  "
              f"surprise F1 {row['per_class_f1']['surprise']:.3f}", flush=True)

    print("=== VAL ONLY. Test is not scored. ===", flush=True)
    record("zero-shot (k=0)", 0, None, [])

    sweep = json.loads((DATA_PROCESSED / "phase14_val_sweep.json").read_text())
    k2 = next(r for r in sweep if r["k_per_class"] == 2)
    runs.append({"config": "k=2 seed 42 (Phase 14)", "k_per_class": 2, "seed": 42,
                 "val_macro_f1": k2["val_macro_f1"],
                 "val_accuracy": k2["val_accuracy"],
                 "val_n_unparseable": k2["val_n_unparseable"],
                 "shots": k2["shots"], "from_checkpoint": True})
    print(f"  {'k=2 seed 42 (Phase 14)':<22} macro F1 {k2['val_macro_f1']:.4f}  "
          f"(from checkpoint)", flush=True)

    for seed in EXTRA_SEEDS:
        record(f"k=2 seed {seed}", 2, seed, select_shots(pool, 2, seed=seed))

    k2_f1 = [r["val_macro_f1"] for r in runs if r["k_per_class"] == 2]
    zs_val = runs[0]["val_macro_f1"]
    summary = {
        "zero_shot_val_macro_f1": zs_val,
        "k2_val_macro_f1_by_seed": {str(r["seed"]): r["val_macro_f1"]
                                    for r in runs if r["k_per_class"] == 2},
        "k2_mean": round(float(np.mean(k2_f1)), 4),
        "k2_std": round(float(np.std(k2_f1, ddof=1)), 4),
        "k2_min": round(min(k2_f1), 4), "k2_max": round(max(k2_f1), 4),
        "k2_range": round(max(k2_f1) - min(k2_f1), 4),
        "seeds_where_k2_beats_zero_shot": int(sum(f > zs_val for f in k2_f1)),
        "n_seeds": len(k2_f1),
        "would_val_have_chosen_k0": bool(zs_val > max(
            r["val_macro_f1"] for r in sweep)),
    }
    save_metrics("phase14_robustness", {
        "phase": "14b_fewshot_robustness",
        "splits_used": ["val only"],
        "split_policy": "val only. Test not scored again; no choice re-made. The "
                        "Phase 14 test result stands as reported.",
        "questions": ["was k=0 better on val than any k in the grid?",
                      "how much does a different draw of 12 examples move the score?"],
        "runs": runs, "summary": summary}, device=device)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
