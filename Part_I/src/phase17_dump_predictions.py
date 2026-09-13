"""Phase 17a — Collect per-row test predictions for every model into one file.

SPLIT USED: test. NO model is fitted, tuned or re-selected. Every configuration is
the one already fixed and reported in Phases 12-15.

Why this script exists: Phases 13 and 14 saved confusion matrices and the full log
of unparseable outputs, but not the per-row predictions error analysis needs. They
are regenerated here with the identical saved configuration (same model, prompt,
shots, greedy decoding, batch size 32, row order). Greedy decoding is
deterministic, so the regenerated macro F1 must EQUAL the reported one; the script
asserts this and stops if it does not. The classical models are re-predicted from
their saved, frozen .joblib files, and checked the same way.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd

import evaluate as ev
from config import DATA_PROCESSED, METRICS, ensure_dirs, set_all_seeds

OUT = DATA_PROCESSED / "test_predictions_all.parquet"


def check(name, y, pred, reported):
    got = ev.quality_metrics(y, pred)["macro_f1"]
    status = "OK" if abs(got - reported) < 1e-9 else "MISMATCH"
    print(f"  {name:<22} regenerated macro F1 {got:.4f}  reported {reported:.4f}  {status}",
          flush=True)
    if status != "OK":
        raise RuntimeError(f"{name}: regenerated predictions do not reproduce the "
                           "reported score; refusing to analyse them.")


def main():
    set_all_seeds()
    ensure_dirs()
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    y = test["label"].to_numpy()
    texts = test["text"].tolist()
    out = test[["row_id", "text", "label"]].copy()
    ckpt = pd.read_parquet(OUT) if OUT.exists() else None

    p12 = json.loads((METRICS / "phase12_classical.json").read_text())
    for key in ("linear_svm", "logreg", "gradient_boosting"):
        m = joblib.load(DATA_PROCESSED / "models" / f"{key}.joblib")
        out[key] = np.asarray(m.predict(texts))
        check(key, y, out[key].to_numpy(), p12["models"][key]["quality"]["macro_f1"])

    ft = pd.read_parquet(DATA_PROCESSED / "models" / "distilbert_finetuned" /
                         "test_predictions_gpu.parquet")
    assert (ft["row_id"].to_numpy() == out["row_id"].to_numpy()).all()
    out["distilbert"] = ft["pred"].to_numpy()
    check("distilbert", y, out["distilbert"].to_numpy(),
          json.loads((METRICS / "phase15_finetune_gpu.json").read_text())
          ["model"]["quality"]["macro_f1"])

    if ckpt is not None and {"qwen_zero_shot", "qwen_few_shot"} <= set(ckpt.columns):
        out["qwen_zero_shot"] = ckpt["qwen_zero_shot"]
        out["qwen_zero_shot_raw"] = ckpt["qwen_zero_shot_raw"]
        out["qwen_few_shot"] = ckpt["qwen_few_shot"]
        out["qwen_few_shot_raw"] = ckpt["qwen_few_shot_raw"]
        print("  qwen predictions: loaded from checkpoint", flush=True)
    else:
        import llm_common as L
        runner = L.LLMRunner(device="cuda:0").load()

        runner.shots = []
        labels, raws, _ = L.run_full_split(runner, texts, batch_size=32, log_every=100)
        out["qwen_zero_shot"] = labels
        out["qwen_zero_shot_raw"] = [r["raw"] for r in raws]

        p14 = json.loads((METRICS / "phase14_fewshot.json").read_text())
        runner.shots = [(s["text"], s["label"]) for s in p14["model"]["notes"]["shots"]]
        labels, raws, _ = L.run_full_split(runner, texts, batch_size=32, log_every=100)
        out["qwen_few_shot"] = labels
        out["qwen_few_shot_raw"] = [r["raw"] for r in raws]

    check("qwen_zero_shot", y, out["qwen_zero_shot"].to_numpy(),
          json.loads((METRICS / "phase13_zeroshot.json").read_text())
          ["model"]["quality"]["macro_f1"])
    check("qwen_few_shot", y, out["qwen_few_shot"].to_numpy(),
          json.loads((METRICS / "phase14_fewshot.json").read_text())
          ["model"]["quality"]["macro_f1"])

    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows, {out.shape[1]} columns)")


if __name__ == "__main__":
    main()
