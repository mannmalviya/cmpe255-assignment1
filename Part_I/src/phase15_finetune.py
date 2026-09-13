"""Phase 15 — Fine-tuned transformer encoder.

SPLITS USED:
  Training       : train_clean (15,923 rows).
  Early stopping : val (2,000 rows) - the epoch with the best val macro F1 is kept.
  Scoring        : test (2,000 rows), ONCE, with that checkpoint.

WHY THIS MODEL. Phases 13 and 14 lost for one reason: the model never learned this
dataset's label conventions from enough examples. Fine-tuning removes exactly that
cause - the model learns from all 15,923 training rows, as the linear SVM did.

`distilbert-base-uncased` (66 M parameters) is the chosen encoder:
  - an encoder with a classification head outputs one of six classes directly,
    so there is no generation and no unparseable output;
  - it trains in minutes on the 8 GB RTX 4060;
  - it is small enough that CPU serving is a realistic option, so latency is
    measured on BOTH devices, each in its own metrics file with its own stamp.

DESIGN FIXED BEFORE TRAINING:
  max_len 128 (Phase 7: longest document is 87 WordPiece tokens) | batch 32 |
  AdamW lr 5e-5, weight decay 0.01 | linear schedule, 10% warm-up | max 4 epochs,
  keep best val macro F1 | class-weighted cross-entropy with 'balanced' weights
  from train_clean (Phase 12: balanced weighting was worth +2.8 macro F1) |
  bf16 autocast on GPU | seed 42.

Usage:
  python phase15_finetune.py train      # GPU: train, early-stop, score, GPU latency
  python phase15_finetune.py eval-cpu   # fresh process: CPU latency on saved model
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import evaluate as ev
from config import (DATA_PROCESSED, LABEL_TO_ID, ID_TO_LABEL, LABELS, METRICS,
                    SEED, ensure_dirs, set_all_seeds)
from hardware import save_metrics

MODEL_ID = "distilbert-base-uncased"
OUT_DIR = DATA_PROCESSED / "models" / "distilbert_finetuned"
MAX_LEN = 128
BATCH = 32
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.10
MAX_EPOCHS = 4


# ------------------------------------------------------------------ data

def load_splits():
    tr = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
    va = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    te = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    assert (len(tr), len(va), len(te)) == (15923, 2000, 2000)
    return tr, va, te


def make_loader(tok, df, shuffle, seed=SEED):
    """Dynamic padding: each batch pads only to its own longest row."""
    enc = tok(df["text"].tolist(), truncation=True, max_length=MAX_LEN)
    labels = df["label"].map(LABEL_TO_ID).to_numpy()
    items = [{"input_ids": enc["input_ids"][i],
              "attention_mask": enc["attention_mask"][i],
              "labels": int(labels[i])} for i in range(len(df))]

    def collate(batch):
        out = tok.pad([{k: b[k] for k in ("input_ids", "attention_mask")}
                       for b in batch], return_tensors="pt")
        out["labels"] = torch.tensor([b["labels"] for b in batch])
        return out

    g = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=BATCH, shuffle=shuffle, collate_fn=collate,
                      generator=g if shuffle else None)


@torch.inference_mode()
def predict_loader(model, loader, device, amp):
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(**batch).logits
        preds.extend(logits.argmax(-1).tolist())
    return np.array([ID_TO_LABEL[p] for p in preds])


def make_predictors(model, tok, device, amp):
    """Raw text -> label string, end to end, for the latency harness."""
    @torch.inference_mode()
    def predict_batch(texts):
        enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(**enc).logits
        out = [ID_TO_LABEL[i] for i in logits.argmax(-1).tolist()]
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()   # time the work, not the kernel queue
        return out

    return (lambda t: predict_batch([t])[0]), predict_batch


def disk_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*")
                     if f.is_file() and not f.is_symlink()) / 1024**2, 1)


# ------------------------------------------------------------------ train

def fit(seed: int, out_dir: Path, log=True):
    """Train with one seed; keep the best-val-macro-F1 epoch in out_dir."""
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    set_all_seeds(seed)
    ensure_dirs()
    if not torch.cuda.is_available():
        raise RuntimeError("training mode requires the GPU")
    device = "cuda:0"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    torch.cuda.reset_peak_memory_stats()

    tr, va, te = load_splits()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS), id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID).to(device)

    train_dl = make_loader(tok, tr, shuffle=True, seed=seed)
    val_dl = make_loader(tok, va, shuffle=False)

    counts = tr["label"].value_counts()
    weights = torch.tensor([len(tr) / (len(LABELS) * counts[lab]) for lab in LABELS],
                           dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total = len(train_dl) * MAX_EPOCHS
    sched = get_linear_schedule_with_warmup(optim, int(WARMUP_FRAC * total), total)

    history, best_f1, best_epoch = [], -1.0, None
    t_train = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0, running = time.perf_counter(), 0.0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()
        epoch_s = time.perf_counter() - t0

        val_pred = predict_loader(model, val_dl, device, amp=True)
        vq = ev.quality_metrics(va["label"].to_numpy(), val_pred)
        row = {"epoch": epoch, "train_loss": round(running / len(train_dl), 4),
               "val_macro_f1": vq["macro_f1"], "val_accuracy": vq["accuracy"],
               "epoch_seconds": round(epoch_s, 1)}
        history.append(row)
        improved = vq["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch = vq["macro_f1"], epoch
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
        print(f"  epoch {epoch}: loss {row['train_loss']:.4f}  val macro F1 "
              f"{vq['macro_f1']:.4f}  acc {vq['accuracy']:.4f}  ({epoch_s:.0f}s)"
              f"{'  <- best, saved' if improved else ''}", flush=True)
    train_s = time.perf_counter() - t_train
    train_peak = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return tok, te, best_epoch, best_f1, history, train_s, train_peak


def train():
    from transformers import AutoModelForSequenceClassification
    device = "cuda:0"
    tok, te, best_epoch, best_f1, history, train_s, train_peak = fit(SEED, OUT_DIR)

    # ------------------------------------------ test ONCE, best checkpoint
    print(f"\n  best epoch {best_epoch} (val macro F1 {best_f1:.4f}); "
          "scoring test once", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR).to(device)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    predict_one, predict_batch = make_predictors(model, tok, device, amp=True)
    texts, y_true = te["text"].tolist(), te["label"].to_numpy()
    y_pred = np.concatenate([predict_batch(texts[i:i + 128])
                             for i in range(0, len(texts), 128)])

    row = ev.evaluate_model(
        name="distilbert_finetuned", family="llm_fine_tuned",
        y_true=y_true, y_pred=y_pred,
        predict_one=predict_one, predict_batch=predict_batch,
        texts=texts, device=device, model_size_mb=disk_mb(OUT_DIR),
        peak_memory_mb=0.0,
        notes={"model_id": MODEL_ID,
               "parameters_millions": round(sum(p.numel() for p in model.parameters())
                                            / 1e6, 2),
               "inference_precision": "bf16 autocast",
               "max_len": MAX_LEN, "batch": BATCH, "lr": LR,
               "weight_decay": WEIGHT_DECAY, "warmup_frac": WARMUP_FRAC,
               "max_epochs": MAX_EPOCHS, "class_weights": "balanced, from train_clean",
               "trained_on": "train_clean (15,923)", "early_stopped_on": "val",
               "scored_on": "test (2,000), once",
               "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
               "history": history, "training_seconds": round(train_s, 1),
               "training_peak_gpu_mb": train_peak})
    row["peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    row["peak_memory_kind"] = "GPU peak allocated during inference and latency tests"

    # test predictions saved for Phase 17 error analysis and CPU parity check
    pd.DataFrame({"row_id": te["row_id"], "text": te["text"], "label": y_true,
                  "pred": y_pred}).to_parquet(OUT_DIR / "test_predictions_gpu.parquet",
                                              index=False)

    finish(row, device, "phase15_finetune_gpu")


# --------------------------------------------------------------- CPU eval

def eval_cpu():
    """Fresh process, so peak RSS reflects the CPU deployment, not the training run."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    set_all_seeds()
    device = "cpu"
    _, _, te = load_splits()
    tok = AutoTokenizer.from_pretrained(OUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR).eval()
    predict_one, predict_batch = make_predictors(model, tok, device, amp=False)
    texts, y_true = te["text"].tolist(), te["label"].to_numpy()
    y_pred = np.concatenate([predict_batch(texts[i:i + 128])
                             for i in range(0, len(texts), 128)])

    gpu = pd.read_parquet(OUT_DIR / "test_predictions_gpu.parquet")
    agree = float((gpu["pred"].to_numpy() == y_pred).mean())

    row = ev.evaluate_model(
        name="distilbert_finetuned_cpu", family="llm_fine_tuned",
        y_true=y_true, y_pred=y_pred,
        predict_one=predict_one, predict_batch=predict_batch,
        texts=texts, device=device, model_size_mb=disk_mb(OUT_DIR),
        peak_memory_mb=ev.peak_rss_mb(),
        notes={"model_id": MODEL_ID, "inference_precision": "fp32",
               "torch_threads": torch.get_num_threads(),
               "same_checkpoint_as": "phase15_finetune_gpu",
               "prediction_agreement_with_gpu_bf16": round(agree, 4)})
    row["peak_memory_kind"] = "process peak RSS (fresh CPU-only process)"
    finish(row, device, "phase15_finetune_cpu")


def finish(row, device, name):
    p12 = json.loads((METRICS / "phase12_classical.json").read_text())
    svm = p12["models"][p12["best_model"]]
    q, s, c = row["quality"], row["latency_single_row"], row["cost"]
    delta = q["macro_f1"] - svm["quality"]["macro_f1"]
    verdict = {
        "best_classical_macro_f1": svm["quality"]["macro_f1"],
        "fine_tuned_macro_f1": q["macro_f1"],
        "delta_vs_best_classical": round(delta, 4),
        "required_macro_f1_to_pass": round(svm["quality"]["macro_f1"] + 0.03, 4),
        "passes_accuracy_bar": bool(delta >= 0.03),
        "passes_latency_budget": bool(s["within_p95_budget"]),
        "worth_it": bool(delta >= 0.03 and s["within_p95_budget"]),
        "latency_ratio_vs_classical_p95": round(
            s["p95_ms"] / svm["latency_single_row"]["p95_ms"], 1),
        "cost_ratio_vs_classical": round(
            c["usd_per_1000_predictions"] / svm["cost"]["usd_per_1000_predictions"], 1),
        "disk_ratio_vs_classical": round(
            row["model_size_on_disk_mb"] / svm["model_size_on_disk_mb"], 1),
    }
    save_metrics(name, {
        "phase": "15_fine_tuned",
        "splits_used": ["train_clean (train)", "val (early stopping)",
                        "test (scored once)"],
        "split_policy": "trained on train_clean; best epoch chosen on val; test "
                        "scored once with that checkpoint.",
        "model": row, "verdict": verdict}, device=device)

    print(f"\n=== {name} ===")
    print(f"  macro F1 {q['macro_f1']:.4f}  accuracy {q['accuracy']:.4f}  "
          f"meets 0.85: {q['meets_target_macro_f1']}")
    print("  per-class F1: " + "  ".join(
        f"{lab}={q['per_class'][lab]['f1']:.3f}" for lab in LABELS))
    print(f"  p50 {s['p50_ms']:.2f} ms  p95 {s['p95_ms']:.2f} ms  "
          f"budget ok: {s['within_p95_budget']}")
    b = row["throughput_batched"]
    print(f"  throughput {b['best_rows_per_second']:,.0f} rows/s @ batch "
          f"{b['best_batch_size']}  cost ${c['usd_per_1000_predictions']:.4g}/1k  "
          f"disk {row['model_size_on_disk_mb']} MB  peak mem {row['peak_memory_mb']} MB")
    if "prediction_agreement_with_gpu_bf16" in row["notes"]:
        print(f"  CPU fp32 vs GPU bf16 prediction agreement: "
              f"{row['notes']['prediction_agreement_with_gpu_bf16']:.4f}")
    print(json.dumps(verdict, indent=2))


def seed_check():
    """Seed variance of the fine-tuning PROCEDURE.

    Each extra seed follows the identical protocol (train on train_clean, best
    epoch on val) and is scored on test. NO CHOICE is made from these runs: the
    reported model remains seed 42. They exist only to show how far the headline
    result would move under a different random initialisation and data order,
    which matters because the margin over the decision threshold is small.
    """
    from transformers import AutoModelForSequenceClassification
    device = "cuda:0"
    out_path = METRICS / "phase15_seed_check.json"
    runs = json.loads(out_path.read_text())["runs"] if out_path.exists() else []
    done = {r["seed"] for r in runs}
    for seed in [int(x) for x in sys.argv[2:]]:
        if seed in done:
            continue
        d = DATA_PROCESSED / "models" / f"distilbert_seed{seed}"
        tok, te, best_epoch, best_f1, history, _, _ = fit(seed, d)
        model = AutoModelForSequenceClassification.from_pretrained(d).to(device).eval()
        _, predict_batch = make_predictors(model, tok, device, amp=True)
        texts = te["text"].tolist()
        y_pred = np.concatenate([predict_batch(texts[i:i + 128])
                                 for i in range(0, len(texts), 128)])
        q = ev.quality_metrics(te["label"].to_numpy(), y_pred)
        pd.DataFrame({"row_id": te["row_id"], "pred": y_pred}).to_parquet(
            d / "test_predictions_gpu.parquet", index=False)
        runs.append({"seed": seed, "best_epoch": best_epoch,
                     "best_val_macro_f1": best_f1,
                     "test_macro_f1": q["macro_f1"], "test_accuracy": q["accuracy"],
                     "per_class_f1": {l: q["per_class"][l]["f1"] for l in LABELS},
                     "history": history})
        print(f"  seed {seed}: best epoch {best_epoch}, val {best_f1:.4f}, "
              f"TEST macro F1 {q['macro_f1']:.4f}", flush=True)
        save_metrics("phase15_seed_check", {
            "phase": "15b_seed_variance",
            "splits_used": ["train_clean (train)", "val (early stopping)",
                            "test (scored, no selection)"],
            "policy": "identical protocol per seed; no choice made from these runs; "
                      "the reported model remains seed 42",
            "runs": runs}, device=device)


if __name__ == "__main__":
    {"train": train, "eval-cpu": eval_cpu, "seed-check": seed_check}[sys.argv[1]]()
