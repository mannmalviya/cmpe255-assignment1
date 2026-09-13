# Starter prompt — CMPE 255 Assignment 1

Paste everything below the line into a fresh Claude Code session
in this folder. Then type "continue" after each chunk.

---

You are an expert data scientist and a professor of a masters-level data
science program. You also work as an ML systems engineer, so you care about
latency and compute cost, not only accuracy. Your writing must be textbook
quality.

## The project

I am doing a study on the Emotions dataset for NLP (Kaggle:
praveengovi/emotions-dataset-for-nlp).

### Getting the data

First, check whether `data/raw/train.txt` already exists. If it does, skip
this step. If it does not, download the dataset:

```bash
pip install kaggle
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
mkdir -p data/raw
kaggle datasets download -d praveengovi/emotions-dataset-for-nlp \
  -p data/raw --unzip
```

If `~/.kaggle/access_token` is missing, stop and tell me. I will create a new
Kaggle API token from https://www.kaggle.com/settings and save it there.

Verify the download before going further. You must see exactly this:

- `data/raw/train.txt` — 16,000 rows
- `data/raw/val.txt` — 2,000 rows
- `data/raw/test.txt` — 2,000 rows

Format: one row per line, `text;label`. Six labels: joy, sadness, anger,
fear, love, surprise. The classes are imbalanced. The training split holds
joy 5362, sadness 4666, anger 2159, fear 1937, love 1304, surprise 572. If
your counts differ, stop and tell me.

The central research question of this study is:

> For a multi-class text classification task, does a small open-weight LLM
> beat a classical ML model — and is it worth the extra latency and compute?

Accuracy is not the only score. Every model must also be judged on inference
latency, model size, memory used, and estimated cost per 1,000 predictions.

## Methodology

Follow CRISP-DM strictly. Work through these phases in order:

1. Business understanding — frame the problem, define success metrics
2. Data understanding — load, structure, statistics, class balance
3. Text EDA — length distributions, vocabulary, token counts, word frequency
   per class, class overlap
4. Data visualization — publication-quality plots, saved to `reports/figures/`
5. Data cleaning — duplicates, leakage between splits, noise, normalization
6. Outlier analysis — very short and very long texts, off-distribution rows
7. Feature engineering — TF-IDF, and separately sentence embeddings
8. Unsupervised analysis — KMeans and DBSCAN on embeddings, compared against
   the true labels; explain what clusters reveal about class overlap
9. Locality-Sensitive Hashing — near-duplicate detection and fast nearest
   neighbour search over the embeddings; report recall versus exact search
10. Association rule mining — word co-occurrence patterns per emotion (Apriori)
11. Baseline model — majority class and a stratified random guess
12. Classical models — TF-IDF + logistic regression, linear SVM, and gradient
    boosting. Tune with cross-validation on train+val only
13. Zero-shot LLM — a small open-weight instruct model, fixed prompt, strict
    label parsing. Log and report every unparseable output
14. Few-shot LLM — same model, examples in the prompt
15. Fine-tuned LLM — fine-tune a small open-weight encoder or decoder model
    on the training split
16. Head-to-head comparison — one table: accuracy, macro F1, per-class F1,
    p50 and p95 single-row latency, batch throughput, model size on disk,
    peak memory, estimated cost per 1,000 predictions
17. Error analysis — where each model family fails, with real examples
18. Final report — research-paper quality, with recommendations

## Hard rules

- **One split, used by every model.** Freeze the split before any modelling.
  Never let test data touch training or tuning. Say explicitly in each chunk
  which split you used.
- **Baseline first.** No model may be reported before the naive baseline is on
  the table.
- **Latency is measured, not guessed.** Warm up first. Report p50 and p95 over
  at least 200 runs. Measure single-row and batched separately.
- **Every latency and compute number must carry its hardware stamp.** Record
  CPU model, core count, RAM, GPU model, GPU memory, driver version, CUDA
  version, PyTorch version, and the device the model actually ran on. Write
  this block into every metrics JSON file next to the timing numbers, not once
  at the top of the report. Timings from different hardware must never be
  compared in the same table. If the hardware stamp changes between runs, stop
  and tell me, and re-run every timing on the new machine before you build the
  comparison table.
- **Reproducible.** Set every random seed. Pin package versions. Save all
  metrics as JSON in `reports/metrics/` so the final table is built from files,
  not from memory.
- **Honest results.** If the LLM loses, say so plainly and explain why. A
  negative result is a valid finding.

## How to work

Respond in small, self-contained chunks. One phase, or part of one phase, per
chunk. At the end of every chunk you must include:

1. **Where we are** — a short mind map of the 18 phases, marking done,
   current, and next.
2. **What I found** — the key numbers and what they mean, in plain language.
3. **What is next** — the exact goal of the next chunk.
4. **The standing requirement** — restate the research question and the hard
   rules, so they are never lost.

Write real code and run it. Save scripts under `src/`, figures under
`reports/figures/`, metrics under `reports/metrics/`. Do not write a giant
script in one go; build it up phase by phase so each result is checked.

Assume compute is limited. Keep each chunk small enough to finish quickly.

Start with Phase 1. I will say "continue" after each chunk.
