# Phase 1 — Business Understanding

CRISP-DM Phase 1 of 18. Emotions Dataset for NLP (Kaggle: `praveengovi/emotions-dataset-for-nlp`).

---

## 1.1 The research question

> **For a multi-class text classification task, does a small open-weight LLM beat a
> classical ML model — and is it worth the extra latency and compute?**

This question has two halves, and they are deliberately in tension.

The first half is a **quality** question: does a decoder- or encoder-based language
model, pre-trained on internet-scale corpora, extract more signal from a short
English utterance than a bag-of-ngrams model fitted on 16,000 labelled examples?
The prior literature says usually yes, but the margin narrows sharply when the task
is narrow, the label set is small and closed, and in-domain labelled data is
plentiful. All three conditions hold here.

The second half is a **systems** question: what does that margin cost? A TF-IDF +
logistic regression model is a sparse matrix multiply — microseconds on one CPU
core, a few megabytes on disk. A 7B-parameter instruct model is tens of gigabytes
of weights and a token-by-token autoregressive decode on an accelerator. If the LLM
buys two points of macro F1 for a 1000x increase in cost-per-prediction, that is not
a win in any deployment that has a budget. A study that reports only accuracy cannot
answer the question that was asked.

We therefore treat **accuracy and cost as co-equal dependent variables**, and the
deliverable is not "the best model" but a defensible frontier: for each operating
budget, which model family should be chosen.

---

## 1.2 The business framing

We assume a realistic host application: a **product-feedback and support triage
system**. Users write short free-text messages. The system must tag each message
with the dominant emotion so that downstream routing can act on it — angry and
fearful messages escalate to a human within minutes, sad messages enter a
retention workflow, joyful messages are candidates for testimonial requests.

This framing is not decoration. It fixes four things that would otherwise be
arbitrary:

**It fixes the unit of value.** Value accrues per correctly routed message, not per
correctly predicted label in the aggregate. A message is routed once, so the
prediction is a single hard label, not a ranked list. Top-k accuracy is therefore
not a reported metric.

**It fixes the cost of each error type.** The classes are not interchangeable.
Missing an `anger` or `fear` message is an escalation failure with direct churn
cost. Missing a `surprise` message is nearly free. But `surprise` is also the
rarest class (572 of 16,000 training rows, 3.6%), and a model that ignores it
entirely loses only 3.6 points of plain accuracy. This is exactly the failure mode
that plain accuracy hides, which drives the metric choice in §1.3.

**It fixes the latency requirement.** Triage is inline with message submission.
The user waits. We set a **p95 single-row budget of 50 ms** for the classification
step. This is not a hard gate — a model that exceeds it is reported, not discarded —
but it is the line against which "is it worth it" is judged.

**It fixes the deployment scale.** We assume 1,000,000 messages per month. This
makes cost-per-1,000-predictions the natural unit, and it makes a difference of
$0.10 per 1,000 a difference of $100/month — small. A difference of $10 per 1,000
is $10,000/month — decisive. The comparison table must be read at this scale.

---

## 1.3 Success metrics

### Primary quality metric: macro F1

**Macro F1** is the unweighted mean of the per-class F1 scores. "Unweighted" is the
whole point: `surprise` (3.6% of train) contributes exactly as much to the score as
`joy` (33.5%).

We choose it over plain accuracy because of the arithmetic above. A classifier that
predicts `joy` for every input scores 33.5% accuracy while being useless. More
insidiously, a *good* classifier that quietly abandons `surprise` and `love` can
score in the high 80s on accuracy while failing the two classes a human reviewer
would notice first. Macro F1 makes that failure visible and expensive.

We report accuracy as well, because it is what a non-technical stakeholder will ask
for, and because the *gap* between accuracy and macro F1 is itself a diagnostic: a
wide gap means the model is trading the tail classes for the head classes.

**We also report per-class F1 for all six classes.** The head-to-head table in Phase
16 carries every one. Aggregates conceal the mechanism; per-class numbers reveal it,
and Phase 17 (error analysis) depends on them.

### Target

**Macro F1 ≥ 0.85 on the held-out test split**, set before any model is fitted.

The justification: published work on this dataset routinely reports 88–93% plain
accuracy for fine-tuned transformer encoders and low-to-mid 80s for tuned TF-IDF
linear models. Macro F1 sits below plain accuracy here because of the tail classes.
0.85 macro F1 is therefore a threshold a strong model can clear and a weak one
cannot — which is what a useful target must be. It is a hypothesis registered in
advance, not a bar we will move afterward.

### Cost and compute metrics

Every model reports all seven, measured under the protocol in §1.4:

| Metric | Definition | Why it matters |
|---|---|---|
| p50 single-row latency | median wall-clock ms for one text, batch size 1 | the typical user's wait |
| p95 single-row latency | 95th-percentile ms, batch size 1 | the tail that produces complaints; judged against the 50 ms budget |
| Batch throughput | rows/second at the model's best batch size | governs bulk re-scoring and backfill cost |
| Model size on disk | MB of all artefacts needed to serve | deployment footprint, cold-start time |
| Peak memory | peak RSS (CPU) or peak CUDA allocation (GPU), MB | decides what hardware tier is required |
| Est. cost / 1,000 predictions | USD, from the model in §1.5 | the single number a budget owner reads |
| Device used | the device the model actually ran on | without it, none of the above are comparable |

### The decision rule

The study answers "is it worth it" with an explicit rule, fixed now:

> The LLM is judged **worth it** only if it beats the best classical model by
> **≥ 3 macro F1 points** *and* stays inside the 50 ms p95 budget — or, if it
> exceeds the budget, delivers a margin large enough that the per-class gains fall
> on the operationally expensive classes (`anger`, `fear`).
>
> A margin under 3 points at any material cost increase is recorded as **not worth
> it**, and that is a valid, publishable finding.

Fixing this rule before seeing results is what stops the conclusion from being
written backwards from the numbers.

---

## 1.4 Measurement protocol (binding)

**Split discipline.** The dataset's own three-way split is used unmodified and is
frozen in Phase 2 before any modelling. Hyper-parameter search and model selection
use `train` + `val` only, via cross-validation. The `test` split is scored exactly
once per model, at the end. Every chunk states which split produced its numbers.

**Baseline first.** No model result is reported before the Phase 11 naive baselines
(majority class, stratified random) are on the table. Without them, a macro F1 of
0.60 has no meaning.

**Latency is measured, never estimated.** For each model: at least 20 warm-up runs
(discarded), then at least 200 timed runs. Single-row and batched throughput are
measured separately, because they answer different business questions. We report
p50 and p95, not the mean — the mean of a latency distribution is dominated by its
tail and describes no real request.

**Hardware stamping.** Every metrics JSON carries a full hardware block next to its
timing numbers: CPU model, physical and logical core count, RAM, GPU model, GPU
memory, driver version, CUDA version, PyTorch version, and the device the model
actually ran on. A fingerprint over the stable fields is frozen in
`reports/metrics/hardware_reference.json`; any run on a different machine raises an
error rather than silently writing an incomparable number. Timings from different
stamps are never placed in the same table.

**Reproducibility.** A single seed (42) is set for Python, NumPy and PyTorch at the
top of every script. Package versions are pinned in `requirements.txt`. Every metric
is persisted as JSON under `reports/metrics/`, and the Phase 16 comparison table is
assembled by reading those files — never by copying numbers from a transcript.

---

## 1.5 Cost model

Cost per 1,000 predictions is derived, not guessed, and the derivation is stated so
a reader can substitute their own rates:

```
cost_per_1000 = (1000 / throughput_rows_per_sec) / 3600 * hourly_rate_usd
```

Reference hourly rates, chosen as representative 2025 on-demand cloud prices:

- **CPU tier** — 8 vCPU / 16 GB general-purpose instance: **$0.192 / hour**
- **GPU tier** — single mid-range 8–24 GB NVIDIA accelerator: **$0.60 / hour**

Two caveats are carried into the final report. First, these rates are assumptions,
not measurements; the *ratios* between models are robust, the absolute dollars are
not. Second, this is a pure compute model: it excludes engineering time, model
storage, and the operational overhead of serving a large model, all of which favour
the classical baseline further. The reported LLM cost is therefore a **lower bound**.

---

## 1.6 Hardware for this study

All timings in this study were produced on one machine, frozen as the reference
stamp:

| Field | Value |
|---|---|
| CPU | AMD Ryzen 7 7435HS |
| Cores | 8 physical / 16 logical |
| RAM | 15.3 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU memory | 8,188 MiB |
| Driver | 580.126.09 |
| OS | Linux 6.17.0-22-generic |
| Python | 3.12.3 |

The 8 GB of GPU memory is a real constraint and it shapes Phase 13–15. It comfortably
fits an encoder model such as DistilBERT or RoBERTa-base for full fine-tuning, and it
fits a small instruct decoder (roughly 1–4B parameters) in reduced precision for
zero- and few-shot inference. It does not fit a 7B model at full precision. The
"small open-weight LLM" in the research question is therefore small by necessity as
well as by design — which happens to be the honest setting for the question, since a
model that cannot be served on commodity hardware has already lost the cost argument.

---

## 1.7 Risks registered in advance

**Label noise and subjectivity.** The labels are single-emotion, but real utterances
carry mixed affect, and `joy`/`love` in particular are semantically adjacent. Some
portion of the error rate is irreducible. Phase 8 (clustering) and Phase 17 (error
analysis) test this directly rather than assuming it.

**Leakage between splits.** The splits ship pre-divided; we did not draw them and
cannot assume they are clean. Phase 5 explicitly searches for exact and near-duplicate
overlap across `train`, `val` and `test`. If test rows also appear in train, every
reported score is inflated and the finding must be stated plainly.

**LLM output parsing.** A generative model asked for a label may return prose,
punctuation, a synonym, or a label outside the six. Phases 13–14 use strict parsing
and log every unparseable output with its raw text. Unparseable outputs are counted
as errors — silently retrying until the model complies would measure a system we are
not deploying.

**Class imbalance.** `surprise` has 572 training rows against `joy`'s 5,362, a 9.4:1
ratio. This is the study's main technical hazard and the reason macro F1 is primary.

**Hardware contention.** This is a laptop, not an isolated benchmark host. Timing
runs are executed with no other heavy workload, and the p95 (not the max) is
reported so that a single scheduler hiccup does not define the result.

---

## 1.8 Deliverables

- `reports/figures/` — publication-quality figures, one concern per figure
- `reports/metrics/*.json` — every metric, each with its own hardware block
- `reports/docs/` — per-phase write-ups
- Phase 16 — the single head-to-head comparison table, assembled from the JSON files
- Phase 18 — the final research-paper-quality report and recommendation

---

*Split used in this phase: none. No data was read for modelling purposes; only row
counts and label counts were verified against the specification.*
