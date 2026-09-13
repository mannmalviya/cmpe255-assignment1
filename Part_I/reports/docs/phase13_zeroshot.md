# Phase 13 — Zero-Shot LLM

CRISP-DM Phase 13 of 18.

**SPLIT USED: `test` only (2,000 rows), scored once.** Nothing was fitted and
nothing was tuned — that is what zero-shot means. The training split was never
read, not even for a class prior.

**Device: `cuda:0`, NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB), driver
580.126.09, CUDA 12.8, PyTorch 2.11.0+cu128.** The full hardware block sits in
`reports/metrics/phase13_zeroshot.json` next to every timing.

Artefacts: `src/llm_common.py` (model, prompt and parser — shared with Phase 14),
`src/phase13_zeroshot.py`, `reports/metrics/phase13_zeroshot.json` (including the
full log of every unparseable output).

---

## 13.1 What was fixed before any result was seen

Three decisions were written into `src/llm_common.py` before the model generated
a single token on test, and none was revised afterwards. Revising the prompt after
reading the score would be prompt-tuning on the test split — the same violation as
tuning a hyper-parameter on it.

**The model.** `Qwen/Qwen2.5-1.5B-Instruct`, 1.544 B parameters, bfloat16. The size
was set by the hardware. At bf16, 1.5 B parameters is about 3.1 GB of weights,
which leaves room in 8 GB of VRAM for a KV cache at useful batch sizes. A 3 B model
would fit its weights but not comfortable batching, and the study needs batched
throughput as much as single-row latency.

**The decoding.** Greedy, `do_sample=False`, at most 6 new tokens. A sampled model
gives a different answer on each run and makes the score irreproducible.

**The prompt.** One system message:

> You are an emotion classifier. You are given one short first-person message.
> Reply with exactly one word: the single emotion the writer is expressing.
> The only allowed answers are: joy, sadness, anger, fear, love, surprise.
> Reply with the word alone. No punctuation, no explanation, no other text.

The message itself is sent as the user turn, with the model's own chat template.

**The parser.** Two tiers, both reported:

- **Strict (the scored rule).** The whole output, lowercased and stripped of
  surrounding whitespace and trailing punctuation, must equal one of the six labels.
  Anything else is **unparseable and counts as an error**.
- **Lenient (a diagnostic only).** The first label word found anywhere in the
  output. The gap between the two measures how much of the model's weakness is
  formatting rather than judgement.

There are **no retries, no repair and no constrained decoding**. Retrying until
the model complies would measure a system nobody deploys, and it would hide the
real cost per prediction.

## 13.2 Four measurement defects found and fixed

All four were in the harness, none in the model. They are recorded because each
would have put a wrong number into the Phase 16 table.

1. **The CUDA install silently did nothing.** `pip install --upgrade torch` from
   the CUDA index found `torch 2.14.0+cpu` and reported it as satisfying the
   request, because pip ignores the local version label (`+cpu`, `+cu128`) when
   deciding whether to upgrade. The fix was to uninstall the CPU build first. The
   CUDA index's newest build is **2.11.0**, so PyTorch moved down from 2.14.0. The
   hardware fingerprint (CPU, RAM, GPU, driver) did not change, so the re-run rule
   did not trigger. Phase 7's embeddings were saved under 2.14.0 and are reused as
   files, not recomputed. Exact versions are pinned in `requirements.lock.txt`.
2. **A progress check reported "still installing" for hours.** It searched for any
   process whose command line contained the install string, and matched its own
   shell.
3. **Model size was double-counted: 5,910.8 MB reported for a 2,955.4 MB model.**
   The Hugging Face cache keeps real files in `blobs/` and symlinks to them in
   `snapshots/`, and the size function followed the links. Fixed here and in the
   shared `evaluate.artifact_size_mb`.
4. **GPU peak memory was captured too early**: 3,360 MB, recorded before the
   batch-128 throughput test that sets the true peak of 4,540 MB. It is now recorded
   after all measurements.

**Reproducibility check.** The phase ran three times while these were fixed.
Accuracy, macro F1 and the count of unparseable outputs were **identical** on every
run, as greedy decoding requires. p95 latency varied between 60.9 and 61.2 ms.

## 13.3 Results

| | Zero-shot Qwen2.5-1.5B | Linear SVM (Phase 12) | Majority baseline |
|---|---:|---:|---:|
| **Macro F1** | **0.4107** | **0.8579** | 0.0860 |
| Accuracy | 0.4975 | 0.8995 | 0.3475 |
| Weighted F1 | 0.5122 | — | 0.1792 |
| Balanced accuracy | 0.4026 | — | — |
| Meets 0.85 target | No | Yes | No |

Per-class F1:

| Class | Test rows | Zero-shot | Linear SVM | Gap |
|---|---:|---:|---:|---:|
| sadness | 581 | 0.625 | 0.935 | −0.310 |
| joy | 695 | 0.567 | 0.927 | −0.360 |
| anger | 275 | 0.501 | 0.896 | −0.395 |
| fear | 224 | 0.322 | 0.864 | −0.542 |
| love | 159 | 0.295 | 0.811 | −0.516 |
| **surprise** | 66 | **0.154** | 0.714 | −0.560 |

**The zero-shot LLM loses by 44.7 macro F1 points.** It beats the naive baselines,
so it has real skill. But it recovers only about 40% of the distance from the
majority baseline to the linear SVM, and it loses on every class. The gap is widest
on exactly the classes that were hardest for everything else — `surprise`, `fear`,
`love`.

## 13.4 Every unparseable output

**143 of 2,000 outputs (7.15%) were unparseable, and all 143 are counted as errors.**
The full log is in the metrics JSON.

**Strict and lenient parsing gave identical results: 0.9285 each.** None of the 143
outputs contains a label word anywhere. So these are not formatting slips like
`"Joy."` or `"The emotion is fear"`, which lenient parsing would have caught. The
model answered with a **different word** — one of 61 distinct words outside the
allowed six:

| Output | Count | | Output | Count |
|---|---:|---|---|---:|
| stress | 12 | | confusion | 5 |
| pain | 12 | | embarrassment | 5 |
| uncertainty | 7 | | pleasure | 4 |
| peace | 7 | | hope | 4 |
| anxiety | 7 | | curiosity | 4 |
| excitement | 6 | | guilt | 4 |

The failures are not spread evenly across classes:

| True class | Unparseable | Share of that class |
|---|---:|---:|
| joy | 34 / 695 | 4.9% |
| sadness | 29 / 581 | 5.0% |
| anger | 20 / 275 | 7.3% |
| love | 13 / 159 | 8.2% |
| fear | 31 / 224 | **13.8%** |
| surprise | 16 / 66 | **24.2%** |

**A quarter of all `surprise` rows came back as a word the parser could not
accept.** For those rows the model said `curiosity`, `confusion`, `confused` — and
once `amazed`, the adjective sitting in the input text. For `fear` rows it said
`uncertainty`, `stress`, `anxiety`, `shy`, `doubt`.

These answers are **not wrong about the emotion**. Anxiety and uncertainty are
fear-family states; curiosity is a surprise-family state. The model is refusing
the **taxonomy**, not misreading the text. It holds a finer-grained vocabulary of
emotion than the dataset's six bins, and a one-line instruction was not enough to
make a 1.5 B model collapse that vocabulary onto the bins.

**What was deliberately not done.** An obvious move would be a synonym map —
`anxiety → fear`, `curiosity → surprise`. It is not applied. Any such map written
now would be written *after* reading these test outputs, which is tuning on test.
A map designed on `val` would be legitimate, but it changes the system being
measured, and Phase 14 already provides the sanctioned lever: showing the model
examples of the convention.

### Formatting is the smaller problem

On the 1,857 rows it **did** parse, the model's accuracy is **53.6%**. Removing
every unparseable output would lift accuracy by only about four points. **Most of
the loss is judgement, not formatting.**

## 13.5 What the model gets wrong

Confusion matrix (rows = true, columns = predicted; the 143 unparseable rows sit
outside this grid):

|  | joy | sadness | anger | fear | love | surprise |
|---|---:|---:|---:|---:|---:|---:|
| **joy** | 320 | 115 | 31 | 6 | **185** | 4 |
| **sadness** | 39 | 421 | 41 | 6 | 45 | 0 |
| **anger** | 16 | 83 | 126 | 8 | 21 | 1 |
| **fear** | 16 | **94** | 21 | 47 | 14 | 1 |
| **love** | 26 | 38 | 7 | 0 | 75 | 0 |
| **surprise** | 16 | 15 | 2 | 1 | 10 | 6 |

How often the model uses each label, against how often it is correct:

| Label | True count | Predicted count | Ratio | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| joy | 695 | 433 | 0.62× | 0.739 | 0.460 |
| sadness | 581 | 766 | 1.32× | 0.550 | 0.725 |
| anger | 275 | 228 | 0.83× | 0.553 | 0.458 |
| fear | 224 | 68 | **0.30×** | 0.691 | **0.210** |
| love | 159 | 350 | **2.20×** | **0.214** | 0.472 |
| surprise | 66 | 12 | **0.18×** | 0.500 | **0.091** |

Three patterns account for most of the damage.

**1. `love` is used as a catch-all for positive warmth.** The model predicts `love`
2.2 times more often than it occurs, and **185 of 695 `joy` rows — 26.6% — are
called `love`**. That single cell is the largest error in the matrix. Its precision
on `love` is 0.214: four out of five times it says `love`, it is wrong.

This is the joy/love boundary again. Phase 5 found annotators disagree on it 7.8×
more than chance. Phase 8 found the embedding space merges the two. Phase 12 found
it is 29.4% of the linear SVM's errors. The supervised SVM learned *where this
dataset draws the line* from 17,923 labelled rows. The zero-shot model has never
seen the line, so it draws its own, and draws it far more generously. **Where a
label boundary is a convention rather than a fact about language, a model that has
never seen the convention cannot follow it.**

**2. `sadness` absorbs the negative classes.** It is over-predicted 1.32×. The model
sends **94 of 224 `fear` rows (42%)** and 83 of 275 `anger` rows to `sadness`. It
appears to collapse negative affect toward its most prototypical member.

**3. `fear` and `surprise` almost vanish.** `fear` is predicted at 0.30× its true
rate, and `surprise` at 0.18× — only 12 times in 2,000. Their recall is 0.210 and
0.091. The fear and surprise rows that do not leak into `sadness` are exactly the
ones that came back unparseable (§13.4). Between the two routes, the model reaches
very few of them.

## 13.6 Cost and latency

End to end from raw text to parsed label, through the shared Phase 11 harness:
20 warm-up runs discarded, 200 timed single-row runs cycling real test rows, then
batched throughput at four batch sizes.

| | Zero-shot LLM | Linear SVM | Ratio |
|---|---:|---:|---:|
| Device | **cuda:0** (RTX 4060) | cpu (Ryzen 7 7435HS) | — |
| p50 single-row | 43.8 ms | 0.220 ms | 199× |
| **p95 single-row** | **61.2 ms** | **0.246 ms** | **248×** |
| p99 single-row | 70.2 ms | — | — |
| Within 50 ms p95 budget | **No** | Yes | — |
| Best throughput | 49.7 rows/s (batch 32) | 104,222 rows/s | 1/2,098 |
| Cost / 1,000 predictions | **$0.003355** | $5.12 × 10⁻⁷ | **6,557×** |
| Cost / month at 1 M messages | **$3.36** | $0.0005 | 6,557× |
| Model on disk | **2,955 MB** | 0.411 MB | **7,191×** |
| Peak memory | 4,540 MB GPU | — | — |
| Model load time | 6.9 s | < 0.1 s | — |

Throughput by batch size:

| Batch | 1 | 8 | 32 | 128 |
|---|---:|---:|---:|---:|
| Rows / s | 24.5 | 42.7 | **49.7** | 46.3 |

**The latency budget is missed.** p95 is 61.2 ms against a 50 ms budget. Even the
median, 43.8 ms, uses almost the whole budget, before any network or application
overhead.

**Batching stops helping at 32.** Throughput rises only 2× from batch 1 to batch 32,
then *falls* at 128. A GPU that was compute-bound would keep scaling. This one is
bound by the Python-level generation loop in `transformers` `generate()`, which runs
six decode steps of per-step bookkeeping regardless of batch width. A dedicated
serving engine with continuous batching would do better. That is an honest caveat,
but the gap to close is three orders of magnitude, and serving optimisations
typically buy one.

**On comparing across devices.** The two models ran on different devices, and each
is named in its row. This is not a like-for-like hardware comparison and is not
presented as one. It is a comparison of **how each model family is realistically
deployed**: nobody would serve a decoder LLM on CPU for inline triage, and nobody
would buy a GPU to serve a linear SVM. The cost ratio already accounts for this,
because each figure uses its own tier's hourly rate from Phase 1 §1.5. Timings from
different devices are shown side by side, with both devices stated, and are never
averaged or mixed within a single column.

## 13.7 The verdict

The Phase 1 decision rule, applied mechanically:

| Criterion | Required | Zero-shot LLM | Pass |
|---|---|---:|---|
| Macro F1 margin over best classical | ≥ +0.030 | **−0.447** | **No** |
| Macro F1 needed to pass | ≥ 0.8879 | 0.4107 | **No** |
| p95 latency | ≤ 50 ms | 61.2 ms | **No** |
| **Worth it** | both | — | **No** |

**The zero-shot small LLM fails every criterion.** It is less accurate by 44.7 macro
F1 points, slower by 248× at p95, costlier by 6,557× per prediction, larger by
7,191× on disk, and outside the latency budget. **This is a negative result and it
is stated plainly.**

### Why it lost

Three causes, each backed by a measurement above.

**The labels are a convention, and zero-shot has never seen it.** The dominant
error, `joy → love` at 26.6% of `joy` rows, lies on a boundary that Phases 5, 8 and
12 independently identified as a labelling convention, not a distinction in the
language. The SVM learned the convention from 17,923 examples. The zero-shot model
learned it from none.

**The model's emotion vocabulary does not match the six bins.** 61 distinct
out-of-set words, reaching 24.2% of `surprise` rows. The model often identifies a
reasonable emotion and then refuses to map it to the allowed one.

**The benchmark rewards what supervision learns cheaply.** Phase 10 showed that a
single adjective fixes the label with confidence up to 1.00. A linear model reads
those adjectives off 17,923 labelled rows. A zero-shot model must infer this
dataset's adjective-to-label mapping from general knowledge, and that mapping is
not the one general English would give.

### Scope of the claim

This is one model (1.5 B parameters), one prompt, and strict parsing. A larger
model, a better prompt, or constrained decoding might score higher. The claim is
bounded accordingly: **a small open-weight instruct model, used zero-shot with a
fixed prompt and strict parsing, does not approach a tuned linear SVM on this
benchmark, and costs thousands of times more.** Phase 14 tests the most direct
remedy — showing the model the convention through examples — without changing
anything else.

---

## Summary of findings

1. **Zero-shot Qwen2.5-1.5B-Instruct: macro F1 0.4107, accuracy 0.4975** on test,
   against the linear SVM's 0.8579 and 0.8995. **A loss of 44.7 macro F1 points.**
2. It beats the naive baselines (macro F1 0.154), so it has real skill, but recovers
   only about 40% of the baseline-to-SVM distance, and loses on every class.
3. **143 outputs (7.15%) were unparseable and counted as errors.** Strict and
   lenient parsing agree exactly: none contained a label word. The model answered
   with **61 distinct out-of-set words** — `stress`, `pain`, `anxiety`,
   `uncertainty`, `peace`, `curiosity`.
4. **24.2% of `surprise` rows and 13.8% of `fear` rows came back unparseable.** The
   model identifies plausible emotions but refuses the six-way taxonomy.
5. **No synonym map was applied**: written after reading test outputs, it would be
   tuning on test.
6. **Formatting is the smaller problem.** Accuracy on parsed rows is only 53.6%.
7. **`love` is a catch-all**: predicted 2.2× its true rate, precision 0.214, and
   **26.6% of `joy` rows called `love`** — the joy/love convention, which zero-shot
   has never seen.
8. **`sadness` absorbs negative affect** (42% of `fear` rows), while **`fear` and
   `surprise` nearly vanish** (recall 0.210 and 0.091).
9. **p95 61.2 ms — outside the 50 ms budget.** 248× slower than the SVM at p95.
10. **Throughput 49.7 rows/s, peaking at batch 32 and falling at 128** — bound by the
    generation loop, not the GPU.
11. **$0.003355 per 1,000 predictions — 6,557× the SVM**, or $3.36/month versus
    $0.0005 at 1 M messages. **2,955 MB on disk (7,191×)**, 4,540 MB peak GPU memory.
12. **Verdict under the Phase 1 rule: not worth it**, failing both the accuracy bar
    and the latency budget.
13. Four harness defects found and fixed (a no-op CUDA install, a self-matching
    progress check, double-counted model size, early GPU-memory capture). Quality was
    identical across all three runs.
