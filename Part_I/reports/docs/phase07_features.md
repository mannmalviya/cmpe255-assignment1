# Phase 7 — Feature Engineering

CRISP-DM Phase 7 of 18.

**SPLIT USED: every fit is on `train_clean` only (15,923 rows).** `val` and `test`
are transformed with the already-fitted objects. No held-out row influences a
vocabulary, an IDF weight, or a model parameter.

Artefacts: `src/phase07_features.py`, `reports/metrics/phase07_features.json`,
`data/processed/features/` (34.7 MB — vectorizers, sparse matrices, embedding
arrays).

Two representations are built independently, because the study compares two model
families that consume different inputs. **TF-IDF** is sparse, lexical, and fitted
to this corpus; it feeds Phase 12. **Sentence embeddings** are dense, semantic,
and pre-trained elsewhere; they feed Phases 8 (clustering) and 9 (LSH).

---

## 7.1 TF-IDF

Three configurations were fitted and costed. **The winner is not chosen here** —
Phase 12 cross-validates that choice on train+val. This phase measures what each
option costs, so the Phase 12 grid is informed rather than arbitrary.

Shared settings: `sublinear_tf=True` (log-scaled term frequency, appropriate when
a repeated word in a 17-word document should not count linearly),
`lowercase=False` and `token_pattern=r"\S+"` — the corpus is already lowercase
(§5.1) and contains only letters and spaces (§2.6), so the default tokenizer's
casing and punctuation handling would be dead work.

| Config | Vocabulary | nnz / row | Density | Fit (s) | Disk (MB) | Zero-vector test rows |
|---|---:|---:|---:|---:|---:|---:|
| unigram, min_df=1 | 15,188 | 16.97 | 1.12 × 10⁻³ | 0.14 | 0.13 | **0** |
| unigram, **min_df=2** | **7,257** | 16.47 | 2.27 × 10⁻³ | 0.13 | 0.06 | **0** |
| bigram, min_df=2 | 33,242 | 29.33 | 8.82 × 10⁻⁴ | 0.31 | 0.30 | **0** |

### `min_df=2` is nearly free

Raising `min_df` from 1 to 2 removes **7,931 features — 52.2% of the vocabulary** —
and costs **0.50 non-zeros per row**, from 16.97 to 16.47. Three percent of the
signal for half the feature space.

This is the Phase 3 hapax finding cashed out. Those 7,931 columns each held a word
seen exactly once in 15,923 documents. A coefficient fitted to one observation is
not an estimate; it is memorisation of a single row, and it will not fire again
because the word will not recur. Removing them shrinks the model, speeds the fit,
and reduces variance, at a cost that rounds to nothing.

### No document is ever unrepresentable

**Zero test rows land on an all-zero vector, in every configuration.** This matters
more than it may appear. A held-out document whose every token is out-of-vocabulary
gets a zero feature vector, and a linear model must then fall back to the
intercept — it predicts the majority class regardless of content, silently. That
failure mode does not exist here, even at `min_df=2`.

The reason is §3.2: 49 word types cover half of all tokens, and every document
contains several of them. The template guarantees representation. It also means
the TF-IDF path has no out-of-vocabulary weakness for an LLM to exploit — one more
way this benchmark is unusually kind to the classical model.

### Bigrams cost 4.6× the vocabulary

Adding bigrams takes the vocabulary from 7,257 to 33,242 (4.6×) and non-zeros per
row from 16.47 to 29.33 (1.8×). Whether that buys anything is an empirical question
about negation and intensifier phrases (`not happy`, `so tired`), which unigrams
cannot represent. Phase 12 decides it by cross-validation. The cost is recorded
here so the decision can be read as a trade rather than a default.

All three fit in **under 0.35 seconds** and serialise to **under 0.3 MB**. For
scale: this is four to five orders of magnitude smaller than the LLM artefacts of
Phases 13–15, and that gap is the substance of the study's cost argument.

## 7.2 Sentence embeddings

Model: **`sentence-transformers/all-MiniLM-L6-v2`** — 6 transformer layers,
**22.71 M parameters**, 384 output dimensions. Chosen as a deliberate
representative of the "small open-weight model" the research question names: it is
the standard workhorse sentence encoder, runs comfortably on CPU, and is small
enough that the cost comparison is honest rather than rigged against the LLM side.

Computed on **CPU**. These embeddings are an **analysis input** — they feed
clustering and LSH — not a timed model, so the device does not enter any latency
comparison. Any embedding-based model that reaches the Phase 16 table is re-timed
there under the full latency protocol on a stated device.

| Split | Rows | Dim | Encode (s) | Rows/s | Disk (MB) |
|---|---:|---:|---:|---:|---:|
| train_clean | 15,923 | 384 | 20.55 | 774.8 | 23.32 |
| val | 2,000 | 384 | 2.72 | 736.5 | 2.93 |
| test | 2,000 | 384 | 2.58 | 775.2 | 2.93 |

All vectors are L2-normalised, so the dot product is cosine similarity — which
Phase 9's LSH construction requires.

### Sequence length: the Phase 2 prediction confirmed

Phase 2 predicted from whitespace word counts that 128 tokens would suffice. That
was an estimate about *words*; transformers consume *sub-word* tokens. Measured
against the actual WordPiece tokenizer on `train_clean`:

| | Sub-word tokens |
|---|---:|
| median | 20 |
| p95 | 45 |
| p99 | 57 |
| **max** | **87** |
| **Rows exceeding 128** | **0** |

**Sub-word to word ratio: 1.161.** A 66-word document becomes at most 87 tokens.
The longest document in the corpus uses 68% of the 128-token budget.

The ratio being so close to 1.0 is itself informative: ordinary English of common
words tokenises almost one-to-one, because frequent words have dedicated vocabulary
entries. A corpus with names, URLs, hashtags or misspellings would run at 1.5–2.0×.
This is more evidence for §2.6 — the corpus has been pre-processed into an
unusually regular form.

**Consequence: sequence length is a settled question for the whole study.** Phases
13–15 use 128 with no truncation and no trade-off to tune.

## 7.3 How much does the pre-trained space already know?

This is the phase's most interesting question, and it has a two-part answer that
points in opposite directions.

### Globally, the class centroids nearly coincide

Mean pairwise cosine over a 4,000-row sample: **0.1754**. Class means:

| from ↓ / to → | joy | sadness | anger | fear | love | surprise |
|---|---:|---:|---:|---:|---:|---:|
| **joy** | **0.176** | 0.172 | 0.155 | 0.159 | 0.168 | 0.164 |
| **sadness** | 0.172 | **0.214** | 0.190 | 0.186 | 0.173 | 0.172 |
| **anger** | 0.155 | 0.190 | **0.192** | 0.173 | 0.156 | 0.157 |
| **fear** | 0.159 | 0.186 | 0.173 | **0.202** | 0.156 | 0.166 |
| **love** | 0.168 | 0.173 | 0.156 | 0.156 | **0.183** | 0.157 |
| **surprise** | 0.164 | 0.172 | 0.157 | 0.166 | 0.157 | **0.187** |

Every class is closest to itself — the diagonal dominates in all six rows — but
the **margins are minute**. `joy` beats its nearest rival by 0.004; `anger` by
0.002. Within-class similarity sits between 1.00× and 1.22× the corpus mean.

`sadness` is the nearest *other* class for five of the six classes. That is a
**hubness** effect, not a finding about sadness: its entire row is uniformly
elevated (0.172–0.214), so it sits in a dense central region of the sphere and
becomes many points' nearest neighbour. High-dimensional spaces routinely produce
such hubs, and the effect is a known bias in nearest-neighbour methods.

### Locally, the neighbourhoods are strongly informative

Mean cosine compares **centroids**, which is a weak statistic in 384 dimensions and
not what a classifier exploits. What matters is the **local neighbourhood**: of a
point's k nearest neighbours, how many share its label?

Measured at k = 10 on raw embeddings with **no fitting of any kind**:

| | Value |
|---|---:|
| Mean kNN label purity | **49.2%** |
| Purity expected by chance | 23.7% |
| kNN majority-vote accuracy | **65.8%** |

| Class | kNN purity | kNN vote accuracy |
|---|---:|---:|
| sadness | 57.5% | **80.3%** |
| joy | 55.9% | 74.6% |
| anger | 42.5% | 57.4% |
| fear | 41.4% | 49.6% |
| love | 28.6% | 36.9% |
| surprise | 18.1% | **19.6%** |

Retrieval is visibly working. Querying with an unseen sentence returns sensible,
correctly-labelled neighbours at high similarity:

> *"i feel so happy and grateful today"* → 0.871 `joy` "i feel happy and grateful to
> you all" · 0.865 `joy` "i feel so happy today me so" · 0.777 `joy` "i am feeling
> so blessed so happy"

### The divergence, and what it predicts

The two measurements disagree because they measure different geometry, and the
disagreement is the useful part:

- **Centroids are close** → methods that summarise a class by its mean will
  struggle. That is exactly what KMeans does.
- **Neighbourhoods are pure** → methods that work from local similarity will do
  well. That is exactly what kNN and LSH do.

**Predictions registered before Phases 8 and 9 are run:**

> **Phase 8 (KMeans/DBSCAN):** clustering will *not* recover the emotion labels.
> Adjusted Rand Index against the true labels should be low. The clusters that do
> emerge should be topical (work, family, relationships, food) rather than
> emotional, because `all-MiniLM-L6-v2` is trained for semantic textual similarity
> — "what is this about" — and "i feel great about my new job" is topically nearly
> identical to "i feel awful about my new job".
>
> **Phase 9 (LSH):** approximate nearest-neighbour retrieval will work well, with
> high recall against exact search, because the local structure it depends on is
> present.

If both hold, the conclusion is that the pre-trained space encodes *topic* globally
and *affect* only locally — which is precisely the gap that fine-tuning in Phase 15
closes.

### The difficulty ordering is already visible

The per-class kNN accuracies are **inversely ordered by class size**, with only one
inversion:

| Class | Train rows | kNN vote accuracy |
|---|---:|---:|
| joy | 5,340 | 74.6% |
| sadness | 4,661 | 80.3% |
| anger | 2,152 | 57.4% |
| fear | 1,923 | 49.6% |
| love | 1,283 | 36.9% |
| surprise | 564 | 19.6% |

`surprise` reaches 19.6% — better than its 3.5% base rate, but unusable. Phase 3
found `surprise` the most *lexically distinctive* class; here it is by far the
hardest. That is the same tension Phase 3 identified, now confirmed in a second,
independent representation: **class size, not separability, is the binding
constraint.**

Note also that raw embeddings plus a parameter-free kNN already reach 65.8%
accuracy. The Phase 11 majority baseline will be 34.75%. The Phase 1 target is
0.85 macro F1. Both numbers are needed to read 65.8% correctly — it is far above
the naive baseline and far below the target, which is why Phases 12–15 exist.

## 7.4 A constraint raised and resolved

At the start of this phase the study volume had **8.2 GB free (97% used)**.
CPU-only PyTorch and `sentence-transformers` were installed here, costing 1.2 GB —
a deliberate choice to unblock Phases 7–12 cheaply while the larger question was
settled.

Phases 13–15 need substantially more: a CUDA build of PyTorch is 3–4 GB, and a
small open-weight instruct model is 2–6 GB on disk. That did not fit.

**Resolved: the user freed space, and 43 GB is now available (85% used).** Phases
13–15 will therefore run on the **GPU** — the RTX 4060 Laptop, 8,188 MiB — using a
CUDA PyTorch build and an instruct model in the 1.5–3 B range, which is what fits
in 8 GB of VRAM at reduced precision.

This matters for the research question, not just for convenience. Zero-shot and
few-shot latency measured on CPU would be a measurement of the wrong system: nobody
deploys a decoder LLM on CPU for inline triage. Measuring on the GPU gives the LLM
family its best honest case, which is the only fair way to ask whether its accuracy
gain is worth its cost.

**Consequence for the hardware stamp.** The frozen reference fingerprint covers CPU,
RAM and GPU identity — not the PyTorch build — so switching from the CPU wheel to
the CUDA wheel does **not** invalidate it, and no re-run is triggered. What does
change is the `device_used` field, which every metrics file already records
alongside its timings. Phase 16 therefore reports CPU-device and GPU-device timings
in clearly separated columns and never averages across them.

---

## Summary of findings

1. All fitting used `train_clean` only; `val` and `test` were transformed, never fitted.
2. **`min_df=2` removes 52.2% of the TF-IDF vocabulary (15,188 → 7,257) at a cost
   of 0.50 non-zeros per row.** The Phase 3 hapax finding, cashed out.
3. **Zero test rows land on an all-zero vector in any configuration.** The template
   guarantees representability; TF-IDF has no out-of-vocabulary weakness here.
4. Bigrams cost 4.6× the vocabulary and 1.8× the non-zeros per row. Phase 12 decides
   by cross-validation whether they pay for themselves.
5. TF-IDF fits in under 0.35 s and serialises to under 0.3 MB — four to five orders
   of magnitude below the LLM artefacts to come.
6. **Sub-word tokens: max 87, p99 57, zero rows over 128.** The Phase 2 sequence-length
   prediction is confirmed against the real tokenizer. Ratio 1.161 — near-one-to-one
   tokenisation, further evidence of an unusually regular corpus.
7. Embeddings: 22.71 M parameters, 384 dimensions, ~775 rows/s on CPU, L2-normalised.
8. **Globally the class centroids nearly coincide** (within-class cosine 1.00–1.22×
   the corpus mean; `joy`'s margin over its nearest rival is 0.004). `sadness` is a
   high-dimensional hub, not a special case.
9. **Locally the neighbourhoods are strongly informative**: kNN purity 49.2% against
   23.7% chance, majority-vote accuracy 65.8%, with no fitting at all.
10. **Prediction registered: KMeans will fail to recover the labels (Phase 8) and LSH
    retrieval will succeed (Phase 9)** — the space encodes topic globally and affect
    only locally.
11. Per-class kNN accuracy is inversely ordered by class size (sadness 80.3% down to
    surprise 19.6%), confirming in a second representation that **class size is the
    binding constraint**.
12. **Open constraint: 8.2 GB disk free.** Phases 13–15 need CUDA PyTorch (3–4 GB)
    plus a 2–6 GB model. Escalated to the user.
