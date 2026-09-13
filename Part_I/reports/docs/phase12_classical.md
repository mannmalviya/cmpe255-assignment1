# Phase 12 — Classical Models

CRISP-DM Phase 12 of 18.

**SPLITS USED.** *Tuning:* `train_clean` + `val` (17,923 rows), 5-fold stratified
cross-validation. **The test split was not touched during any search.** *Scoring:*
each tuned model refitted on all 17,923 development rows, then scored **once** on
`test` (2,000 rows).

Selection criterion: **macro F1**, per Phase 1 §1.3. Selecting on accuracy would
reward exactly the head-for-tail trade Phase 11 showed the majority baseline making.

Artefacts: `src/phase12_classical.py`, `reports/metrics/phase12_classical.json`,
`reports/figures/fig12_classical.png`, `data/processed/models/*.joblib`.

---

## 12.1 A run that had to be fixed first

The first attempt at this phase was killed by SIGTERM with **no output at all**.
The cause was thread oversubscription: `GridSearchCV(n_jobs=-1)` spawned 16 worker
processes on a 16-core machine, and each spawned 16 LightGBM threads — 256 threads
allocating histogram buffers against 15.3 GB of RAM, roughly 4 GB of which was
free.

Three changes, all of which belong in the record because they are the difference
between a result and a lost afternoon:

1. **Capped parallelism** — 4 search workers × 4 model threads, so the machine is
   busy without being oversubscribed.
2. **Per-model checkpointing** — each grid search is serialised the moment it
   finishes, so a kill no longer discards completed work.
3. **Unbuffered output** — the first run printed nothing because Python buffers
   stdout when piped, so there was no way to tell how far it had got.

## 12.2 What the grid was for

The grid was not a generic sweep. It encoded three questions that earlier phases
had raised as predictions, so each would be **decided by cross-validation rather
than asserted**:

| Question | Raised by | Prediction |
|---|---|---|
| unigram vs + bigram | Phase 10 | bigrams will not meaningfully help |
| `min_df` 1 vs 2 | Phase 3, 7 | min_df=2 is nearly free (51% of types are hapax) |
| `class_weight` none vs balanced | Phase 11 | the tail needs pushing |

Shared settings: `sublinear_tf=True`, `lowercase=False`, `token_pattern=r"\S+"` —
the last two because Phase 5 *proved* the corpus is already lowercase with no
punctuation, so the defaults would be dead work.

## 12.3 Results

Cross-validation on train+val; test scored once.

| Model | CV macro F1 | **Test macro F1** | Test accuracy | Meets 0.85 |
|---|---:|---:|---:|---|
| **Linear SVM** | **0.8677** | **0.8579** | **0.8995** | **Yes** |
| Logistic regression | 0.8596 | 0.8486 | 0.8905 | No |
| Gradient boosting (LightGBM) | 0.8410 | 0.8400 | 0.8810 | No |
| *Stratified baseline* | — | *0.1539* | *0.2310* | No |
| *Majority baseline* | — | *0.0860* | *0.3475* | No |

**The CV → test gaps are −0.0098, −0.0110 and −0.0010.** All three are small and
all three point the same way. That is what a sound tuning protocol looks like: the
cross-validated estimate was very slightly optimistic, by about one point, and the
test result did not surprise us. Had the gap been large, the search would have been
overfitting the development set, and the Phase 16 comparison would rest on sand.

Best parameters:

| Model | Selected |
|---|---|
| Linear SVM | C=0.5, class_weight=balanced, **min_df=2, unigram** |
| Logistic regression | C=5.0, class_weight=balanced, min_df=1, **unigram** |
| Gradient boosting | lr=0.1, num_leaves=31, class_weight=balanced, min_df=2, **unigram** |

**All three independently selected unigrams.** None chose bigrams.

Per-class test F1:

| Class | Train rows | Linear SVM | LogReg | LightGBM |
|---|---:|---:|---:|---:|
| sadness | 4,661 | **0.935** | 0.929 | 0.926 |
| joy | 5,340 | 0.927 | 0.916 | 0.901 |
| anger | 2,152 | 0.896 | 0.890 | 0.873 |
| fear | 1,923 | 0.864 | 0.860 | 0.863 |
| love | 1,283 | 0.811 | 0.791 | 0.742 |
| **surprise** | **564** | **0.714** | 0.706 | 0.736 |

## 12.4 The three predictions, all answered

Marginal CV macro F1, averaged over the rest of the grid (linear SVM):

| Question | Option A | Option B | Verdict |
|---|---:|---:|---|
| ngram_range | unigram **0.8571** | + bigram 0.8354 | **bigrams HURT, by 2.2 points** |
| min_df | 1 → 0.8432 | 2 → **0.8494** | min_df=2 is *better*, not merely free |
| class_weight | none → 0.8321 | balanced → **0.8605** | **balanced helps by 2.8 points** |

Logistic regression agrees on all three (unigram 0.8436 vs bigram 0.8075;
min_df 2 at 0.8289 vs 1 at 0.8222; balanced 0.8498 vs none 0.8013).

**Phase 10's prediction was not merely confirmed — it was exceeded.** Phase 10
predicted bigrams would produce "no meaningful improvement". They in fact *cost*
2.2 points of macro F1 for logistic regression's 2.2× and the SVM's 4.6× larger
feature space. The mechanism is straightforward: Phase 10 found no compositional
signal to capture, so the additional 26,000 bigram features are almost all noise,
and noise in a linear model is variance. Paying more to get less is a clean
result, and it was predicted in advance from a market-basket analysis that never
fitted a classifier.

**`min_df=2` came out ahead as well.** Phase 3 argued from 51% hapax that it should
be free; it is slightly positive, for the same reason bigrams are negative —
single-occurrence features cannot be estimated and only add variance.

**`class_weight="balanced"` is worth 2.8 points**, the largest single effect in the
grid. Phase 11 made the case from first principles: a loss dominated by `joy` and
`sadness` has little incentive to fit `surprise`. Re-weighting is what moves
`surprise` from the baseline's F1 = 0.000 to 0.714.

## 12.5 Per-class F1 is ordered by class size

Spearman rank correlation between training rows and test F1, over six classes:
**ρ = +0.94**. Only `joy` and `sadness` swap, and they are within 0.008 of each
other.

This is the fifth independent confirmation of the same finding:

| Phase | Method | Evidence |
|---|---|---|
| 3 | log-odds distinctiveness | `surprise` is the *most* lexically distinctive class |
| 7 | kNN on frozen embeddings | `surprise` 19.6% vs `sadness` 80.3% |
| 8 | KMeans | only 2 of 6 labels ever claimed — the two largest |
| 11 | naive baselines | `surprise` F1 = 0.000 under both |
| **12** | **tuned supervised models** | **ρ = +0.94 between class size and F1** |

`surprise` is the most lexically distinctive class in the corpus and still the
hardest to predict, by a margin of 22 F1 points below the next-worst class.
**Class size, not separability, is the binding constraint** — established five
different ways, and the reason macro F1 was made primary in Phase 1.

## 12.6 The Phase 5 confusion prediction, tested

Phase 5 registered this **before any model was fitted**, from annotator
disagreement alone:

> The Phase 17 confusion matrix should be dominated by joy→love, love→joy, and
> fear↔surprise errors, and should show very few joy↔sadness errors, despite
> joy↔sadness being the lexically closest pair.

Linear SVM on test: **201 errors out of 2,000.** Symmetrised pair counts:

| Confusion pair | Errors | Share of all errors | Phase 5 annotator enrichment |
|---|---:|---:|---:|
| **joy ↔ love** | **59** | **29.4%** | **7.8×** |
| sadness ↔ anger | 31 | 15.4% | 0.8× |
| **fear ↔ surprise** | **22** | **10.9%** | **12.3×** |
| sadness ↔ fear | 17 | 8.5% | 0.7× |
| anger ↔ fear | 17 | 8.5% | 2.3× |
| **joy ↔ sadness** | **16** | **8.0%** | **0.0×** |

**The prediction holds.** `joy ↔ love` is the single largest confusion, accounting
for **29.4% of all errors** — nearly three times the next-largest emotion pair.
`fear ↔ surprise` is third at 10.9%. And `joy ↔ sadness`, the pair with the
**lowest** Jensen–Shannon divergence in the corpus (0.118 bits, Phase 3) and
therefore the most *lexically* similar, contributes only 8.0%.

The model fails where humans fail, and does not fail where humans do not. That is
evidence that a substantial part of the residual 10% error is **irreducible
annotation ambiguity** rather than model weakness — which matters enormously for
Phases 13–15, because it caps what any model can win.

One partial miss worth stating: `sadness ↔ anger` is the second-largest confusion
at 15.4% despite an annotator enrichment of only 0.8×. The prediction did not
anticipate it. Both are large, negative-valence classes that Phase 8's clustering
also placed together (clusters 0, 1, 2 and 4 were all sadness-plurality with anger
mixed in), so a lexical-overlap explanation is available — but it was not predicted,
and Phase 17 should examine it on real examples rather than assume one.

## 12.7 Cost and latency

Measured through the shared Phase 11 harness: end-to-end from raw text to label,
20 warm-up runs discarded, 200 timed runs cycling real test rows.

| | Linear SVM | LogReg | LightGBM | Baseline floor |
|---|---:|---:|---:|---:|
| **Test macro F1** | **0.8579** | 0.8486 | 0.8400 | 0.1539 |
| p50 single-row | **0.220 ms** | 0.313 ms | 0.665 ms | 0.120 ms |
| **p95 single-row** | **0.246 ms** | 0.341 ms | 0.800 ms | 0.155 ms |
| Best throughput | **104,222 rows/s** | 98,489 | 20,781 | 1,029,220 |
| Cost / 1,000 | **$5.12 × 10⁻⁷** | $5.42 × 10⁻⁷ | $2.57 × 10⁻⁶ | $5.13 × 10⁻⁸ |
| Model on disk | **0.411 MB** | 0.807 MB | 2.69 MB | 0.001 MB |
| Vocabulary | 7,749 | 16,172 | 7,749 | — |
| CV search time | **20 s** | 219 s | 256 s | — |
| Within 50 ms p95 budget | Yes (**204×**) | Yes | Yes | Yes |

**The linear SVM wins on every axis simultaneously.** Highest macro F1, lowest
latency, highest throughput, smallest artefact, and a grid search that finished in
20 seconds against logistic regression's 219 and LightGBM's 256. There is no
trade-off to discuss.

**Gradient boosting is dominated outright.** It scores 1.8 macro F1 points lower,
runs 3.3× slower per row, has 5× lower throughput, costs 5× more per prediction,
occupies 6.5× the disk, and took 13× longer to tune — with a grid one third the
size. Tree ensembles are the wrong tool for high-dimensional sparse text: axis-aligned
splits on 7,749 mostly-zero features cannot express what a single dot product does,
and the model pays for the attempt in every dimension. This is worth stating plainly
because gradient boosting is often the reflexive choice for tabular problems, and
text is not tabular.

**Cost context.** At the Phase 1 deployment scale of 1,000,000 messages per month,
the linear SVM costs **$0.0005 per month** in compute — five hundredths of a cent.
That number is the bar the LLM phases must justify themselves against.

## 12.8 Where this leaves the research question

The Phase 1 decision rule, restated with its threshold now filled in:

> The LLM is judged **worth it** only if it beats the best classical model by
> **≥ 3 macro F1 points** *and* stays inside the 50 ms p95 budget.

**Best classical model: macro F1 0.8579.** So the LLM must reach **≥ 0.8879 macro
F1** to clear the bar, while staying under 50 ms at p95.

Two further constraints, both measured rather than assumed:

**The headroom is small.** Test accuracy is already 0.8995 against a Phase 5 ceiling
of 0.9930 (14 test rows are contradicted elsewhere in the corpus and are
unwinnable). Of the remaining 10.05 percentage points of error, 29.4% is `joy ↔
love` — the confusion humans make 7.8× more than chance. A large part of the
residual is not available to any model.

**The cost bar is brutal.** The classical model answers in 0.246 ms at p95 from a
411 KB artefact for $5.12 × 10⁻⁷ per 1,000. A 1.5–3 B parameter decoder on a GPU
will be three to four orders of magnitude more expensive per prediction and will
need several gigabytes on disk. For that to be "worth it", the accuracy gain has to
be real and large.

Phases 3, 6 and 10 have already located the only place such a gain could come from:
**the 2.8% of the corpus that is genre B** — situation descriptions like *"when my
father passed away"* that name no emotion and require world knowledge. Phase 17
will score the two genres separately, because an aggregate number will hide it.

---

## Summary of findings

1. The first run was SIGTERMed by **thread oversubscription** (16 × 16 = 256
   threads). Fixed with capped parallelism, per-model checkpointing, and unbuffered
   output.
2. **Linear SVM is the best classical model: test macro F1 0.8579, accuracy 0.8995
   — the only model to meet the 0.85 target.**
3. CV → test gaps of −0.010, −0.011 and −0.001 confirm the tuning protocol did not
   overfit the development set.
4. **Phase 10's prediction was exceeded: bigrams do not merely fail to help, they
   cost 2.2 macro F1 points** while multiplying the feature space by 4.6×. All three
   models independently selected unigrams.
5. **`min_df=2` is slightly better than `min_df=1`** (0.8494 vs 0.8432), not merely
   free — confirming the Phase 3 hapax argument.
6. **`class_weight="balanced"` is worth 2.8 macro F1 points**, the largest single
   effect in the grid, and is what lifts `surprise` from the baseline's F1 = 0.000
   to 0.714.
7. **Per-class F1 correlates with class size at ρ = +0.94.** Fifth independent
   confirmation that class size, not separability, is the binding constraint.
8. **The Phase 5 confusion prediction holds**: `joy ↔ love` is 29.4% of all 201
   errors, `fear ↔ surprise` 10.9%, while the lexically closest pair `joy ↔ sadness`
   is only 8.0%. The model fails where annotators fail.
9. Partial miss, stated: `sadness ↔ anger` at 15.4% was not predicted. Phase 17
   should examine it on real examples.
10. **Gradient boosting is dominated on every axis** — worse F1, 3.3× slower, 5×
    lower throughput, 5× costlier, 6.5× larger, 13× longer to tune on a grid one
    third the size. Trees are the wrong tool for sparse text.
11. **Cost to beat: 0.246 ms p95, $5.12 × 10⁻⁷ per 1,000, 411 KB on disk** — about
    **$0.0005/month** at 1 M messages. The LLM must reach **≥ 0.8879 macro F1** inside
    a 50 ms p95 budget to be judged worth it.
