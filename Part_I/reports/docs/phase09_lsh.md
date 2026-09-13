# Phase 9 — Locality-Sensitive Hashing

CRISP-DM Phase 9 of 18.

**SPLITS USED.** *Part A (near-duplicates):* `train_clean`, `val` and `test` —
leakage is a relation between splits and cannot be found any other way. Nothing is
modified. *Part B (ANN benchmark):* the index is built from `train_clean` only;
`test` supplies **query vectors only**, and **no test label is read**.

Artefacts: `src/phase09_lsh.py`, `reports/metrics/phase09_lsh.json`,
`reports/figures/fig10_lsh.png`.

---

## 9.1 The method

The LSH family used is **random hyperplane LSH (SimHash)**, which is the correct
family for cosine similarity. For two unit vectors separated by angle θ, a random
hyperplane through the origin separates them with probability θ/π, so

```
P(same bit)  = 1 − θ/π
P(same n-bit signature)  = (1 − θ/π)^n
P(collide in ≥ 1 of L tables)  = 1 − (1 − (1 − θ/π)^n)^L
```

That last expression is the classic **S-curve**. Widening the signature (more
hyperplanes, `n`) makes each table more selective; adding tables (`L`) restores
recall by OR-ing their candidate sets. The two knobs trade recall against the
number of candidates that must then be re-ranked exactly.

The index is implemented from scratch (`HyperplaneLSH`, 48 lines) rather than
imported, because the point of this phase is to measure the mechanism, and the
mechanism has to be visible to be measured.

---

## 9.2 Part A — Near-duplicate detection, and the leakage Phase 5 could not see

Phase 5 searched for **byte-identical** duplicate texts. String matching is blind
to a document that differs by one character, and such a document leaks between
splits just as effectively.

Exact chunked cosine scan over all 19,923 rows (1.8 seconds — the corpus is small
enough that ground truth is affordable):

| Threshold | Pairs | Not byte-identical | Cross-split, non-identical | Labels agree |
|---|---:|---:|---:|---:|
| cos ≥ 0.90 | 430 | 425 | **138** | **122** |
| cos ≥ 0.95 | 55 | 50 | 20 | 17 |
| cos ≥ 0.99 | 9 | 4 | 2 | 2 |

### This is a different kind of leakage from Phase 5's

The contrast with Phase 5 is sharp and matters:

| | Phase 5 (exact match) | Phase 9 (near-duplicate, cos ≥ 0.90) |
|---|---:|---:|
| Cross-split pairs found | 19 | 138 |
| Labels agree | **0 (0%)** | **122 (88%)** |
| Effect on reported scores | pessimistic — unwinnable rows | **optimistic — free points** |

Phase 5's exact duplicates all *disagreed*, because the dataset was de-duplicated
on the (text, label) pair, so identical texts survived only where annotators
disagreed. Near-duplicates were never de-duplicated at all, so they behave the
ordinary way: a near-copy of a training row, carrying the same label, sitting in
the test set.

Examples, all cross-split:

| cos | A | B |
|---:|---|---|
| 0.999 | `train`/sadness — i get scared i feel ignored i feel happy i get silly | `test`/sadness — *(same text, differing later in the string)* |
| 0.998 | `train`/anger — i **didn t** think that it would come that fast… | `val`/anger — i **didnt** think that it would come that fast… |
| 0.980 | `train`/fear — i am feel overwhelmed | `val`/surprise — i feel very overwhelmed |
| 0.979 | `train`/anger — i did not care much about the number of viewers… | `test`/anger — *(near-identical)* |

The second row is the clearest demonstration of why exact matching is
insufficient: `didn t` versus `didnt` is a tokenisation artefact of the upstream
punctuation stripping. To a string comparison these are different documents. To a
reader, and to any model, they are the same sentence.

### How much test data is affected

The number that matters is not pairs but **test rows**:

| Threshold | Test rows with a near-duplicate in train | …of which ≥ 1 shares the label |
|---|---:|---:|
| cos ≥ 0.90 | 45 (2.25%) | **42 (2.10% of test)** |
| cos ≥ 0.95 | 9 (0.45%) | 8 (0.40%) |
| cos ≥ 0.99 | 1 (0.05%) | 1 (0.05%) |

**Roughly 2.1% of the test set is a near-copy of a training row with the same
label.** A model that fits the training data will answer those rows almost
perfectly.

### Decision, and why

**No rows are removed**, consistent with the Phase 5 rule that `val` and `test` are
the benchmark and a cleaning choice must never be able to flatter a result.
Removing the 42 easy rows *and* the 14 unwinnable rows of Phase 5 would produce a
test set that is neither the published benchmark nor comparable with anything.

Instead it is **quantified and carried into Phase 16**. The two effects push in
opposite directions and partially cancel:

- Phase 5: **14 test rows (0.70%)** are contradicted elsewhere → unwinnable.
- Phase 9: **42 test rows (2.10%)** are near-copies with matching labels → nearly
  free.

Together, 2.8% of test rows do not measure generalisation cleanly.

> **Correction (made in Phase 17).** An earlier version of this section concluded
> that "roughly 1.4 points of test accuracy" were artefact, by netting the two row
> *counts* (2.10% − 0.70%). That reasoning was wrong. A model that is 90% accurate
> on ordinary rows would have got most near-copies right anyway, so the inflation is
> the *difference* in accuracy on those rows, not their number. Phase 17 measured it
> directly from per-row predictions: the near-copies add about 4 rows and the
> contradicted rows remove about 7–10, for a **net effect of −0.28 accuracy points
> (linear SVM) and −0.26 (DistilBERT)**. The artefact rows slightly *understate*
> clean-row accuracy, by about a quarter of a point. They affect both models almost
> equally, so the comparison between models is unaffected, which was the claim that
> mattered.

---

## 9.3 Part B — ANN search: LSH loses to brute force

Index: 15,923 train vectors, 384 dimensions. Queries: the 2,000 test texts. Ground
truth: exact brute-force cosine top-10. Sixteen configurations, 4 signature widths
× 4 table counts.

**Exact brute force: 0.47 s total, 0.234 ms per query.**

| bits × tables | recall@10 | index scanned | ms/query | vs exact |
|---|---:|---:|---:|---|
| 8 × 1 | 0.054 | 0.70% | 0.048 | 4.9× faster |
| 8 × 4 | 0.206 | 3.54% | 0.174 | 1.3× faster |
| 8 × 8 | 0.417 | 9.94% | 0.514 | **2.2× slower** |
| **8 × 16** | **0.625** | 17.37% | 0.986 | **4.2× slower** |
| 12 × 16 | 0.229 | 1.98% | 0.145 | 1.6× faster |
| 16 × 16 | 0.082 | 0.28% | 0.101 | 2.3× faster |
| 20 × 16 | 0.028 | 0.03% | 0.087 | 2.7× faster |

**There is no good operating point.** Every configuration is either low-recall or
slower than the exact search it is supposed to approximate. The best recall
achieved — 62.5% — costs a 4.2× slowdown. At the speeds that *do* beat exact
search, recall is at most 23%.

### The Phase 8 prediction was wrong, and the counter-risk was right

Phase 8 registered:

> LSH should achieve **high recall against exact search**, because the local
> structure it depends on is demonstrably present. […] The counter-risk is the same
> dimensionality concentration that broke DBSCAN: if typical cosine distances
> cluster tightly around 0.46, random hyperplanes may struggle to separate near
> neighbours from mid-range ones, and recall at small hash widths could be poor.

**The counter-risk is what happened.** The main prediction failed. Two independent
causes, both measured rather than inferred:

### Cause 1: the neighbours are not near enough

Where do a test row's *true* neighbours actually sit?

| | median cosine | p10 | p90 |
|---|---:|---:|---:|
| 1st nearest neighbour | 0.632 | 0.504 | 0.802 |
| 5th | 0.568 | 0.447 | 0.714 |
| **10th** | **0.540** | 0.423 | 0.670 |

LSH is designed for the case where true neighbours sit at 0.9+ similarity, far to
the right of the bulk. Here the 10th neighbour sits at **0.54**, on the shoulder of
a distribution whose mass is centred near 0.18.

The theory predicts the outcome exactly. At cosine 0.540, θ = 57.3°, so
p_bit = 1 − 57.3/180 = 0.682:

| bits (×16 tables) | Predicted P(found) | Observed recall@10 |
|---:|---:|---:|
| 8 | 0.533 | 0.625 |
| 12 | 0.149 | 0.229 |
| 16 | 0.034 | 0.082 |
| 20 | 0.007 | 0.028 |

Observed recall tracks theory and sits slightly above it, as expected, since some
true neighbours are closer than the median. **This is not an implementation
failure. The geometry of this embedding space does not admit a good LSH
configuration,** and the S-curve says so in advance.

Figure 10b makes the diagnosis visual: the histogram of actual pairwise
similarities peaks at 0.18, the true-neighbour marker sits at 0.54, and the
collision curve passing through that point is at 53%. Widening the curve to catch
those neighbours also catches the bulk, so the candidate set explodes — which is
precisely what the 8-bit rows show (17.4% of the index scanned).

### Cause 2: exact search is already extremely fast

Even with good geometry, LSH would struggle here, because the baseline it must beat
is one BLAS call.

Exact top-10 over a 15,923 × 384 index is a single matrix multiply — about 6.1
million multiply-accumulates, which a modern CPU with AVX does in well under a
millisecond. Measured: **0.234 ms per query**, and that includes the top-k sort.

LSH replaces that with per-table hash computation, dictionary lookups, array
concatenation and a `np.unique` — Python-level and memory-bound work that does not
vectorise. **The bookkeeping costs more than the arithmetic it avoids.**

This is a general and often-forgotten point about approximate search: it pays off
at millions of vectors, where the exact matmul no longer fits in cache and the
asymptotics dominate. At 16,000 vectors, the constant factors decide, and they
favour brute force. **Reporting this rather than tuning until a flattering number
appeared is the honest result.**

### What would change the verdict

Stated so the negative result is bounded rather than overgeneralised:

- **Scale.** At 10⁶–10⁸ vectors the exact matmul stops being free. The crossover
  is a property of index size, not of LSH.
- **A fine-tuned embedding space.** Phase 15 reshapes the representation around
  the labels, which would push true neighbours toward 0.9 and move the S-curve into
  a usable position. LSH's failure here is a fact about *frozen* `all-MiniLM-L6-v2`
  on this corpus, not about LSH.
- **A compiled implementation.** Bit-packed signatures with popcount in C would cut
  the constant factor substantially. It would not fix the recall, which is a
  geometry problem.

---

## 9.4 What this phase contributes to the research question

Phase 9 adds two things, one to each side of the ledger.

**On the benchmark's validity.** The test set is not a clean measure of
generalisation. About 2.1% of it is near-copied from training data with matching
labels (optimistic), and 0.7% is contradicted elsewhere (pessimistic). Measured in
Phase 17, their net effect on reported accuracy is small, about −0.3 points, and
nearly identical across models. Model *comparisons* are unaffected, since every model
faces the same test set — and the comparison is the question being asked.

**On the cost argument.** This is the second phase in a row where the
sophisticated method loses to the simple one on a small corpus, for the same
underlying reason: **the classical operation is already so cheap that the clever
alternative cannot amortise its overhead.** Phase 8 found KMeans unable to beat a
majority-class baseline; Phase 9 finds LSH unable to beat a single matrix multiply.

That is a preview of the study's central tension, and it should temper expectations
for Phases 13–15. A 15,923-row, 17-word-per-document, template-structured corpus is
a setting where simple methods are hard to beat — not because the sophisticated
methods are bad, but because the problem is small enough that overhead dominates.
Whether an LLM escapes that logic is exactly what the remaining phases measure.

---

## Summary of findings

1. **Near-duplicate detection found leakage that Phase 5's exact matching could
   not see:** 138 cross-split near-duplicate pairs at cosine ≥ 0.90 that are not
   byte-identical.
2. **These behave oppositely to Phase 5's exact duplicates.** Exact: 19 pairs, 0%
   label agreement, pessimistic. Near: 138 pairs, **88% label agreement**,
   optimistic. `didn t` vs `didnt` is the clearest case — invisible to string
   matching, identical to any model.
3. **2.10% of test rows are near-copies of a training row with the same label.**
   Combined with Phase 5's 0.70% contradicted rows, 2.8% of test does not measure
   generalisation cleanly. No rows removed. *(Corrected in Phase 17: the net effect on
   reported accuracy, measured per row, is about −0.3 points and equal across models —
   not the +1.4 first estimated by netting row counts.)* Model comparisons remain valid.
4. **LSH loses to exact brute force at every one of 16 configurations.** Best recall
   62.5% at a 4.2× slowdown; the fastest settings reach at most 23% recall.
5. **The Phase 8 prediction failed and its stated counter-risk occurred.** True 10th
   neighbours sit at cosine **0.540**, on the shoulder of a bulk centred at 0.18 —
   too close for the S-curve to separate.
6. **Observed recall matches LSH theory** (8 bits × 16 tables: 0.533 predicted,
   0.625 observed). This is a geometry result, not an implementation defect.
7. **Exact search is one BLAS matmul at 0.234 ms/query.** LSH's hashing and set
   bookkeeping cost more than the arithmetic they avoid at this index size.
8. Bounded, not overgeneralised: LSH would win at 10⁶+ vectors, or on a fine-tuned
   embedding space where neighbours sit near 0.9.
9. **Second consecutive phase where the sophisticated method loses to the simple one
   on this corpus** — a preview of the study's central cost tension.
