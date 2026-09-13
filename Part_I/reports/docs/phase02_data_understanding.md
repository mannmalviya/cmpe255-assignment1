# Phase 2 — Data Understanding

CRISP-DM Phase 2 of 18.

**Splits used: `train`, `val` and `test` were all read, for description only.**
Nothing was fitted. No row was altered, dropped or re-assigned. The three-way
split was frozen to disk at the end of this phase and is never re-drawn.

Artefacts: `src/data.py`, `src/phase02_data_understanding.py`,
`reports/metrics/phase02_data_understanding.json`,
`data/processed/{train,val,test}.parquet`.

---

## 2.1 Provenance and parsing

| File | Bytes | Rows |
|---|---|---|
| `data/raw/train.txt` | 1,658,616 | 16,000 |
| `data/raw/val.txt` | 204,240 | 2,000 |
| `data/raw/test.txt` | 206,760 | 2,000 |

The raw format is one record per line, `text;label`. The obvious parse —
`line.split(";")` — is wrong in principle, because a semicolon inside the text
would silently shift the label into the text column and corrupt a row without
raising anything. We therefore split on the **last** semicolon
(`str.rpartition(";")`) and assert that a separator was found on every line.

As it happens the corpus contains **zero** semicolons inside any text field, so
both parses agree here. We keep the defensive version anyway: the cost is one
method call, and the failure it prevents is a silent one.

Every row carries a stable `row_id` of the form `train_0`, `test_1999`. These
identifiers are assigned once, before any transformation, and are what Phases 5,
9 and 17 use to point back at a specific record.

## 2.2 Verification against the specification

All three checks in `data.verify()` passed:

- Row counts: train 16,000 / val 2,000 / test 2,000 — exact.
- Label set: exactly `{joy, sadness, anger, fear, love, surprise}` — no seventh
  label, no casing variant, no whitespace-padded duplicate.
- Train label counts: joy 5,362 / sadness 4,666 / anger 2,159 / fear 1,937 /
  love 1,304 / surprise 572 — exact.

This matters beyond bookkeeping. It establishes that the artefact we downloaded
is the artefact the study was designed against, so the phases that follow are
analysing the intended population and not a re-uploaded variant.

## 2.3 Structure

The combined frame is 20,000 rows × 4 columns (`row_id`, `text`, `label`,
`split`), all of dtype `object`, occupying 5.95 MB in memory with deep string
accounting.

- **Null cells: 0**, in every column, in every split.
- **Blank or whitespace-only texts: 0.**

There is no missing-data problem in this dataset. This is worth stating
explicitly rather than passing over, because it removes an entire class of
decisions — no imputation strategy, no missingness indicator, no
listwise-deletion bias — from the rest of the study.

## 2.4 Class balance

| Label | train n | train % | val n | val % | test n | test % |
|---|---:|---:|---:|---:|---:|---:|
| joy | 5,362 | 33.51 | 704 | 35.20 | 695 | 34.75 |
| sadness | 4,666 | 29.16 | 550 | 27.50 | 581 | 29.05 |
| anger | 2,159 | 13.49 | 275 | 13.75 | 275 | 13.75 |
| fear | 1,937 | 12.11 | 212 | 10.60 | 224 | 11.20 |
| love | 1,304 | 8.15 | 178 | 8.90 | 159 | 7.95 |
| surprise | 572 | 3.58 | 81 | 4.05 | 66 | 3.30 |

Two structural facts govern the rest of the study.

**The imbalance is real but moderate.** The max:min ratio is 9.37:1 in train and
10.53:1 in test. This is not the 1000:1 regime of fraud detection, where
specialised sampling is mandatory. It is the regime where a plain linear model
will still learn every class, but will systematically under-predict the tail —
`surprise` and `love` — because the loss is dominated by `joy` and `sadness`.
The defence is `class_weight="balanced"` in Phase 12 plus macro F1 as the
selection criterion, not resampling.

**Accuracy is compromised as a headline metric.** The majority class is `joy` at
34.75% of test. A constant `joy` predictor scores 34.75% accuracy and 0.086
macro F1. That gap — a factor of four — is precisely why Phase 1 made macro F1
primary. Phase 11 puts both numbers on the table so that every later score is
read against them.

## 2.5 Distribution drift between splits

The splits ship pre-divided. We did not draw them, so we must check whether they
were drawn from one population. If `test` carried a materially different class
mix from `train`, every held-out score would be measuring distribution shift as
well as model skill, and the two would be inseparable.

We use **total variation distance** (TVD) between the two label distributions,
`0.5 · Σ|pᵢ − qᵢ|`. It is 0 for identical mixes and 1 for disjoint ones, and it
reads directly as "the largest probability disagreement between the two
distributions over any set of labels".

| Comparison | TVD | Largest single-class share gap |
|---|---:|---:|
| train vs val | 0.0317 | 1.69 pp (joy) |
| train vs test | 0.0149 | 1.24 pp (fear) |

Both are small. A TVD of 0.015 means train and test agree to within 1.5
percentage points of total probability mass. The largest per-class disagreement
anywhere is 1.7 points. This is the scale of fluctuation expected from random
sampling of 2,000 rows — `surprise` at 3.6% in a 2,000-row draw has a standard
error of roughly 0.4 points, so a 0.3–0.8 point wobble is ordinary noise.

**Conclusion: the splits are stratified-compatible.** Held-out scores can be
interpreted as skill, not drift. This licenses the Phase 16 comparison.

## 2.6 Character-level composition

This is the phase's most consequential finding, and it was not anticipated.

| Property | Rows affected (of 20,000) |
|---|---:|
| Contains an uppercase letter | **0** |
| Contains a digit | **0** |
| Contains punctuation | **0** |
| Contains a non-ASCII character | **0** |
| Leading or trailing whitespace | **0** |
| Contains a double space | **0** |
| Contains a semicolon | **0** |

**Distinct characters in the entire corpus: 27** — the 26 lowercase ASCII
letters plus the space. Character frequencies follow ordinary English:
space (362,701), `e` (211,790), `t` (139,379), `i` (136,025), `a` (118,338),
`o` (109,461), `n` (104,483).

The corpus has already been aggressively normalized upstream by whoever built
the dataset: lowercased, stripped of punctuation and digits, ASCII-folded, and
whitespace-collapsed. Three implications follow.

**Phase 5 cleaning collapses to almost nothing.** The standard text-cleaning
pipeline — lowercase, strip punctuation, normalize unicode, collapse whitespace —
is a no-op on this corpus, verifiably so rather than presumptively. Applying it
anyway would be theatre. Phase 5's real work is therefore duplicate and leakage
detection, not normalization.

**Tokenization choices simplify.** There is no casing signal for a model to use
or lose, and no punctuation, so the emphatic cues a human reader relies on —
`!!!`, `?`, ALL CAPS — are absent. Both the TF-IDF pipeline and the transformer
tokenizers see the same flattened character stream.

**It removes a source of LLM advantage, and this is important for the research
question.** Part of a pre-trained language model's usual edge on raw social text
is robustness to casing, emoji, punctuation and typos, which a fitted TF-IDF
vocabulary handles badly. None of that variation exists here. The corpus has been
pre-processed into a form that suits a bag-of-words model unusually well. We
should therefore expect the classical baseline to be *stronger than typical*, and
the LLM's margin *narrower than typical*. This is a property of the benchmark, not
of the model families, and it is registered here — before any model is fitted — so
that it constrains the interpretation in Phase 16 rather than being produced as an
excuse afterwards.

## 2.7 Coarse size

Full length analysis is Phase 3. The headline figures:

| Split | chars min / median / mean / max | words min / median / mean / max |
|---|---|---|
| train | 7 / 86 / 96.9 / 300 | 2 / 17 / 19.2 / 66 |
| val | 11 / 85 / 95.4 / 295 | 2 / 17 / 18.9 / 61 |
| test | 14 / 86 / 96.6 / 296 | 3 / 17 / 19.2 / 61 |

The three splits are near-identical in size profile, reinforcing §2.5. The
documents are short — a median of 17 words, a maximum of 66 — and the mean sits
above the median in every split, which is the signature of a right-skewed length
distribution. Phase 3 characterises that tail.

Two engineering consequences. First, a 66-word maximum means a transformer
sequence length of 128 tokens covers the entire corpus with margin; there is no
truncation trade-off to reason about, and no need to pay for 512-token attention.
Second, short documents mean sparse TF-IDF rows — roughly 17 non-zero unigram
positions per document — which is why the classical inference path will be
measured in microseconds.

## 2.8 Duplicate preview

Counted, not removed. Phase 5 owns cleaning.

- Exact duplicate texts **within train**: 31
- Exact duplicate texts across all three splits combined: 52

Fifty-two duplicated strings out of 20,000 is 0.26% — small. But the number that
matters for validity is not the total; it is how many of those duplicates
**straddle** the train/test boundary, because those are leakage and they inflate
every reported score. That decomposition, together with near-duplicate detection
(Phase 9, via LSH), is deferred to Phase 5 where it can be acted on.

## 2.9 The frozen split

The split is now written to `data/processed/{train,val,test}.parquet` and is
**frozen**. Every subsequent phase loads it through `data.load_frozen(name)`.

The discipline this enforces:

- The split is never re-drawn, re-shuffled or re-stratified.
- `train` + `val` are the only data available for fitting and hyper-parameter
  selection. Cross-validation in Phase 12 runs over their union.
- `test` is read exactly once per model, to produce that model's final row in
  the Phase 16 table. It is not used for early stopping, threshold selection,
  feature selection, or any other choice.
- Every phase write-up names the splits it touched.

---

## Summary of findings

1. Data verified exact against specification — 16,000 / 2,000 / 2,000, six labels,
   train counts matching to the row.
2. Zero nulls, zero blank texts. No missing-data handling is required anywhere in
   this study.
3. Class imbalance is 9.4:1 (train) and 10.5:1 (test). A constant-`joy` predictor
   would score 34.75% accuracy but 0.086 macro F1 — the gap that justifies macro F1
   as the primary metric.
4. Splits are distributionally compatible (TVD ≤ 0.032, largest class gap 1.7 pp).
   Held-out scores measure skill, not drift.
5. **The corpus is already fully normalized: 27 distinct characters, all lowercase
   letters and space.** Phase 5 cleaning reduces to duplicate and leakage work.
   This benchmark is unusually favourable to bag-of-words models, which narrows the
   expected LLM margin — registered now, before any model is fitted.
6. Documents are short: median 17 words, max 66. Sequence length 128 suffices for
   any transformer, with no truncation trade-off.
7. 52 exact duplicate texts exist corpus-wide. The leakage-relevant subset is
   quantified in Phase 5.
