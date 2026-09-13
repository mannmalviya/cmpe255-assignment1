# Phase 5 — Data Cleaning and Leakage

CRISP-DM Phase 5 of 18.

**SPLIT USED: `train`, `val` and `test`.** Leakage is a relation *between* splits
and cannot be detected from one split alone, so all three are read. Only their
**text** is compared. No held-out row informs a fitting decision, and **`val` and
`test` are left byte-identical** — every removal falls on the training side.

Artefacts: `src/phase05_cleaning.py`, `reports/metrics/phase05_cleaning.json`,
`data/processed/train_clean.parquet`.

---

## 5.1 Normalization: proved to be a no-op

Phase 2 observed that the corpus contains only 27 characters. That is an
observation about *characters*; it does not by itself prove that the standard
text-cleaning pipeline would change nothing. So we ran the pipeline and measured
it, rather than reasoning from the character count.

The pipeline applied: Unicode NFKD decomposition → strip non-ASCII → lowercase →
strip punctuation → collapse whitespace → strip.

| Rows in corpus | Rows changed | Share |
|---:|---:|---:|
| 20,000 | **0** | 0.000000 |

**No cleaning transform is applied in this study.** The corpus arrives normalized,
and the finding is verified rather than assumed.

This is a real methodological point, not a formality. The default reflex in a text
pipeline is to lowercase and strip punctuation because that is what one always
does. Here it would have been dead code in every downstream phase — code that
still has to be read, maintained, and reasoned about when a result looks wrong.
Measuring first replaced a habit with a fact, and the fact removed the code.

## 5.2 Duplicate texts within each split

| Split | Rows | Distinct texts | Repeated texts | …of which labels conflict |
|---|---:|---:|---:|---:|
| train | 16,000 | 15,969 | 31 | **30** |
| val | 2,000 | 1,998 | 2 | **2** |
| test | 2,000 | 2,000 | 0 | 0 |

The `test` split contains no internal duplicates at all. In `train` and `val`,
almost every repeated text carries **conflicting labels** — 30 of 31, and 2 of 2.

That ratio is the first sign that something systematic is going on. If duplicates
were ordinary collection noise we would expect most of them to repeat the same
label. The opposite is true.

## 5.3 Overlap across splits

| Comparison | Shared texts | Rows affected in the later split | Labels agree |
|---|---:|---:|---:|
| train ↔ val | 5 | 5 (0.250% of val) | **0 / 5** |
| train ↔ test | 11 | 11 (0.550% of test) | **0 / 11** |
| val ↔ test | 3 | 3 (0.150% of test) | **0 / 3** |

**Not one cross-split duplicate agrees on its label. Zero out of nineteen.**

### This is not the leakage it appears to be — it is worse

The standard concern with train/test overlap is *optimistic bias*: the model has
memorised the answer and gets a free point, so the reported score overstates true
skill. That is not what is happening here.

Combining §5.2 and §5.3 across the whole corpus:

- Texts appearing more than once anywhere: **52**
- …carrying **conflicting** labels: **51**
- …carrying a single label (genuine redundancy): **1**

The dataset was evidently de-duplicated on the **(text, label) pair** rather than
on text alone. A text that appeared twice with the same label was collapsed; a
text that appeared twice with *different* labels survived, because the two rows
were not identical as pairs. The duplicates that remain are therefore, almost by
construction, exactly the ones annotators disagreed about.

The consequence inverts the usual analysis. A model that successfully learns
`"i feel cared for and accepted" → joy` from the training copy is **guaranteed to
be marked wrong** on the test copy, which is labelled `love`. These rows do not
gift the model free points; they deny them. They are adversarial.

Four real examples, train label first:

| Text | train | held-out |
|---|---|---|
| i feel cared for and accepted | joy | love |
| i feel so weird and scattered with all wonders about a million different things | fear | surprise |
| i have not conducted a survey but it is quite likely that many of them feel as assaulted b… | fear | sadness |
| i loved the feeling i got during an amazing slalom run whether it was in training or in a… | surprise | joy |

Read them as a human annotator would. Each is genuinely ambiguous. "I feel cared
for and accepted" *is* both joy and love. The dataset forces a single label onto
an utterance that carries two, and different annotators resolved it differently.
The disagreement is a property of the task, not a defect in the recording.

### The test-set accuracy ceiling

Fourteen distinct `test` rows (0.700%) have their text contradicted by a differently
labelled copy elsewhere in the corpus. For any model that fits the training data,
those rows are unwinnable.

> **No model in this study can exceed 99.30% test accuracy**, and the true ceiling
> is well below that, since these 14 rows are only the ambiguous cases we can
> *detect* — those that happen to have been duplicated. Ambiguous rows that appear
> once are invisible to this method and are certainly more numerous.

This number is carried into Phase 16 and Phase 18. It is the difference between
"the model achieved 89%" and "the model achieved 89% of a maximum that is not 100%".

> **Qualification (added in Phase 17).** "Unwinnable" holds only for a model that
> trained on the conflicting copy. §5.5 below removes those copies from
> `train_clean`, which disarms the trap for any model trained on `train_clean`
> alone. On these 14 rows, DistilBERT (trained on `train_clean` only) was right on
> 6, while the SVM, logistic regression and LightGBM, which also trained on `val`,
> scored 0 of 3 on the rows whose conflicting copy sits in `val`. So 99.30% is not a
> hard ceiling. What survives is the point that matters: these rows are genuinely
> ambiguous. Accuracy on them stays low for every model (21–43%), well below the
> 90%+ on ordinary rows.

## 5.4 Which emotions do annotators actually confuse?

The 51 conflicting texts are a direct, **model-free** measurement of label
confusability: no classifier is involved, only two humans disagreeing about the
same sentence.

We compare the observed collisions against what class sizes alone would produce.
Under independent draws, `P(pair = {a,b}) = 2·pₐ·p_b / (1 − Σpᵢ²)`.

| Pair | Observed | Expected by chance | Enrichment |
|---|---:|---:|---:|
| joy ↔ love | **29** | 3.7 | **7.8×** |
| fear ↔ surprise | **7** | 0.6 | **12.3×** |
| anger ↔ fear | 5 | 2.2 | 2.3× |
| anger ↔ sadness | 4 | 5.3 | 0.8× |
| joy ↔ surprise | 3 | 1.6 | 1.8× |
| fear ↔ sadness | 3 | 4.6 | 0.7× |
| **joy ↔ sadness** | **0** | **13.1** | **0.0×** |
| anger ↔ joy | 0 | 6.1 | 0.0× |
| fear ↔ joy | 0 | 5.4 | 0.0× |
| *(the remaining six pairs)* | 0 | ≤ 3.2 each | 0.0× |

**Caveat, stated plainly.** n = 51 is small, so most rows above are descriptive
rather than tested. Two are large enough to be robust: joy↔love (29 observed
against 3.7 expected) and joy↔sadness (0 observed against 13.1 expected, which has
probability ≈ 3 × 10⁻⁷ under the null). The middle of the table should not be
over-read.

### The finding that refines Phase 3

Phase 3 measured **joy ↔ sadness** as the *lexically closest* pair — the lowest
Jensen–Shannon divergence in the corpus, 0.118 bits. It is also the pair that
annotators, on this evidence, **never** confuse: expected 13 collisions, observed
none.

Lexical similarity and semantic confusability are therefore **different
quantities**, and this dataset separates them cleanly:

- **joy ↔ sadness** share vocabulary because they share the `i feel …` frame and
  the ordinary English around it. They are opposite in valence, so no annotator
  mistakes one for the other. *Lexically close, semantically far.*
- **joy ↔ love** and **fear ↔ surprise** are enriched 7.8× and 12.3×. These are
  same-valence neighbours — love is arguably a *sub-type* of joy, and surprise
  in this corpus is frequently negative surprise, adjacent to fear. *Semantically
  close, and genuinely ambiguous.*

The practical prediction, registered before any model is fitted:

> **The Phase 17 confusion matrix should be dominated by joy→love, love→joy, and
> fear↔surprise errors, and should show very few joy↔sadness errors, despite
> joy↔sadness being the lexically closest pair.**

If that prediction holds, the models are failing where humans fail, which is
evidence that the remaining error is irreducible ambiguity rather than model
weakness. If instead the models confuse joy↔sadness, they are failing in a way
humans do not, which would point at the feature representation. Either outcome is
informative, which is what makes it worth registering now.

This also sharpens the `love` and `surprise` problem. Both are small classes
(1,304 and 572 training rows) *and* both are the ones most confused with a much
larger neighbour. Small **and** ambiguous against a dominant class is the hardest
combination in classification, and it is why macro F1 was made primary in Phase 1.

## 5.5 Cleaning policy applied

Three removals, in order, **applied to `train` only**:

| | Rule | Rows removed |
|---|---|---:|
| **A** | training rows whose text also appears in `val` or `test` | 16 |
| **B** | all copies of a training text carrying conflicting labels | 60 |
| **C** | exact redundant repeats within train (same text, same label) → keep one | 1 |
| | **Total** | **77** (0.48% of train) |

Result: **train 16,000 → 15,923 rows. val 2,000 and test 2,000 unchanged.**

### Why the asymmetry

`val` and `test` are the benchmark. Three reasons not to touch them:

**Comparability.** Scores on a modified test set cannot be compared against any
other published work on this dataset. The 99.30% ceiling is a property of the
benchmark and should be *reported*, not engineered away.

**Integrity of the measurement.** Removing the 14 hard rows from test would raise
every model's score by up to 0.7 points without any model improving. A cleaning
choice must never be able to flatter a result.

**Direction of the error.** Removing rows from `train` can only cost us
information. It cannot manufacture an advantage. When a cleaning decision is
genuinely uncertain, the side that can only hurt you is the safe side to put it on.

### Why rule B removes *all* copies

When `"i feel cared for and accepted"` appears twice in train, once as `joy` and
once as `love`, there is no principled basis for keeping either. Keeping the first
by file order would encode an arbitrary tie-break as ground truth, and the model
would then be trained to be confidently wrong half the time. Removing both costs
60 rows of 16,000 and removes a contradiction the model cannot resolve.

### What is deliberately *not* done

- **No normalization transform** — proved a no-op in §5.1.
- **No stopword removal** — Phase 3 showed IDF suppresses the head automatically,
  and the frame words carry the template and negation structure.
- **No stemming or lemmatisation** — the signal is emotion adjectives, and stemming
  would merge distinctions (`amazed`/`amazing`, `loving`/`loved`) that the log-odds
  analysis shows are class-discriminative.
- **No near-duplicate removal** — exact matching is all that is possible without an
  embedding space. Near-duplicates are Phase 9's job, via LSH.
- **No class rebalancing** — imbalance is handled at model level in Phase 12 with
  `class_weight="balanced"`, and by macro F1 as the selection criterion.

## 5.6 Cleaned training data

`data/processed/train_clean.parquet`, 15,923 rows:

| Label | raw | clean | Δ |
|---|---:|---:|---:|
| joy | 5,362 | 5,340 | −22 |
| sadness | 4,666 | 4,661 | −5 |
| anger | 2,159 | 2,152 | −7 |
| fear | 1,937 | 1,923 | −14 |
| love | 1,304 | 1,283 | −21 |
| surprise | 572 | 564 | −8 |

The class balance is essentially unchanged — the largest relative loss is `love`
at 1.6%, consistent with §5.4, since `love` is the class most involved in
collisions. The imbalance ratio moves from 9.37:1 to 9.47:1.

**Both files are retained.** `train.parquet` is the frozen raw split, kept so the
Phase 2 verification stays reproducible. `train_clean.parquet` is what Phases 7–15
fit on. Every phase states which it used.

---

## Summary of findings

1. **Normalization is a measured no-op**: 0 of 20,000 rows changed by the full
   standard pipeline. No cleaning transform is used anywhere in this study.
2. `test` has no internal duplicates. `train` has 31 repeated texts, `val` 2.
3. **19 texts straddle split boundaries, and not one agrees on its label** — 0/5
   train↔val, 0/11 train↔test, 0/3 val↔test.
4. Corpus-wide, **51 of 52 repeated texts carry conflicting labels**. The dataset
   was de-duplicated on (text, label), so surviving duplicates are precisely the
   annotator disagreements.
5. **These rows are adversarial, not advantageous.** A model that learns the
   training copy is guaranteed to miss the held-out copy.
6. **Test accuracy ceiling: 99.30%**, and the true ceiling is lower — these are
   only the ambiguous rows that happen to be duplicated.
7. **Annotator confusion is concentrated**: joy↔love enriched 7.8×, fear↔surprise
   12.3×, while **joy↔sadness — the lexically closest pair — never collides at all**
   (0 observed, 13.1 expected). Lexical similarity ≠ semantic confusability.
8. Prediction registered for Phase 17: confusion should be dominated by joy↔love
   and fear↔surprise, with joy↔sadness rare.
9. Cleaning removed **77 training rows (0.48%)**. `val` and `test` are byte-identical
   to the originals. Cleaned data: `data/processed/train_clean.parquet`, 15,923 rows.
