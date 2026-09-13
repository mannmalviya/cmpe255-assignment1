# Phase 3 — Text EDA

CRISP-DM Phase 3 of 18.

**SPLIT USED: `train` only (16,000 rows).** `val` and `test` were not read.
Vocabulary statistics are a modelling input — the feature space in Phase 7, the
stopword and min-df decisions in Phase 12, the prompt wording in Phases 13–14 all
descend from them. Computing them on held-out data would be leakage through the
analyst rather than through the code.

Artefacts: `src/phase03_text_eda.py`, `reports/metrics/phase03_text_eda.json`.

Tokenization is whitespace splitting. Because Phase 2 established that the corpus
contains only 27 characters — the 26 lowercase letters and the space — whitespace
tokens *are* words, exactly, with no edge cases. Sub-word token counts are deferred
to Phase 7, where the model tokenizer is loaded and the answer becomes
model-specific.

---

## 3.1 Length distributions

### Overall (train, words)

| Statistic | Value |
|---|---:|
| min | 2 |
| p1 | 4 |
| p5 | 5 |
| p25 | 11 |
| **p50 (median)** | **17** |
| p75 | 25 |
| p90 | 35 |
| p95 | 41 |
| p99 | 52 |
| max | 66 |
| mean | 19.17 |
| std | 10.99 |

The distribution is **right-skewed**: Pearson's second skewness coefficient
`3(mean − median)/σ` is **+0.59**. The mean sits two words above the median, and
the upper tail stretches from p99 = 52 to a maximum of 66 while the lower bound is
hard at 2. This is the standard shape for user-generated text: a dense mass of
short utterances plus a thin tail of people who kept typing.

Three consequences are load-bearing for later phases.

**Sequence length is a solved problem.** p99 is 52 words and the maximum is 66. A
transformer `max_length` of 128 sub-word tokens covers every document in the
corpus with comfortable margin even after WordPiece or BPE expansion. There is no
truncation/compute trade-off to tune in Phases 13–15, and no 512-token attention
cost to pay. Phase 7 confirms this against the real tokenizer.

**TF-IDF rows will be extremely sparse.** A median document contributes about 17
non-zero unigram positions against a vocabulary of 15,212. That sparsity is the
mechanical reason the classical inference path will be measured in microseconds in
Phase 12 — a sparse dot product over ~17 entries is not a meaningful amount of
arithmetic.

**The short lower tail is a real risk.** One percent of documents are 4 words or
shorter, and the shortest is 2. At that length there may be no emotion-bearing
word at all. Phase 6 examines these rows directly rather than assuming they are
informative, and Phase 17 checks whether they concentrate the errors.

### Per class

| Class | mean words | median | p95 | max |
|---|---:|---:|---:|---:|
| joy | 19.50 | 17 | 42 | 66 |
| sadness | 18.36 | 16 | 40 | 66 |
| anger | 19.23 | 17 | 42 | 62 |
| fear | 18.85 | 17 | 41 | 63 |
| love | 20.70 | 18 | 45 | 62 |
| surprise | 19.97 | 18 | 43 | 55 |

**Length carries essentially no label signal.** The spread between the
longest-winded class (`love`, 20.70 words) and the tersest (`sadness`, 18.36) is
**2.34 words**, against a within-corpus standard deviation of **10.99 words**. The
between-class spread is under a quarter of one standard deviation.

This is a useful negative result. A document-length feature would contribute almost
nothing to any classifier, so we do not engineer one, and any model that appears to
exploit length is exploiting noise. It also rules out a specific confound: the
classes are not distinguishable by verbosity, so whatever separates them must be
lexical or semantic. Phase 4 plots the overlapping densities that make this visible.

## 3.2 Vocabulary

| Statistic | Value |
|---|---:|
| Total tokens | 306,661 |
| Distinct types | 15,212 |
| Type–token ratio | 0.0496 |
| Mean tokens per document | 19.17 |
| Hapax legomena (frequency 1) | 7,813 (**51.4%** of types) |
| Dis legomena (frequency 2) | 2,141 (14.1% of types) |
| Types appearing in ≥1% of documents | 204 |

### The hapax problem

**More than half the vocabulary — 7,813 of 15,212 types — occurs exactly once in
the entire training set.** Together with dis legomena, two-thirds of the vocabulary
occurs at most twice.

This is normal for natural language (it is the direct consequence of the Zipf
distribution below), but it is decisive for feature engineering. A type seen once
cannot support estimation of anything: a logistic regression coefficient fitted to
a single observation is noise with a confidence interval the width of the parameter
space, and it will not generalise because the word will almost certainly not recur
in val or test. Retaining all 15,212 types costs memory and invites overfitting for
no return.

**Decision carried into Phase 7:** TF-IDF uses `min_df = 2` as the default, which
removes 7,813 useless columns — 51% of the feature space — before any fitting.
`min_df` is nonetheless included in the Phase 12 cross-validation grid rather than
fixed by assertion, so the choice is validated on train+val instead of asserted here.

### Coverage

| Token mass covered | Types required |
|---|---:|
| 50% | 49 |
| 80% | 620 |
| 90% | 1,782 |
| 95% | 4,179 |
| 99% | 12,146 |

**Forty-nine word types account for half of all 306,661 tokens.** The top of that
list is `i` (25,859), `feel` (11,183), `and` (9,589), `to` (8,972), `the` (8,370),
`a` (6,200), `feeling` (5,112), `that` (5,112), `of` (4,990), `my` (4,283).

This is a steep head, and it has a direct methodological implication: TF-IDF's
inverse-document-frequency weighting will suppress these 49 types automatically,
because a token present in most documents has near-zero IDF. **An explicit stopword
list is therefore redundant and is not used.** More importantly, removing stopwords
by hand would be actively harmful here — `i`, `my`, `so`, `like` and the
`feel`/`feeling` alternation are part of the template structure identified in §3.4,
and negation and intensifier patterns depend on exactly these high-frequency
function words. We let IDF do the weighting and keep the information.

### Zipf's law

Fitting `log(frequency)` against `log(rank)` by least squares gives:

- slope **−1.293**
- R² **0.9725**

The corpus obeys Zipf's law closely (R² = 0.97 against a straight line in log-log
space). The slope is steeper than the canonical −1, which indicates the head is
heavier than a textbook Zipf corpus — consistent with the templated structure in
§3.4, where a small set of frame words repeats in nearly every document.

The practical reading: this is ordinary English text with an unusually dominant
head, not a corrupted or synthetic token stream. It also confirms that the hapax
share is a property of language, not a data defect.

## 3.3 Distinctive words per class

Ranking words by raw frequency within a class returns `i`, `feel`, `and`, `to` for
all six classes — the head of §3.2 — which is uninformative. Ranking by a plain
ratio returns rare noise, because a word seen twice in one class and never
elsewhere achieves an infinite ratio.

We therefore use the **log-odds ratio with an informative Dirichlet prior**
(Monroe, Colaresi & Quinn, 2008). Each word's class rate is shrunk toward its
corpus-wide rate, and the shrunk log-odds is divided by its own standard error. The
result is a z-statistic answering: *how confidently is this word over-used by this
class relative to the rest of the corpus?* It suppresses both stopwords (no
differential) and rare noise (large standard error) without a hand-tuned cutoff.

| Class | Top distinctive words (by log-odds z) |
|---|---|
| **joy** | successful, glad, satisfied, confident, cool, rich, more, brave, divine, honored, superior, useful, talented, festive, popular |
| **sadness** | exhausted, miserable, punished, melancholy, lethargic, awkward, discouraged, burdened, gloomy, ashamed, homesick, shitty, devastated, aching, groggy |
| **anger** | irritable, greedy, offended, resentful, fucked, angry, dissatisfied, bothered, insulted, dangerous, cranky, irritated, wronged, violent, pissed |
| **fear** | terrified, shaken, vulnerable, nervous, apprehensive, unsure, uncertain, reluctant, pressured, anxious, paranoid, intimidated, scared, hesitant, shaky |
| **love** | sympathetic, caring, loving, longing, horny, nostalgic, loyal, naughty, supportive, tender, gentle, fond, lovely, liked, hot |
| **surprise** | amazed, impressed, curious, surprised, shocked, funny, dazed, overwhelmed, weird, stunned, strange, amazing, enthralled, impacted, shinobi |

**Every one of these lists is a list of emotion adjectives.** Not topics, not named
entities, not syntactic patterns — adjectives that name the emotion, in the
predicate position of a "feel" clause. The separating signal in this dataset is
almost purely lexical.

This is the single most important finding in the phase, and it bears directly on
the research question. The advantage a pre-trained language model normally holds
lies in what bag-of-words cannot represent: word order, negation scope, long-range
dependency, sarcasm, compositional meaning. If the label is recoverable from the
presence of one adjective, almost none of that machinery is needed, and the
classical model can approach the ceiling. **We should expect a small LLM's margin
over TF-IDF to be narrow on this benchmark.** Registered in advance, as in §2.6.

Two caveats stated plainly:

**`surprise` z-scores are inflated.** Its top words score 7.5–8.7 versus 5.8–5.9
for `joy`. This is not because `surprise` is better separated; it is because
`surprise` has 572 training rows against `joy`'s 5,362, so a word concentrated in
`surprise` is a larger share of a smaller denominator. The z-scores are comparable
*within* a class, not *across* classes of different sizes.

**`shinobi` is an artefact.** It appears exactly once in the corpus (`train_14842`,
labelled `surprise`) inside a sentence about a console game. A single occurrence
reaching rank 15 of a distinctiveness list is the small-denominator effect above,
operating at the boundary of what the Dirichlet prior can shrink. It is a reminder
that the tail of any such list is noise, and it is flagged for Phase 6 (outlier
analysis) as an off-distribution row.

## 3.4 Corpus structure: the "feel" template

The word lists prompted a direct test, which confirmed a strong structural regularity:

- **15,566 of 16,000 documents (97.3%)** contain `feel`, `feeling`, `feels` or `felt`.
- **6,070 documents (37.9%)** begin literally with `i feel` or `im feeling`.

The corpus is **templated**. The dominant form is `i feel <adjective> [because ...]`.
The frame words are constant across all six classes; the adjective in the predicate
slot carries the label.

This explains three earlier observations at once: why the Zipf slope is steeper than
−1 (the frame repeats), why length carries no class signal (the frame is
class-independent and dominates length), and why the distinctive words are uniformly
adjectives (the frame is shared, so only the slot differs).

It also sharpens the central caution of this study. **The task, as posed by this
dataset, is closer to keyword lookup in a fixed syntactic slot than to open-domain
natural language understanding.** A bag-of-words model is nearly a sufficient
statistic for it. Benchmarks of this shape systematically understate the advantage
of pre-trained language models, and any conclusion in Phase 16 or 18 must be stated
as a conclusion about *this* benchmark rather than about text classification in
general. This limitation belongs in the final report's threats-to-validity section,
and it is recorded here so that it is a prediction rather than a rationalisation.

## 3.5 Class overlap

### Vocabulary Jaccard (type-set overlap)

|  | joy | sadness | anger | fear | love | surprise |
|---|---:|---:|---:|---:|---:|---:|
| **joy** | — | **0.336** | 0.286 | 0.280 | 0.261 | 0.180 |
| **sadness** | | — | 0.324 | 0.315 | 0.280 | 0.203 |
| **anger** | | | — | 0.316 | 0.287 | 0.239 |
| **fear** | | | | — | 0.300 | 0.254 |
| **love** | | | | | — | 0.257 |

### Jensen–Shannon divergence (bits) between class unigram distributions

JSD is 0 for identical distributions and 1 bit for distributions with no shared
support. It weights by how often words occur, whereas Jaccard only asks whether a
word occurs at all.

|  | joy | sadness | anger | fear | love | surprise |
|---|---:|---:|---:|---:|---:|---:|
| **joy** | — | **0.118** | 0.135 | 0.135 | 0.132 | 0.171 |
| **sadness** | | — | 0.120 | 0.126 | 0.148 | 0.169 |
| **anger** | | | — | 0.139 | 0.158 | 0.178 |
| **fear** | | | | — | 0.161 | 0.169 |
| **love** | | | | | — | **0.189** |

- Most similar pair: **joy | sadness** (JSD 0.118)
- Most different pair: **love | surprise** (JSD 0.189)

### What the two measures disagree about, and why it matters

The measures rank the pairs differently, and the disagreement is the interesting part.

**Jaccard** says `joy|sadness` overlap most (0.336). That is a size artefact: joy
and sadness are the two largest classes, with 5,362 and 4,666 documents, so they
have the largest vocabularies and mechanically share the most types. Jaccard on
type-sets is confounded by class size.

**JSD** also puts `joy|sadness` closest (0.118), but on frequency-weighted grounds,
which are not size-confounded in the same way. The two largest classes share the
`feel` template and the ordinary English that surrounds it, and differ only in a
predicate slot that is a small fraction of the token mass.

The overall range is narrow: every pair sits between 0.118 and 0.189 bits. **All six
classes are lexically close to one another,** which is exactly what §3.4 predicts —
97.3% of documents share a frame, so most of the probability mass is common to every
class and only the adjective slot separates them. The absolute JSD values are low
because the shared frame dilutes the signal, not because the classes are truly
confusable.

**`love|surprise` is the most separated pair (0.189).** Both are small classes with
narrow, highly specific adjective sets (`sympathetic, caring, loving` versus `amazed,
impressed, curious`) and little vocabulary in common. Small, specific classes are
*lexically* distinctive.

**This is the phase's key tension.** Lexical distance and classification difficulty
are not the same quantity. `surprise` is lexically the most distinctive class in the
corpus, yet it will almost certainly be the hardest to predict — because it has 572
training rows, the fewest by a factor of nine, and because a model minimising
aggregate loss has little incentive to fit it. `joy` and `sadness` are lexically
closest, yet both will score well, because each has thousands of examples. **The
binding constraint on this task is class size, not class separability.** Phase 8
(clustering on embeddings) tests whether the same ordering holds in a semantic
space rather than a lexical one, and Phase 17 checks it against real errors.

### Vocabulary spread across classes

| Appears in N classes | Types | Share |
|---:|---:|---:|
| 1 | 8,688 | **57.1%** |
| 2 | 2,576 | 16.9% |
| 3 | 1,354 | 8.9% |
| 4 | 926 | 6.1% |
| 5 | 743 | 4.9% |
| 6 | 925 | **6.1%** |

**57.1% of types appear in exactly one class.** This looks like strong evidence of
separability, and it is largely an illusion: it is the hapax population of §3.2
seen from another angle. A word occurring once necessarily occurs in one class. Of
the 15,212 types, 7,813 are hapax — so at most 875 of the 8,688 single-class types
occur more than once. **Genuine single-class signal is thin; the appearance of
separability is mostly sparsity.**

At the other end, **925 types (6.1%) appear in all six classes.** These carry no
discriminative signal alone and are precisely the high-IDF-suppressed frame words —
`i`, `feel`, `and`, `to`, `the`. That 6.1% of types accounts for the bulk of the
token mass is the same fact as "49 types cover 50% of tokens", restated.

---

## Summary of findings

1. Documents are short and right-skewed: median 17 words, p99 = 52, max 66, skew
   +0.59. Sequence length 128 covers the corpus; there is no truncation trade-off.
2. **Length carries no label signal.** Between-class mean spread is 2.34 words
   against a 10.99-word standard deviation. No length feature is engineered.
3. Vocabulary is 15,212 types over 306,661 tokens. **51.4% are hapax legomena.**
   `min_df = 2` removes half the feature space at no information cost; it is
   cross-validated in Phase 12 rather than fixed here.
4. Coverage is steeply headed — 49 types cover 50% of tokens. IDF suppresses them
   automatically, so **no explicit stopword list is used**; removing them by hand
   would destroy the template and negation signal.
5. Zipf fit is clean (R² = 0.97) with slope −1.29, steeper than canonical, which is
   the signature of the template.
6. **Distinctive words per class are uniformly emotion adjectives.** The separating
   signal is lexical, not compositional.
7. **97.3% of documents contain a `feel` word; 37.9% begin with `i feel`/`im
   feeling`.** The corpus is templated: `i feel <adjective>`. The task is closer to
   keyword lookup in a fixed slot than to open-domain understanding. This benchmark
   structurally favours bag-of-words models and narrows the expected LLM margin —
   registered before any model is fitted, and carried to the threats-to-validity
   section of Phase 18.
8. All class pairs are lexically close (JSD 0.118–0.189 bits) because the shared
   frame dominates the token mass. `joy|sadness` are closest; `love|surprise` most
   separated.
9. **Lexical distinctiveness and predictive difficulty diverge.** `surprise` is the
   most lexically distinctive class and will still be the hardest, because it has
   572 training rows. Class size, not separability, is the binding constraint.
10. 57.1% of types appear in one class only, but this is mostly the hapax
    population re-expressed, not genuine signal. 6.1% appear in all six and carry
    no discriminative information alone.
11. `shinobi` (`train_14842`) flagged as an off-distribution row for Phase 6.
