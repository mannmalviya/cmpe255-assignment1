# Phase 10 — Association Rule Mining (Apriori)

CRISP-DM Phase 10 of 18.

**SPLIT USED: `train_clean` only** (15,923 rows). Rules mined from held-out data
would be leakage, and these rules inform the Phase 12 feature decision.

Artefacts: `src/phase10_apriori.py`, `reports/metrics/phase10_apriori.json`,
`reports/figures/fig11_apriori.png`.

---

## 10.1 Encoding and implementation

Each **document is a transaction**; its **distinct words are the items**. Word
order and repetition are discarded by construction — which is precisely why this
phase is informative about the research question. Apriori sees exactly what a
bag-of-words model sees. If word co-occurrence carries signal that single words do
not, Apriori will find it; if it does not, no amount of sequence modelling can
manufacture it from this representation.

Apriori is **implemented for this study** rather than imported. Support is counted
by intersecting an inverted index (item → set of document ids), so the support of
an itemset is the size of a set intersection, and candidates are pruned by
**downward closure**: every subset of a frequent itemset is frequent, so a
k-candidate can be discarded without counting whenever any of its (k−1)-subsets
was infrequent.

The pruning is what makes this tractable, and it is measurable:

| Corpus-wide pass (min support 0.3% = 48 docs) | |
|---|---:|
| Transactions | 15,923 |
| Itemsets actually counted | 417,200 |
| **Candidates pruned by downward closure** | **646,176** |
| Frequent 1-itemsets | 708 |
| Frequent 2-itemsets | 5,742 |
| Frequent 3-itemsets | 18,117 |
| Runtime | 3.7 s |

**61% of candidates were eliminated without ever being counted.** That is the
Apriori property doing its job, and it is reported because a phase that merely
prints rules has not demonstrated that it ran the algorithm.

Two support thresholds are used, deliberately. The per-class pass uses 1% of that
class. The corpus-wide pass uses 0.3% (≈48 documents) because the emotion
adjectives that matter are rarer than 1% — `amazed` appears in 70 documents, well
under the 159 that a 1% corpus threshold would demand. Choosing 1% everywhere
would have excluded the very words Phase 3 identified as the signal.

## 10.2 Part A — Co-occurrence within each emotion

Frequent itemsets per class (min support 1% of that class):

| Class | n | 1-itemsets | 2-itemsets | 3-itemsets |
|---|---:|---:|---:|---:|
| joy | 5,340 | 247 | 1,463 | 3,115 |
| sadness | 4,661 | 272 | 1,381 | 2,737 |
| anger | 2,152 | 249 | 1,620 | 3,352 |
| fear | 1,923 | 239 | 1,532 | 3,052 |
| love | 1,283 | 247 | 1,934 | 4,567 |
| surprise | 564 | 225 | 2,018 | **5,107** |

Note the inversion: **`surprise`, the smallest class, produces the most frequent
itemsets.** This is an artefact of relative support, not a property of surprise. A
1% threshold on 564 documents means 6 documents, so almost any coincidence clears
the bar. The same threshold on `joy` demands 54 documents. Small classes produce
more, and less reliable, itemsets — a standard trap in per-group market-basket
analysis, and worth naming because the raw counts invite the opposite conclusion.

### What the top rules actually are

Ranked by lift, the strongest within-class associations in the raw run were:

> `href` → `a + http` (lift 79–88, confidence 0.88–0.96) — in **every** class

This is the HTML residue Phase 8 discovered, and it dominates every lift ranking
because markup tokens co-occur near-perfectly with each other. It is a clean
independent confirmation of the Phase 8 finding: the strongest "association" in
this emotion corpus is `<a href="http://…">`.

It is reported once for that reason, then the 13 markup tokens are excluded so the
phase can say something about emotion. After exclusion, the top rules become:

> `don` → `feel + t` (lift 16.8–22.1) · `don` → `t + to` (lift 22.3)

Which is **also** a tokenisation artefact: `don't` became `don` + `t` when
punctuation was stripped upstream, so the two fragments co-occur almost perfectly.

**Part A's honest conclusion is a negative one.** Within-class market-basket
analysis on this corpus surfaces its *mechanical* regularities — surviving markup,
split contractions — not its emotional ones. The reason is structural: within a
single emotion class, every document already shares the `i feel …` frame, so the
highest-lift pairs are whatever else happens to be perfectly correlated. Emotional
content is constant within a class and therefore invisible to a within-class
analysis.

That is a real methodological lesson, not a failure to find anything: **the
interesting question was never "which words co-occur inside a class" but "which
word sets predict a class", which is Part B.**

## 10.3 Part B — Rules of the form {words} → emotion

Here a rule's confidence is `P(emotion | itemset)` and its lift is
`confidence / P(emotion)`.

**A rule is emitted for every emotion, not only the itemset's plurality label.**
The first version of this analysis took the majority label and produced *zero*
rules for `anger`, `love` and `surprise` — with a 3.5% prior, no frequent itemset
ever has `surprise` as its plurality class, yet an itemset that raises `surprise`
from 3.5% to 15% has lift 4.2 and is exactly the kind of rule the phase exists to
find. Taking majorities silences the small classes by construction.

130,320 rules met the support floor of 30 documents. The strongest per emotion:

| Rule | Confidence | Lift | Support |
|---|---:|---:|---:|
| `impressed` → **surprise** | 0.97 | 27.3× | 63 |
| `amazed` → **surprise** | 0.94 | 26.6× | 70 |
| `sympathetic` → **love** | 1.00 | 12.4× | 54 |
| `longing` → **love** | 0.94 | 11.7× | 54 |
| `apprehensive` → **fear** | 1.00 | 8.3× | 58 |
| `shaken` → **fear** | 0.98 | 8.2× | 65 |
| `feel + offended` → **anger** | 1.00 | 7.4× | 49 |
| `dissatisfied + i` → **anger** | 1.00 | 7.4× | 51 |
| `listless` → **sadness** | 1.00 | 3.4× | 48 |
| `disheartened` → **sadness** | 1.00 | 3.4× | 54 |
| `superior` → **joy** | 1.00 | 3.0× | 52 |
| `festive` → **joy** | 1.00 | 3.0× | 52 |

**Several rules reach confidence 1.00 on 48–58 supporting documents.** The word
`sympathetic` appears 54 times in the training set and the label is `love` all 54
times. `apprehensive` appears 58 times and is `fear` every time. `listless`: 48
times, `sadness` every time.

**A single adjective determines the label with near-certainty.** This is the Phase 3
finding — "the distinctive words are emotion adjectives" — upgraded from a ranking
to a probability, and it is the sharpest statement of the benchmark's structure
anywhere in this study.

### A caveat about lift

Lift is confidence divided by the class prior, so **the rarest class wins any global
lift ranking mechanically**. An unfiltered top-15 by lift is entirely `surprise`,
whose 3.5% prior inflates every rule by ~28×. That says more about the prior than
about the rules. Figure 11b therefore shows the two best rules *per emotion*, and
comparisons should be made within an emotion, not across.

## 10.4 The compositional test: do word pairs beat single words?

This is the phase's most important measurement, and it is the study's central
question asked in its cheapest possible form.

For every frequent 2-itemset, we take its strongest rule and compare its confidence
against the strongest rule of the better of its two constituent words. If combining
words carries information that individual words do not, pairs should systematically
beat their parents.

**5,446 pairs compared:**

| | |
|---|---:|
| Mean confidence gain | **−0.0086** |
| Median confidence gain | **−0.0070** |
| Pairs better than their best parent | 39.8% |
| Pairs gaining more than 10 points | **1.1%** |
| Pairs changing which emotion is predicted | 21.4% |

**On average, adding a second word makes the rule slightly worse.** Fewer than
four in ten pairs improve on their best single word at all, and barely one in a
hundred gains more than 10 confidence points. Figure 11a shows the whole population
hugging the diagonal.

The 21.4% that *change* the predicted emotion without improving confidence are the
tell: that is variance, not composition. A pair that flips the prediction while
scoring no better is redistributing a small sample, not discovering a phrase.

The largest genuine gains are unremarkable and semantically thin:

| Pair | Gain | Reading |
|---|---:|---|
| `more + now` → joy | +19.7 pts | frequency accident; `more` alone is 48.7% joy |
| `for + things` → sadness | +16.7 pts | flips from joy to sadness on 78 documents |
| `or + way` → sadness | +16.2 pts | `or` alone is 29.8% sadness |

None of these is a negation, an intensifier, or an idiom. There is no `not happy`,
no `so tired`, no phrase whose meaning differs from its parts. The compositional
structure that bigrams are supposed to capture is **absent from this corpus at
measurable scale**.

### What this predicts, and what it does not

**Prediction registered for Phase 12:** adding bigrams to the TF-IDF representation
will produce **no meaningful macro-F1 improvement**, despite costing 4.6× the
vocabulary and 1.8× the non-zeros per row (Phase 7). Cross-validation should select
unigrams, or select bigrams with a difference inside noise.

**What this does *not* establish.** Apriori sees only what a bag-of-words model
sees. It cannot detect signal that lives in word *order*, in long-range dependency,
or in constructions that span more than a co-occurrence window — because the
transaction encoding has already destroyed all three. The correct conclusion is
narrow and should stay narrow:

> **Within the bag-of-words representation, word combinations add essentially
> nothing over single words on this corpus.**

That is a statement about the representation's internal headroom, not about the
ceiling of the task. If a fine-tuned transformer in Phase 15 substantially beats
the classical models, the gain must come from something Apriori structurally cannot
see — order, context, or the world knowledge that Phase 6's genre B rows require.
Phase 10's contribution is to rule out the cheap explanation, so that any Phase 15
gain has to be attributed to the expensive one.

---

## Summary of findings

1. Apriori implemented for this study, with support counted by inverted-index
   intersection. **646,176 candidates pruned by downward closure** against 417,200
   actually counted — 61% eliminated without evaluation.
2. Two support thresholds used deliberately: 1% per class, **0.3% corpus-wide**,
   because the emotion adjectives that matter appear in 48–70 documents and a 1%
   corpus threshold (159 docs) would have excluded them.
3. **`surprise`, the smallest class, produces the most frequent itemsets** (5,107
   3-itemsets vs joy's 3,115) — an artefact of relative support on 564 documents,
   not a property of the class.
4. **Within-class rules surface mechanical artefacts, not emotion**: `href → a +
   http` (lift 79–88) in every class, independently confirming the Phase 8 HTML
   finding; and after excluding markup, `don → feel + t` from the split contraction
   `don't`. Within a class, the emotional content is constant and therefore
   invisible.
5. **Rules must be emitted for every emotion, not the plurality label.** Taking
   majorities produced zero rules for `anger`, `love` and `surprise`.
6. **Single adjectives determine the label with near-certainty**: `sympathetic` →
   love at confidence 1.00 on 54 documents; `apprehensive` → fear at 1.00 on 58;
   `listless` → sadness at 1.00 on 48; `impressed` → surprise at 0.97 on 63.
7. Lift favours rare classes by construction — an unfiltered top-15 is entirely
   `surprise`. Rules are compared within an emotion, not across.
8. **Word pairs do not beat single words.** Across 5,446 pairs: mean gain
   **−0.0086**, median **−0.0070**, only **39.8%** improve at all, only **1.1%**
   gain over 10 points. The 21.4% that flip the prediction without improving are
   variance, not composition.
9. The largest gains are frequency accidents (`more + now`, `for + things`). **No
   negation, intensifier or idiom appears anywhere** in the top gains.
10. **Prediction for Phase 12: bigrams will not meaningfully improve macro F1**,
    despite costing 4.6× the vocabulary.
11. Scope stated explicitly: this bounds the headroom *inside* bag-of-words. It
    says nothing about order, context or world knowledge — which is what any Phase
    15 gain would have to be attributed to.
