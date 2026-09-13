# Phase 6 — Outlier Analysis

CRISP-DM Phase 6 of 18.

**SPLIT USED: `train_clean` only** (15,923 rows, from Phase 5). Deciding what
counts as off-distribution is a modelling judgement, so it is made on training
data. `val` and `test` are not read.

Artefacts: `src/phase06_outliers.py`, `reports/metrics/phase06_outliers.json`,
`reports/figures/fig07_outliers.png`,
`data/processed/train_outlier_flags.parquet`.

---

## 6.1 Policy: flag, do not remove

Nothing is deleted in this phase.

An outlier is only worth removing when it is *corrupt* — when it carries no
information about the population we want to model. The rows examined below are
not corrupt. They are unusual, and unusual rows are precisely where models fail.
Removing them would raise every reported score without any model improving, which
is the same defect §5.5 rejected for the test split.

The output is therefore a **flag table**, `train_outlier_flags.parquet`, one row
per training document with five boolean columns. Phase 17 joins against it to ask
which model fails where.

## 6.2 The cue lexicon

Three of the four detectors need an answer to "does this document contain an
emotion word at all?" Rather than import a general-purpose sentiment lexicon —
which would be calibrated to a different corpus and would silently import its
assumptions — we derive one from this corpus with the Phase 3 estimator.

A word is a **cue** if its log-odds z against the rest of the corpus is ≥ 3 for
some class, capped at the 150 strongest per class.

| Class | Cue words |
|---|---:|
| joy | 106 |
| sadness | 98 |
| anger | 50 |
| fear | 46 |
| love | 30 |
| surprise | 12 |
| **Union** | **336** |

No class reached the 150 cap, so the threshold is what binds, not the cap. The
lexicon is small — 336 words against a 15,000-type vocabulary, 2.2% — so
"contains a cue" is a selective test, not a vacuous one.

The per-class sizes are themselves informative. `surprise` yields only 12 cue
words and `love` 30, against 106 for `joy`. That is the sample-size effect again:
with 564 training rows, `surprise` cannot accumulate enough evidence for many
words to clear z ≥ 3. The class with the fewest reliable lexical cues is also the
class with the fewest examples.

## 6.3 Length outliers

Tukey fences on word count (1.5 × IQR): Q1 = 11, Q3 = 25, IQR = 14, **upper fence
46**. The lower fence is −10, which is below the possible minimum, so the standard
procedure identifies **no short outliers at all**.

This is a known limitation of Tukey fences on a right-skewed, positively-bounded
variable, so we define the short group by percentile instead:

| Group | Definition | n | Share |
|---|---|---:|---:|
| Short | ≤ 5 words (the p5 from Phase 3) | 845 | 5.31% |
| Long | ≥ 46 words (upper Tukey fence) | 440 | 2.76% |

### Short documents: a hypothesis that was wrong

Phase 5 closed by predicting that the shortest documents might carry no emotion
word and therefore be unanswerable. **That prediction was wrong, and the
measurement says so clearly.**

| Length bucket | n | % containing a cue | Mean cues |
|---|---:|---:|---:|
| 2–5 words | 845 | **97.5%** | 1.87 |
| 6–10 | 3,017 | 99.0% | 2.30 |
| 11–17 | 4,364 | 99.6% | 2.80 |
| 18–25 | 3,759 | 99.9% | 3.41 |
| 26–40 | 3,111 | 99.9% | 4.12 |
| 41–66 | 827 | 100.0% | 5.29 |

There is no collapse. Even at 2–5 words, 97.5% of documents contain a cue, and
the average such document contains 1.87 of them. The reason is the template: a
5-word document is `i feel <adjective>` plus one word, and the adjective *is* the
cue. Shortness does not remove the signal here; it removes the padding around it.

The typical short rows bear this out — `i am feeling grouchy`, `i feel romantic
too`, `i feel angered and firey`. Each is short *and* unambiguous.

The class mix of short documents is near the corpus baseline (largest deviation:
`sadness` at 1.22× lift, `love` at 0.59×), so shortness is not a proxy for a class.

**Consequence for the study.** Short documents are not the hard cases, and a model
that fails on them is not failing because of length. This also weakens a possible
LLM advantage: one might expect a language model to do better on terse input by
supplying context, but there is little context to supply. Figure 7a records this
negative result rather than burying it.

### Long documents

440 rows at ≥ 46 words, up to the 66-word maximum. Their class mix is close to
baseline. They contain more cues (5.29 on average at 41–66 words), not fewer, so
if anything they are *easier* per-document — though more cues can mean more
*competing* cues, which Phase 17 should check.

The operational point from Phase 2 stands: 66 words is comfortably inside a
128-token budget, so long documents cost nothing in truncation.

## 6.4 Vocabulary-rarity outliers

For each document we compute the share of its tokens that are rare in the
training corpus (count ≤ 2). Documents at or above the p99 of that share — **22%**
— are flagged: **187 rows (1.17%)**.

The most extreme rows are revealing:

| row_id | label | rare share | text |
|---|---|---:|---|
| train_11624 | sadness | 0.70 | i kali ni feeling aku dah bertukar jadi boring benci |
| train_4150 | fear | 0.50 | earth crake |
| train_4997 | joy | 0.50 | during lectures |
| train_8824 | fear | 0.50 | in sweden |
| train_4741 | joy | 0.46 | im feeling virtuous i do a spinach feta cranberry salad with balsamic … |

Four distinct failure modes appear in five rows:

1. **`train_11624` is not English.** It is Malay ("i kali ni feeling aku dah
   bertukar jadi boring benci"). A TF-IDF model fitted on English will represent
   it as a near-empty vector. A multilingual pre-trained model would read it.
2. **`earth crake`, `in sweden`, `during lectures`** are two-word fragments with
   no emotion content whatsoever. These are genuinely unanswerable from the text
   alone, and they belong to the second genre identified in §6.5.
3. **`train_4741`** is a false positive of the detector: it is a perfectly normal
   sentence whose rare words are food nouns (`feta`, `cranberry`, `balsamic`). Its
   emotion cue, `virtuous`, is present and clear.
4. `shinobi` (`train_14842`), flagged in Phase 3, falls in the same category as (3)
   — unusual topical vocabulary, ordinary emotional content.

**The detector conflates "unusual topic" with "unanswerable".** That limitation is
recorded rather than patched: the flag is a pointer for Phase 17, not a verdict,
and Phase 17 reads the text before drawing conclusions.

Panel (b) of Figure 7 shows why the flag skews short. Rare-word *share* has
denominator equal to document length, so a 2-word document containing one rare
word scores 0.50 while a 40-word document would need 20 rare words to match it.
Short documents dominate this flag by construction. The visible hyperbolic bands
in the scatter are the discrete values k/n, not structure in the data.

## 6.5 The finding: the corpus contains two text genres

433 rows (2.72%) contain no `feel`/`feeling`/`feels`/`felt` at all. Inspection
shows these are not merely atypical sentences — they are a **different kind of
text**.

| Genre | Definition | n | Share | Median words |
|---|---|---:|---:|---:|
| **A** — first-person feel-report | starts with `i`/`im`/`ive`… **and** contains a feel-word | 15,474 | 97.18% | 17 |
| **B** — other | everything else | 449 | 2.82% | 18 |

Genre B examples:

> on a boat trip to denmark *(joy)*
> when my mums brother passed away after having been involved in a car accident *(sadness)*
> when i almost walked on a snake *(fear)*
> a study visit to a chicken factory the butchery *(anger)*
> as a child i suffered of nightmares even since than *(fear)*
> when my mother kept me in leadingstrings *(anger)*

Genre A reports a **feeling**. Genre B describes the **situation that caused**
one, and never names the emotion.

### The evidence

- 433 rows (2.72%) contain no feel-word.
- 100 of them open with `when` or a scene-setting preposition.
- 68 rows in the whole corpus start with `when`; **66 of those 68 are off-template.**
- Median length of off-template, non-first-person rows is 12.5 words against 17.0
  corpus-wide.
- Several show non-native English constructions (`suffered of nightmares even
  since than`, `kept me in leadingstrings`) and terse noun-phrase answers
  (`earth crake`, `in sweden`, `during lectures`).

The pattern — describe a situation, don't name the feeling, terse or
non-native phrasing — is the answer format of a *"describe a time you felt X"*
survey rather than a first-person status post.

**Stated as a limitation:** this is a stylistic inference from the text itself. We
did not verify it against the dataset's documented provenance, and it should be
read as a strong hypothesis about corpus composition, not an established fact
about sources.

Genre B's class mix is skewed: `anger` at **1.90× lift**, `fear` 1.22×, `love`
1.16×, against `surprise` 0.63× and `sadness` 0.72×. Whatever produced these rows
over-sampled anger relative to the main corpus.

### Why this matters more than anything else in the phase

Consider what a model must do with `when my father passed away`.

Nothing in that sentence names sadness. There is no adjective in the predicate
slot, because there is no predicate slot. The label is recoverable **only from
world knowledge** — from knowing that bereavement causes grief.

This is the exact capability that separates a pre-trained language model from a
bag-of-words model, and it is the one thing Phases 3 and 5 concluded this
benchmark does not test. It does test it — on 2.8% of the corpus.

**Consequence, registered before any model is fitted:**

> Phase 17 must report accuracy **separately for genre A and genre B**. An
> aggregate score will hide the only sub-population where a small LLM has a
> structural advantage. If the LLM's overall margin is small but its genre-B
> margin is large, the correct conclusion is not "the LLM does not help" but
> "the LLM helps on 2.8% of this corpus, and that fraction would be far larger in
> a deployment that did not pre-filter for the `i feel …` template."

This reframes the research question's answer from a single number into a
conditional one, which is the more useful answer for a deployment decision.

## 6.6 Cue-free rows

**73 rows (0.46%)** contain no word from the 336-word cue lexicon. Class lift:
`anger` **2.84×**, `fear` 1.81×, against `love` 0.34× and `surprise` **0.00×**.

These are largely the genre B rows of §6.5 — `on a boat trip to denmark`,
`fear of thief`, `the funeral of a friend who was killed in a car accident`,
`when my mother kept me in leadingstrings`.

`surprise` has zero cue-free rows. With 564 training examples, every one of them
is a template sentence carrying an explicit cue. The rarest class is, paradoxically,
the most lexically stereotyped.

## 6.7 Flag overlap

| Flags per row | Rows | Share |
|---|---:|---:|
| 0 | 14,072 | 88.4% |
| 1 | 1,748 | 10.98% |
| 2+ | 103 | 0.65% |
| 3+ | 20 | 0.13% |

The five detectors are largely independent: only 0.65% of rows trip two or more.
The 20 rows tripping three or more are the genuinely degenerate cases — short,
off-template, rare-vocabulary and cue-free at once — and they are the first place
Phase 17 should look.

---

## Summary of findings

1. **Policy: flag, never remove.** Flags are written to
   `data/processed/train_outlier_flags.parquet` for Phase 17 to join against.
2. A **336-word cue lexicon** was derived from this corpus (log-odds z ≥ 3).
   `surprise` yields only 12 cue words, `love` 30, against `joy`'s 106 — the
   sample-size effect again.
3. Tukey fences find **no short outliers** (lower fence −10), a known limitation on
   a right-skewed positive variable. Short documents were defined by percentile.
4. **The Phase 5 prediction that short documents would be uninformative was wrong.**
   97.5% of 2–5 word documents contain an emotion cue; there is no collapse at any
   length. Shortness removes padding, not signal.
5. 187 rows (1.17%) are vocabulary-rarity outliers, including one **non-English
   (Malay)** row and several two-word fragments with no emotion content. The
   detector also produces false positives on ordinary sentences with unusual topical
   nouns — a limitation recorded, not patched.
6. **The corpus contains two text genres.** 97.2% is first-person feel-report
   (`i feel <adjective>`); **2.8% describes the situation that caused an emotion and
   never names it** (`when my father passed away` → sadness). Evidence: 66 of the 68
   `when`-initial rows are off-template; non-native English; terse noun-phrase
   answers. Inferred from style, not from documented provenance.
7. Genre B over-represents `anger` at 1.90× lift.
8. **Genre B is the only sub-population in this corpus that requires world
   knowledge rather than keyword lookup** — the one place a small LLM has a
   structural advantage. Phase 17 must score the two genres separately, or the
   aggregate will conceal it.
9. 73 rows (0.46%) contain no cue word; `anger` is enriched 2.84× and `surprise`
   has none at all.
10. Detectors are largely independent — 88.4% of rows trip no flag, 0.65% trip two
    or more, 20 rows trip three or more.
