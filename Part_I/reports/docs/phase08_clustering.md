# Phase 8 — Unsupervised Analysis: KMeans and DBSCAN

CRISP-DM Phase 8 of 18.

**SPLIT USED: `train_clean` only** (15,923 rows), via the Phase 7 embeddings.
**Labels are used only to score clusters after the fact.** No clustering step sees
a label.

Artefacts: `src/phase08_clustering.py`, `reports/metrics/phase08_clustering.json`,
`reports/figures/fig08_clustering_metrics.png`,
`reports/figures/fig09_cluster_label_contingency.png`.

---

## 8.1 The prediction under test

Phase 7 registered this before any clustering was run:

> KMeans will **not** recover the emotion labels. ARI against the true labels
> should be low. The clusters that emerge should be **topical rather than
> emotional**, because `all-MiniLM-L6-v2` is trained for semantic textual
> similarity — "what is this about" — and "i feel great about my new job" is
> topically nearly identical to "i feel awful about my new job".

**The first half was confirmed decisively. The second half was partly wrong, and
the way it was wrong is the more interesting result.**

## 8.2 KMeans: the sweep

Ten values of k, full KMeans (`n_init=10`, seed 42), cosine silhouette on a fixed
5,000-row sample.

| k | Silhouette | ARI | NMI | Homogeneity | Completeness |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0439 | 0.0210 | 0.0214 | 0.0154 | 0.0352 |
| 3 | 0.0292 | 0.0374 | 0.0330 | 0.0279 | 0.0404 |
| 4 | 0.0242 | 0.0405 | 0.0358 | 0.0336 | 0.0383 |
| 5 | 0.0258 | 0.0459 | 0.0562 | 0.0568 | 0.0556 |
| **6** | **0.0249** | **0.0486** | **0.0606** | 0.0646 | 0.0570 |
| 8 | 0.0316 | 0.0333 | 0.0480 | 0.0552 | 0.0424 |
| 10 | 0.0292 | 0.0383 | 0.0572 | 0.0697 | 0.0485 |
| 12 | 0.0287 | 0.0304 | 0.0498 | 0.0639 | 0.0409 |
| 15 | 0.0301 | 0.0303 | 0.0609 | 0.0822 | 0.0484 |
| 20 | 0.0280 | 0.0229 | 0.0620 | 0.0894 | 0.0475 |

### There is no natural cluster count

**Silhouette never exceeds 0.044 at any k.** For reference, values above 0.5
indicate strong structure and 0.25–0.5 moderate structure. Everything here sits at
0.02–0.04 — indistinguishable from no structure at all.

The embedding space is not made of separated blobs. It is a roughly uniform cloud
on the unit sphere, and KMeans, asked for k pieces, will always return k pieces
regardless of whether k pieces exist. The silhouette curve is flat because the
partition is arbitrary at every k.

Figure 8a is plotted on the **interpretable** scale (−0.05 to 0.65) rather than
auto-fitted. Auto-scaling would have magnified a 0.02 spread into a dramatic curve
and contradicted the finding, which is that every value is near zero. The
structure-strength bands are drawn so the reader can see how far below "moderate"
the data sits.

### Clusters do not recover the emotions

**ARI at k = 6 is 0.0486.** ARI is chance-corrected: 0.0 is what random label
assignment produces, 1.0 is a perfect match. 0.049 is essentially chance.

The generous reading — assign each cluster its majority label, with the true labels
in hand — gives **43.0% accuracy**, and the six clusters between them claim only
**two of the six labels** (`joy` and `sadness`). For comparison, the Phase 11
majority-class baseline will be **34.75%**: an oracle-mapped six-way clustering
barely beats always guessing `joy`, and four of the six emotions are never named at
all.

Homogeneity rises with k (0.065 at k=6 to 0.089 at k=20) while completeness falls
(0.057 to 0.048). That is the expected signature of slicing a continuum: more,
smaller pieces are each slightly purer, and each class is scattered across more of
them. Neither number approaches a level that would indicate real class structure.

## 8.3 What the clusters actually are

Reading the clusters is where the prediction gets interesting. Distinctive words
per cluster, by the Phase 3 log-odds estimator:

| Cluster | n | Majority | Purity | Distinctive words | What it really is |
|---:|---:|---|---:|---|---|
| 0 | 2,571 | sadness | 0.33 | him he her his she you hes me | **third-person pronouns** — who the text is about |
| 1 | 2,171 | sadness | 0.32 | was had remember feeling didnt felt terrified | **past tense** — grammatical, not semantic |
| 2 | 2,448 | sadness | 0.36 | feeling im exhausted lethargic stressed today agitated irritable | negative, low-energy affect |
| 3 | 2,389 | joy | **0.63** | blessed thankful love divine wonderful joyful grateful happy | **positive affect** |
| 4 | 3,032 | sadness | 0.45 | myself life i am lonely helpless alone hopeless | negative, isolation |
| 5 | 3,312 | joy | 0.47 | http href they the popular are blog book | **web/blog boilerplate** |

**The prediction said "topical". The truth is more varied and more damning.** Three
different organising principles are visible, and none of them is the label set:

**Grammar.** Clusters 0 and 1 are organised by *pronoun person* and *verb tense*.
`him he her his she` versus `was had remember didnt`. These are syntactic
properties with no emotional content, and together they hold 4,742 rows — 30% of
the corpus.

**Provenance.** Cluster 5 is organised by *where the text came from* — `http href
blog` — which is discussed in §8.5.

**Valence, not emotion.** Clusters 2, 3 and 4 *are* affective, but they encode a
coarser distinction than the labels do. Cluster 3 is positive (63% joy + 17% love =
**80% positive valence**) and clusters 2 and 4 are negative, split by arousal:
cluster 2 is low-energy irritation (`exhausted lethargic stressed agitated`),
cluster 4 is despair (`lonely helpless alone hopeless`).

This last point is the real finding. **The unsupervised structure of the embedding
space is valence and arousal — the two axes of the circumplex model of affect — not
the six discrete categories the dataset labels.** Cluster 3 is the only cluster with
meaningful purity (0.63), and it achieves that by merging `joy` and `love` —
**exactly the pair Phase 5 found annotators confuse 7.8× more often than chance.**

Two independent methods — human annotator disagreement in Phase 5, and unsupervised
geometry here — place `joy` and `love` in the same region. That is strong converging
evidence that the joy/love boundary is a labelling convention imposed on a
continuum, not a distinction present in the language.

Figure 9 shows this directly: every row of the contingency table is a mixture, and
only cluster 3 has a single dominant cell.

## 8.4 DBSCAN: no density structure exists

`eps` was chosen from the k-distance curve rather than guessed. With
`min_samples = 10`, each point's distance to its 10th nearest neighbour:

| Percentile | 50 | 75 | 90 | 95 | 99 |
|---|---:|---:|---:|---:|---:|
| Cosine distance | **0.458** | 0.518 | 0.570 | 0.598 | 0.652 |

A *typical* point's 10th neighbour is 0.46 away in cosine distance. There is no
knee — the curve rises smoothly from 0.46 to 0.65 across the entire upper half of
the distribution. **A knee is what DBSCAN needs, and there isn't one.**

The sweep confirms it:

| eps | Clusters | Noise | ARI |
|---:|---:|---:|---:|
| 0.25 | 42 | **93.4%** | −0.001 |
| 0.30 | 47 | 83.6% | +0.003 |
| 0.35 | 18 | 70.2% | +0.009 |
| 0.40 | 2 | 50.1% | +0.012 |
| 0.45 | 1 | 27.4% | — |
| 0.50 | 1 | 12.0% | — |

There is no usable setting. Below the median neighbour distance, DBSCAN calls
almost everything noise and finds dozens of tiny fragments. Above it, everything
merges into a single component. The transition from "42 clusters, 93% noise" to
"1 cluster, 27% noise" happens across 0.2 of cosine distance, with no plateau in
between. **Best ARI at any setting: 0.012.**

This is the curse of dimensionality behaving exactly as the textbook says. In 384
dimensions, pairwise distances concentrate — the ratio between the nearest and
farthest neighbour approaches 1 — so "dense region" stops being a meaningful
notion. Density-based clustering has no signal to work with, and no amount of
parameter tuning creates one.

**Reported as a negative result, not a tuning failure.** We did not keep searching
for an `eps` that produced a flattering number; the k-distance curve says in advance
that none exists.

## 8.5 A defect the clustering found

Cluster 5's distinctive words were `http href they the popular are blog book`. That
prompted a direct check, which found a data-quality problem **that Phases 2 and 5
both missed**:

| Marker | Rows |
|---|---:|
| http | 189 |
| href | 158 |
| blog | 131 |
| amp | 73 |
| www | 53 |
| rel | 21 |
| target | 19 |
| **Any of the above** | **268 (1.68%)** |

Examples:

> i stopped feeling so exhausted **a href http** provokingbeauty
> i feel so dazed **a href http** twitter
> i feel unwelcome at work sometimes and think people might be talking about me **rel bookmark** i feel unwelcome at work sometimes and…

Upstream punctuation stripping turned `<a href="http://…">` into the bare tokens
`a href http …`, and `&amp;` into `amp`. **The HTML markup survived as
ordinary-looking words.**

Phase 2 certified the corpus clean on a **character** inventory; Phase 5 proved
normalization was a no-op. Both are correct, and both are blind at the **token**
level. A character-level check cannot see that `href` is markup rather than English,
because `href` is a perfectly ordinary-looking sequence of letters. The third
example also shows scraper-induced within-row duplication — the same sentence
repeated inside one document.

**Assessment, and what we do about it.** 1.68% of rows. **Left in place**,
consistent with the Phase 6 flag-don't-remove policy and the Phase 5 rule that
cleaning must never be able to flatter a result. The class lift is mild (`love`
1.30×, `joy` 1.19×, against `fear` 0.46×, `anger` 0.58×), so these tokens add noise
rather than spurious class signal, and both `min_df` and IDF already down-weight
them.

It is recorded as a limitation of Phases 2 and 5 — and as evidence that
**unsupervised analysis earns its place in the pipeline**. Neither the character
audit nor the duplicate audit could have surfaced this; a clustering that grouped
documents by provenance did, incidentally, in the course of failing at its nominal
task.

## 8.6 What this says about the research question

The two measurements from Phase 7 are now both resolved, and they point in opposite
directions as predicted:

| Measure | Result | What it uses |
|---|---:|---|
| Phase 7: kNN majority vote (k=10), no fitting | **65.8%** | local neighbourhoods |
| Phase 8: KMeans k=6 with oracle label mapping | **43.0%** | global centroids |
| Phase 11 baseline (to come): always predict `joy` | 34.75% | nothing |

**The pre-trained embedding space contains substantial class information locally
and almost none globally.** A method that asks "what are my nearest neighbours"
recovers two-thirds of the labels; a method that asks "what are the six main groups"
recovers essentially nothing beyond the majority class.

This matters for the study's central comparison in a specific way. A common claim
for pre-trained representations is that they place semantically similar things
together, so downstream tasks become easy. That is true here only at short range.
At the scale the label set operates on, the space is organised by valence, tense and
provenance — not by the six target categories.

**The consequence is that the "small LLM" family cannot win this task by
representation alone.** Whatever advantage it has must come from either supervised
adaptation (Phase 15 fine-tuning, which reshapes the space around the labels) or
from instruction-following over the raw text (Phases 13–14, which bypass the
embedding geometry entirely). Phase 8 rules out the third possibility — that the
frozen representation already solves the problem — and does so before any model is
fitted.

### Prediction registered for Phase 9

> LSH approximate nearest-neighbour search should achieve **high recall against
> exact search**, because the local structure it depends on is demonstrably present
> (Phase 7, kNN purity 49.2% vs 23.7% chance). The number of hyperplanes needed
> will be modest, because the neighbourhoods are genuinely local rather than
> diffuse.
>
> The counter-risk is the same dimensionality concentration that broke DBSCAN: if
> typical cosine distances cluster tightly around 0.46, random hyperplanes may
> struggle to separate near neighbours from mid-range ones, and recall at small
> hash widths could be poor. Phase 9 reports which effect dominates.

---

## Summary of findings

1. **KMeans does not recover the emotion labels. ARI at k=6 is 0.0486** — chance is
   0.0. NMI 0.0606.
2. **There is no natural cluster count.** Silhouette never exceeds 0.044 at any k
   from 2 to 20; 0.25 would be the threshold for even moderate structure.
3. An **oracle** mapping of clusters to majority labels reaches only **43.0%**
   accuracy against a 34.75% majority baseline, and names only **two of six**
   labels.
4. **The clusters are organised by grammar, provenance and valence — not emotion.**
   Cluster 0 = third-person pronouns; cluster 1 = past tense; cluster 5 = web
   boilerplate; clusters 2/3/4 = arousal and valence.
5. **The only meaningfully pure cluster (0.63) merges `joy` and `love`** — the same
   pair Phase 5 found annotators confuse 7.8× more than chance. Two independent
   methods converge: the joy/love boundary is a labelling convention, not a
   distinction in the language.
6. **DBSCAN finds no density structure at any setting.** Best ARI 0.012. The
   k-distance curve has no knee; the space transitions from 93% noise to a single
   component with no plateau. Distance concentration in 384 dimensions, reported as
   a negative result rather than a tuning failure.
7. **The clustering found a data defect Phases 2 and 5 missed**: HTML/URL residue
   surviving as word tokens in **268 rows (1.68%)** — `a href http …`, `amp`, `rel
   bookmark`. Character-level audits are blind to it. Left in place per policy;
   recorded as a limitation.
8. **Local structure is strong, global structure is absent**: kNN vote 65.8% vs
   KMeans-with-oracle 43.0%. The frozen pre-trained representation does not solve
   this task, so any LLM advantage must come from fine-tuning or from
   instruction-following, not from the embedding geometry.
