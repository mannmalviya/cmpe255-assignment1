# Phase 4 — Data Visualization

CRISP-DM Phase 4 of 18.

**SPLIT USED: `train` only** for every analysis figure. Figure 1b displays class
*shares* for all three splits; this reuses counts already reported in Phase 2 as
part of data understanding, and no `val` or `test` text was read here.

Artefacts: `src/viz.py` (shared figure style), `src/phase04_visualization.py`,
`reports/figures/fig01`–`fig06`, `reports/metrics/phase04_visualization.json`
(figure manifest and palette validation record).

---

## 4.1 Design method

Figures in a research report are arguments, not decoration. Each of the six
figures below carries exactly one claim, and the claim is written in the figure's
own title so a reader who sees only the figure still receives the finding. The
subtitle carries the number that supports it.

**Palette.** Three categorical hues are used — blue `#2a78d6`, orange `#eb6834`,
aqua `#1baf7a` — and the set was **validated with a checker rather than by eye**,
against the light chart surface `#fcfcfb` with every pair in play
(`--pairs all`, the strict mode required when marks are not merely adjacent):

| Check | Result |
|---|---|
| Lightness band | PASS — all three inside L 0.43–0.77 |
| Chroma floor | PASS — all three ≥ 0.1 |
| Colour-vision-deficiency separation | PASS — worst pair ΔE 9.2 (deuteranopia), 9.6 (tritanopia) |
| Normal-vision floor | PASS — worst pair ΔE 24.0 |
| Contrast vs surface | **WARN** — aqua at 2.74:1, below 3:1 |

The contrast warning is not dismissable, so the **relief rule** is applied: the
only figure using aqua (Fig. 1b) carries a visible numeric label on every bar, so
identity and value never depend on the fill colour alone. This is why Fig. 1b's
bars are labelled and the others are not — it is a requirement, not a flourish.

Five of the six figures use **one hue only**, which sidesteps the question
entirely: identity is carried by facet titles, axis labels and direct labels.
Every figure is therefore legible in greyscale and in print.

**Other rules applied throughout.** Sequential magnitude (Fig. 6) uses a single
blue ramp, light → dark, never a rainbow. Bars are rounded on the data end and
square on the baseline, so the bar visibly sits on its axis. A 2 px surface gap
separates adjacent fills. Grid and axis lines are hairline and recessive; text
always wears ink tokens, never a series colour. No figure uses a dual axis.

**Every figure was rendered and inspected before acceptance**, which is how three
defects were caught and fixed: a corner-radius routine that mixed x- and y-axis
units (bars rendered square), legend swatches that vanished because the proxy
patches were the hidden originals, and a percentile label in Fig. 2 printed in
muted grey on top of a blue bar. None of these are visible in code review; all
three are obvious on sight.

---

## 4.2 The figures

### Figure 1 — `fig01_class_balance.png`
**Claim: the training set is imbalanced 9.4 : 1, and the three splits nonetheless
carry the same class mix.**

Panel (a) ranks the six training classes: `joy` 5,362 (33.5%) down to `surprise`
572 (3.6%). Bars are directly labelled with count and share, so the figure doubles
as a reference table.

Panel (b) places train, val and test side by side as shares. The bars track each
other closely at every class; the subtitle gives the supporting statistic (total
variation distance train vs test = 0.015, largest single-class gap 1.2 pp).

The two panels belong together because they answer the two questions a reader has
about any split: *how skewed is it*, and *is the skew the same everywhere*. The
first is the study's main technical hazard; the second is what licenses the Phase
16 comparison.

### Figure 2 — `fig02_length_distribution.png`
**Claim: documents are short, with a thin right tail; a 128-token sequence covers
the corpus.**

A histogram of words per document (bin width 2) with dashed reference lines at the
median (17), p95 (41) and p99 (52). The mode sits near 12 words and the tail
decays smoothly to the 66-word maximum.

The reference lines do the analytical work: they show that the tail, though
visible, is thin — 95% of the corpus is under 41 words — which is what makes the
sequence-length decision in Phases 13–15 free rather than a trade-off.

### Figure 3 — `fig03_length_by_class.png`
**Claim: document length carries no class signal.**

Six small multiples on shared axes. Each panel overlays that class's length
density (blue step) on the whole-corpus density (grey fill), with the class mean
marked in orange.

This is the clearest available form for a *negative* result. The blue line hugs
the grey fill in all six panels, and the orange means sit at visibly the same
place. A reader does not need to consult the numbers to conclude that length does
not separate the classes — though the subtitle supplies them: 2.34 words of
between-class spread against a 10.99-word standard deviation.

Small multiples were chosen over six overlaid densities deliberately. Six
overlapping lines would need six hues, which cannot clear the all-pairs
colour-separation floor; and overlaid curves this similar would be visually
illegible regardless of colour. Faceting solves both problems at once, and the
shared reference fill in every panel preserves the comparison that overlaying was
meant to provide.

### Figure 4 — `fig04_vocabulary_structure.png`
**Claim: the vocabulary is Zipfian, with a very heavy head and a 51% hapax tail.**

Panel (a) plots rank against frequency on log-log axes with the least-squares fit
(slope −1.29, R² 0.973). Four words are marked in place — `i`, `feel`, `the`,
`amazed` — so the abstract curve is anchored to the actual corpus. The empirical
curve sits above the fit at the extreme head, which is the template effect of §3.4
made visible.

Panel (b) plots cumulative token coverage against vocabulary size, with the 50 /
80 / 90 / 95% points marked, and shades the hapax region at the right.

The two panels answer the two feature-engineering questions. (a) confirms this is
ordinary English rather than a corrupted token stream. (b) justifies both
`min_df = 2` (the shaded block is half the vocabulary and the last 0.6% of tokens)
and the decision to use no stopword list (IDF already suppresses the 49 types that
cover half the corpus).

### Figure 5 — `fig05_distinctive_words.png`
**Claim: every class is separated by emotion adjectives, not by topic.**

Six panels, each the top ten words for one class by log-odds z with an informative
Dirichlet prior, sorted descending. Panel titles carry the class token count, which
is the context needed to read the bars honestly.

The figure's argument is made by repetition across panels: every list, without
exception, is a list of adjectives naming the emotion. No topical or entity
structure appears anywhere. This is the visual form of the phase's central caution
— if one adjective determines the label, a bag-of-words model is close to a
sufficient statistic for the task.

The subtitle states the comparison's limit explicitly: **bars are comparable within
a panel, not across panels**, because a smaller class yields a larger z. Without
that sentence a reader would conclude `surprise` (z up to 8.7) is better separated
than `joy` (z up to 5.9), which is an artefact of class size. Stating the
limitation on the figure rather than only in the text is the honest placement,
since figures travel separately from their prose.

### Figure 6 — `fig06_class_overlap.png`
**Claim: all six classes sit in a narrow lexical band, because they share the
"i feel …" frame.**

A lower-triangle heatmap of pairwise Jensen–Shannon divergence, on a single blue
sequential ramp with every cell directly labelled. The empty first row and last
column are trimmed rather than drawn as blank cells. The extremes are named inside
their own cells — `closest` on joy|sadness (0.118), `widest gap` on love|surprise
(0.189) — rather than by leader lines, which in an earlier draft collided with the
cell values.

The claim rests on the *range*, not on any single cell: every pair falls between
0.118 and 0.189 bits on a scale whose maximum is 1. The colour ramp is deliberately
scaled to 0.11–0.19 so that the within-corpus variation is visible; the subtitle
supplies the absolute scale so the reader is not misled into reading the darkest
cell as "far apart".

Direct cell labels also mean the figure survives greyscale printing and a
colour-vision-deficient reader intact: the ramp is one hue, and every value is
written out.

---

## Summary

Six figures, one claim each, all from the `train` split:

1. **fig01** — imbalance is 9.4:1; the three splits agree (TVD 0.015).
2. **fig02** — median 17 words, p99 52, max 66; 128 tokens suffices.
3. **fig03** — length carries no class signal (2.34 words spread vs 10.99 σ).
4. **fig04** — Zipfian (slope −1.29, R² 0.97); 49 types cover half the tokens;
   51% of types are hapax.
5. **fig05** — classes separate on emotion adjectives alone.
6. **fig06** — all pairs within 0.118–0.189 bits; the shared frame dominates.

The palette was validated by script, not by eye; the aqua contrast warning was
resolved by direct labels under the relief rule; every figure was rendered and
inspected, which caught three defects invisible to code review.
