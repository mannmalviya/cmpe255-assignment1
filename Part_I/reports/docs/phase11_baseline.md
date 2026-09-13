# Phase 11 — Naive Baselines

CRISP-DM Phase 11 of 18. **The first models in the study.**

**SPLIT USED:** fitted on `train_clean` (15,923 rows); scored **once** on `test`
(2,000 rows). `val` is unused — these models have no hyper-parameters.

Artefacts: `src/evaluate.py` (the shared harness), `src/phase11_baseline.py`,
`reports/metrics/phase11_baseline.json`,
`data/processed/models/baseline_{majority,stratified}.joblib`.

---

## 11.1 The hard rule this phase satisfies

> **Baseline first. No model may be reported before the naive baseline is on the
> table.**

Without it, a macro F1 of 0.60 has no meaning. The baseline converts every later
number from an absolute into a comparison, and it is the only way to say what
"learning something" means on this dataset.

## 11.2 The shared harness

This phase also builds `src/evaluate.py`, through which **every** model in the
study is scored — baselines, classical, zero-shot, few-shot, fine-tuned — so the
Phase 16 table compares like with like. Two decisions in it change the numbers and
are stated rather than buried:

**Latency is measured end to end**, from a raw text string to a predicted label
string. For a TF-IDF model that includes vectorisation; for an LLM it will include
tokenisation, generation and label parsing. A latency figure that excludes
pre-processing measures a component nobody deploys.

**Timing runs cycle through real held-out rows** rather than repeating one string,
so a single cache-warm input cannot flatter the result. Warm-up runs are executed
and discarded, so lazy imports, allocator growth and cold caches do not land in the
reported distribution.

## 11.3 Why two baselines

They fail in opposite directions, and the pair brackets "no skill" better than
either alone.

**Majority** always predicts the most frequent training class. Among constant
predictors it *maximises* plain accuracy and *minimises* macro F1 — precisely the
failure mode Phase 1 chose macro F1 to expose.

**Stratified** samples a label at random from the training class distribution. It
predicts every class, so macro F1 is non-zero, but it is uninformative by
construction.

A model that does not beat **both**, on **both** metrics, has learned nothing.

## 11.4 Results

Scored on `test` (2,000 rows), once.

| | Majority | Stratified |
|---|---:|---:|
| **Accuracy** | **0.3475** | 0.2310 |
| **Macro F1** | 0.0860 | **0.1539** |
| Weighted F1 | 0.1792 | 0.2385 |
| Accuracy − macro F1 | **+0.2615** | +0.0771 |
| Classes never predicted | **5 of 6** | none |
| Meets 0.85 target | No | No |

Per-class F1:

| Class | Majority | Stratified |
|---|---:|---:|
| joy | 0.516 | 0.336 |
| sadness | 0.000 | 0.276 |
| anger | 0.000 | 0.143 |
| fear | 0.000 | 0.125 |
| love | 0.000 | 0.043 |
| **surprise** | **0.000** | **0.000** |

### The two metrics disagree, and that is the point

**Majority wins on accuracy (0.3475 vs 0.2310). Stratified wins on macro F1
(0.1539 vs 0.0860).** The ranking flips depending on which metric you read.

This is the Phase 1 argument made concrete. A stakeholder shown only accuracy would
prefer the model that predicts a single word for every input and is silent about
five of six emotions. Macro F1 refuses that trade: majority's 0.086 is close to the
arithmetic floor, because five of its six per-class F1 scores are exactly zero.

The **gap** between the metrics is itself the diagnostic. Majority's +0.2615 gap is
the signature of a model buying head-class performance with the entire tail.
Stratified's +0.0771 is what an honest-but-useless model looks like. Every later
model's gap is read against these two.

### `surprise` scores zero even when it is predicted

The stratified baseline **did** predict `surprise` — 87 times out of 2,000, close
to its 3.5% training rate. **Not one of those 87 landed on an actual `surprise`
row**, so precision, recall and F1 are all exactly 0.

This is not a defect. With 66 true `surprise` rows in 2,000, 87 random guesses are
expected to hit about 2.9, and hitting zero has probability ≈ (1 − 0.033)⁸⁷ ≈ 5%.
Unlucky, but ordinary.

It illustrates something the earlier phases predicted from three directions. Phase 3
found `surprise` the most lexically distinctive class; Phase 7 found it the worst
under kNN (19.6%); Phase 10 found it produced the highest-lift rules. **And it still
scores zero here, because there are only 66 of it.** Random guessing cannot reach a
3.3% class, and neither can a model that is not specifically pushed to. This is why
`class_weight="balanced"` enters the Phase 12 grid and why macro F1 is the
selection criterion.

## 11.5 The latency and cost floor

A baseline does no work on the text, so its timing is the **irreducible overhead of
the harness itself**. Every later model's latency should be read as an increment
above this floor, not as an absolute.

| | Majority | Stratified |
|---|---:|---:|
| p50 single-row | 0.121 ms | 0.240 ms |
| **p95 single-row** | **0.155 ms** | 0.253 ms |
| Best throughput | 1,039,780 rows/s | 516,944 rows/s |
| Best batch size | 128 | 128 |
| Cost / 1,000 predictions | **$5.13 × 10⁻⁸** | $1.03 × 10⁻⁷ |
| Model on disk | 0.001 MB | 0.001 MB |
| Protocol | 20 warm-up discarded, 200 timed runs | same |

Both sit **over 190× inside** the 50 ms p95 budget (majority 323×) from Phase 1. Stratified is twice as
slow as majority purely because sampling from a distribution costs more than
returning a constant.

Two notes on reading these figures honestly. The cost model is reported to four
significant figures rather than fixed decimals, because costs in this study will
span many orders of magnitude and fixed rounding would print the cheap models as
`$0.00`. And a throughput of one million rows per second is not a claim about
useful work — it is the cost of calling a Python function that ignores its
argument.

*Timings above are the values in `reports/metrics/phase11_baseline.json`, from the final run of this phase. An earlier draft quoted a previous run (p95 0.130 ms); the difference is run-to-run timer noise at sub-millisecond scale and changes no conclusion.*

## 11.6 The floor, recorded

```
accuracy floor                 0.3475      (majority)
macro F1 floor                 0.1539      (stratified)
p95 latency floor              0.155 ms
throughput ceiling             1,039,780 rows/s
cost floor                     $5.13e-08 / 1,000
```

Every subsequent model is judged against this. The Phase 1 decision rule now has
its lower anchor: the target is macro F1 ≥ 0.85, the floor is 0.1539, and the
question for Phases 12–15 is how much of that 0.70 gap each family closes, and what
each charges to do it.

### What we already know about the gap

Two measurements from earlier phases sit between the floor and the target, and both
were obtained without fitting a classifier:

| | Macro F1 | Accuracy |
|---|---:|---:|
| Majority baseline | 0.086 | 0.348 |
| Stratified baseline | 0.154 | 0.231 |
| *Phase 7: kNN on frozen embeddings, k=10* | — | *0.658* |
| *Phase 8: KMeans k=6 with oracle label mapping* | — | *0.430* |
| **Phase 1 target** | **0.85** | — |

The kNN figure is the informative one: a parameter-free method on a pre-trained
representation already reaches 65.8% accuracy. So the interesting territory for
Phases 12–15 is not "beat 34.75%" — that is easy — but "approach 0.85 macro F1
without paying LLM prices."

---

## Summary of findings

1. **The hard rule is satisfied**: naive baselines are on the table before any other
   model.
2. `src/evaluate.py` built as the shared harness for every model in the study —
   end-to-end latency from raw text to label, warm-up discarded, 200+ timed runs
   cycling real held-out rows.
3. **Majority baseline: accuracy 0.3475, macro F1 0.0860.** Predicts `joy` for
   everything; five of six per-class F1 scores are exactly zero.
4. **Stratified baseline: accuracy 0.2310, macro F1 0.1539.** Predicts all six
   classes.
5. **The two metrics rank the baselines oppositely.** Majority wins accuracy,
   stratified wins macro F1. This is the Phase 1 argument for macro F1, made
   concrete on real numbers.
6. The accuracy − macro F1 **gap** is a diagnostic: +0.2615 for majority (buying the
   head with the tail), +0.0771 for stratified.
7. **`surprise` scores F1 = 0.000 under both baselines**, including the one that
   predicted it 87 times and hit zero of 66 true rows. Random guessing cannot reach
   a 3.3% class.
8. **Latency floor: p95 0.155 ms**, 323× inside the 50 ms budget. Throughput ceiling
   1.03 M rows/s. Cost floor $5.13 × 10⁻⁸ per 1,000. These are harness overhead with
   no work done.
9. The gap Phases 12–15 must close is **0.154 → 0.85 macro F1**. Phase 7's
   parameter-free kNN already reached 65.8% accuracy, so the real question is
   reaching the target *cheaply*, not beating the floor.
