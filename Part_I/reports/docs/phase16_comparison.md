# Phase 16 — Head-to-Head Comparison

CRISP-DM Phase 16 of 18.

**SPLIT USED: none new.** Every number below is read from `reports/metrics/*.json` —
test-split results produced in Phases 11–15. Nothing was re-run, re-scored, re-fitted
or re-selected.

Artefacts: `src/phase16_comparison.py`, `reports/metrics/phase16_comparison.json`,
`reports/metrics/phase16_comparison_table.csv`,
`reports/docs/phase16_tables_generated.md` (the three tables, machine-written),
`reports/figures/fig13_tradeoff.png`.

---

## 16.1 How the table was built

The brief required that "the final table is built from files, not from memory".
`src/phase16_comparison.py` enforces this literally. It loads the metrics JSON files,
and the three tables in §16.3 are **written by the script** to
`phase16_tables_generated.md`, then reproduced here. No number was typed by hand.

Building it also exposed one inconsistency. The Phase 11 write-up quoted baseline
timings from an earlier run (p95 0.130 ms), while its JSON held the final run
(0.155 ms). The document has been corrected to match the file. No conclusion
changed. The Phase 12–15 write-ups were checked against their JSON files and match.

## 16.2 Two audits, run before any table

Either audit failing would stop the script, and it would print no table.

### Hardware audit — passed

Every metrics file carries a hardware block. Its stable fingerprint was compared with
the reference frozen in Phase 1:

> `AMD Ryzen 7 7435HS | 16 logical cores | 15.3 GB RAM | NVIDIA GeForce RTX 4060 Laptop GPU | 8,188 MiB | driver 580.126.09`

**All 19 metrics files match.** Every timing in this study came from one machine, so
the rule "timings from different hardware must never be compared in the same table"
is satisfied by construction.

The PyTorch version is deliberately **not** part of the fingerprint, and it did change
during the study. It is disclosed here:

| PyTorch | Files | Timings affected |
|---|---|---|
| none installed | Phases 2–6 | none — no timed models |
| 2.14.0+cpu | Phases 7–12 | none — baseline and classical models do not use PyTorch; Phase 7 embeddings were not timed |
| 2.11.0+cu128 | Phases 13–16 | **all PyTorch-dependent timings**: Qwen zero-shot, Qwen few-shot, DistilBERT GPU and CPU |

**Every model that uses PyTorch was timed under the same version (2.11.0+cu128).** The
version change cannot bias a comparison between those models. The classical models'
timings do not touch PyTorch at all.

### Protocol audit — passed

All **9 timing rows** report **≥ 200 timed single-row runs** after **≥ 20 discarded
warm-up runs**, and each names the device it actually ran on.

## 16.3 The comparison

### How to read it

- **Table A (quality)** is device-independent. The fine-tuned model's CPU (fp32) and GPU
  (bf16) runs gave identical predictions on all 2,000 test rows, so it appears once.
- **Tables B and C (latency, throughput, cost, footprint)** are split by the device the
  model **actually ran on**. CPU and GPU timings never share a column. The cost column
  already accounts for the device, because each row is priced at its own tier's hourly
  rate (Phase 1 §1.5: CPU $0.192/h, GPU $0.60/h).
- **The decision bar** is the best classical macro F1 (linear SVM, 0.8579) **+ 0.03 =
  0.8879**, together with p95 ≤ 50 ms. Both were fixed in Phase 1, before any result.

### Table A — Quality on the test split

| Model | Family | Macro F1 | Accuracy | Acc − macro F1 | F1 joy | F1 sadness | F1 anger | F1 fear | F1 love | F1 surprise | Clears 0.85 | Clears bar ≥ 0.8879 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Majority class | baseline | **0.0860** | 0.3475 | +0.2615 | 0.516 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | no | no |
| Stratified random | baseline | **0.1539** | 0.2310 | +0.0771 | 0.336 | 0.276 | 0.143 | 0.125 | 0.043 | 0.000 | no | no |
| TF-IDF + logistic regression | classical | **0.8486** | 0.8905 | +0.0419 | 0.916 | 0.929 | 0.890 | 0.860 | 0.791 | 0.706 | no | no |
| TF-IDF + linear SVM | classical | **0.8579** | 0.8995 | +0.0416 | 0.927 | 0.935 | 0.896 | 0.864 | 0.811 | 0.714 | yes | no |
| TF-IDF + LightGBM | classical | **0.8400** | 0.8810 | +0.0410 | 0.901 | 0.926 | 0.873 | 0.863 | 0.742 | 0.736 | no | no |
| Qwen2.5-1.5B zero-shot | prompted LLM | **0.4107** | 0.4975 | +0.0868 | 0.567 | 0.625 | 0.501 | 0.322 | 0.295 | 0.154 | no | no |
| Qwen2.5-1.5B few-shot (12) | prompted LLM | **0.3427** | 0.4750 | +0.1323 | 0.638 | 0.492 | 0.438 | 0.085 | 0.346 | 0.058 | no | no |
| DistilBERT fine-tuned | fine-tuned encoder | **0.8978** | 0.9310 | +0.0332 | 0.948 | 0.970 | 0.932 | 0.888 | 0.853 | 0.795 | yes | **yes** |

Uncertainty on the fine-tuned row (Phase 15): test macro F1 across three training seeds
**0.8929 ± 0.0044** (range 0.8895–0.8978). Paired-bootstrap 95% CI on its margin over
the SVM **[+0.0214, +0.0591]**. P(margin ≥ 0.03) = **0.849**. McNemar exact p =
**5.3 × 10⁻⁷**.

### Table B — Served on CPU (`cpu`, AMD Ryzen 7 7435HS)

| Model | Macro F1 | p50 ms | p95 ms | p99 ms | Within 50 ms p95 | Best rows/s (batch) | $ / 1,000 | $ / month at 1 M | Disk MB | Peak memory MB† |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Majority class | 0.0860 | 0.121 | 0.155 | 0.185 | yes | 1,039,779.8 (128) | $5.13e-08 | $5.13e-05 | 0.001 | 374.2 |
| Stratified random | 0.1539 | 0.240 | 0.253 | 0.324 | yes | 516,944.0 (128) | $1.03e-07 | $1.03e-04 | 0.001 | 374.6 |
| TF-IDF + logistic regression | 0.8486 | 0.313 | 0.341 | 0.349 | yes | 98,489.2 (128) | $5.42e-07 | $5.41e-04 | 0.807 | 419.7 |
| **TF-IDF + linear SVM** | **0.8579** | **0.220** | **0.246** | 0.260 | yes | **104,221.7** (128) | **$5.12e-07** | **$5.12e-04** | **0.411** | 419.7 |
| TF-IDF + LightGBM | 0.8400 | 0.665 | 0.800 | 0.929 | yes | 20,780.6 (128) | $2.57e-06 | $2.57e-03 | 2.690 | 446.7 |
| **DistilBERT fine-tuned (CPU)** | **0.8978** | 14.036 | **18.257** | 20.155 | yes | 132.8 (8) | $4.02e-04 | **$0.4015** | 256.500 | 1497.8 |

### Table C — Served on GPU (`cuda:0`, NVIDIA GeForce RTX 4060 Laptop GPU)

| Model | Macro F1 | p50 ms | p95 ms | p99 ms | Within 50 ms p95 | Best rows/s (batch) | $ / 1,000 | $ / month at 1 M | Disk MB | Peak memory MB† |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B zero-shot | 0.4107 | 43.847 | 61.187 | 70.181 | **no** | 49.7 (32) | $3.35e-03 | $3.3550 | 2,955.400 | 4539.7 |
| Qwen2.5-1.5B few-shot (12) | 0.3427 | 91.062 | 109.713 | 112.210 | **no** | 13.7 (8) | $0.0122 | $12.1800 | 2,955.400 | 6799.7 |
| **DistilBERT fine-tuned (GPU)** | **0.8978** | 3.006 | **3.508** | 3.629 | yes | **2,414.5** (32) | $6.90e-05 | **$0.0690** | 256.300 | 1484.7 |

† **Peak memory is not the same quantity across the two tables**, and must not be
compared across them. Table C reports PyTorch's peak *GPU allocation*. Table B reports
peak *process RSS*. The DistilBERT CPU figure comes from a fresh single-purpose
process. The baseline and classical figures come from processes that also loaded data
and grid-search checkpoints, so they **overstate** those models' serving memory. That
bias runs *against* the classical models, and so cannot flatter them.

## 16.4 The decision rule, applied to every LLM configuration

| Configuration | Device | Margin vs SVM | Clears F1 bar | p95 ms | Within budget | Cost × SVM | Extra $/month at 1 M | **Worth it** |
|---|---|---:|---|---:|---|---:|---:|---|
| Qwen2.5-1.5B zero-shot | cuda:0 | **−0.4472** | no | 61.19 | no | 6,557× | +$3.35 | **No** |
| Qwen2.5-1.5B few-shot (12) | cuda:0 | **−0.5152** | no | 109.71 | no | 23,803× | +$12.18 | **No** |
| DistilBERT fine-tuned | cuda:0 | **+0.0399** | yes | 3.51 | yes | 135× | +$0.069 | **Yes** |
| DistilBERT fine-tuned | cpu | **+0.0399** | yes | 18.26 | yes | 785× | +$0.40 | **Yes** |

**Of four LLM configurations, two pass and two fail, and the split falls exactly along
one line: whether the model was trained on the labelled data.** Both prompted
configurations of a 1.5 B-parameter generative model fail on accuracy *and* on the
latency budget. The fine-tuned 67 M-parameter encoder passes on both devices.

## 16.5 What the figure shows

`fig13_tradeoff.png` plots every model twice: macro F1 against cost, and macro F1
against p95 latency. The top row shows the full range; the bottom row zooms on the
contenders.

**Panels a–b** show the shape of the study in one glance. The prompted LLMs sit in the
**worst corner** of both plots — the most expensive and slowest points, and far below
every trained model. Their extra cost buys nothing. Both lie outside the 50 ms budget.

**Panels c–d** show the decision itself. On a 0–1 axis DistilBERT would sit on top of
the bar, so these panels zoom to 0.83–0.91. DistilBERT is the only model above the
dashed bar. The vertical bar on its points spans its three training seeds: **even its
worst seed (0.8895) clears the bar, but by only 0.16 points.** The three classical
models sit between 1 and 5 points below the bar.

## 16.6 The efficient frontier

A model is **Pareto-optimal** if no other model is at least as accurate *and* at least
as cheap, and strictly better on one of the two. Computed from the table:

| Trade-off | Pareto-optimal models, cheapest first |
|---|---|
| Macro F1 vs cost, any device | majority → stratified → **linear SVM** → **DistilBERT (GPU)** |
| Macro F1 vs p95 latency, any device | majority → **linear SVM** → **DistilBERT (GPU)** |
| Macro F1 vs cost, **CPU only** | majority → stratified → **linear SVM** → **DistilBERT (CPU)** |

Every other model is **dominated** — something else is both more accurate and cheaper:

- **Logistic regression and LightGBM** are dominated by the linear SVM, on every axis.
- **Both Qwen configurations** are dominated by every trained model on both axes. They
  are the most expensive models in the study and the least accurate models above the
  baselines.
- **DistilBERT on CPU** is dominated by DistilBERT on GPU when a GPU is available. It
  rejoins the frontier when deployment is restricted to CPU.

**For any real deployment, only two models are worth considering: the linear SVM and
the fine-tuned DistilBERT.** The choice between them is the choice this study exists to
inform.

## 16.7 Reading the cost at deployment scale

Ratios make the fine-tuned model look expensive (135× to 785× the SVM per prediction).
Phase 1 §1.2 fixed the scale at which such numbers must be read: 1,000,000 messages a
month.

| Moving from SVM to DistilBERT | GPU | CPU |
|---|---:|---:|
| Extra compute per month at 1 M messages | **+$0.069** | **+$0.40** |
| Accuracy gained (test) | +3.15 points | +3.15 points |
| Misclassified messages avoided per month | **31,500** | **31,500** |
| Compute cost per avoided error | **$2.2 × 10⁻⁶** | **$1.3 × 10⁻⁵** |
| Compute cost per macro-F1 point per month | $0.017 | $0.10 |

Phase 1 framed each message as a routing decision, and a misrouted `anger` or `fear`
message as an escalation failure with a real business cost. Against that, a compute
bill of **about one thousandth of a cent per avoided error** is negligible. **In compute
dollars, the fine-tuned model's gain is essentially free.**

Phase 1 §1.5 also warned that compute is not the whole cost, so these figures are a
lower bound. The fine-tuned model's real costs sit elsewhere:

| | Linear SVM | DistilBERT |
|---|---|---|
| Serving artefact | 411 KB | 256 MB (**624×**) |
| Serving memory | well under 0.5 GB process | ~1.5 GB process |
| Runtime dependency | scikit-learn | PyTorch + transformers |
| Retraining | 20 s CV search, CPU | 2–3 min, **GPU** |
| Cold start | milliseconds | seconds |

**For DistilBERT, the operational burden is the real price, not the compute bill**:
a heavier runtime, a GPU in the training pipeline, a larger artefact to version and
ship, and slower cold starts. None of these appears in cost per 1,000 predictions.

## 16.8 What Phase 16 establishes

1. **Two model families, two opposite answers.** A small *prompted* generative LLM does
   not beat classical ML. It loses by 45–52 macro F1 points, misses the latency budget,
   and costs thousands of times more. A small *fine-tuned* pre-trained encoder does beat
   it — by +4.0 points (seed 42) or +3.5 (seed mean) — inside the latency budget on both
   CPU and GPU.
2. **The fine-tuned model passes the pre-registered rule.** Its superiority over the SVM
   is statistically established. That the margin exceeds the 3-point threshold is
   probable (85%), not certain.
3. **Only the linear SVM and the fine-tuned DistilBERT sit on the efficient frontier.**
   Everything else, including both Qwen configurations, is dominated.
4. **In compute dollars at the Phase 1 scale, the fine-tuned model's gain is nearly
   free**: +$0.07–$0.40 a month for about 31,500 fewer misclassified messages. Its real
   cost is operational.
5. **The comparison is valid**: one machine for all 19 metrics files, every
   PyTorch-dependent timing under one PyTorch version, and every timing row compliant
   with the latency protocol.

Phase 17 examines *where* each family fails, with real examples, including the Phase 6
genre-B rows. Phase 18 turns these results into a recommendation.

---

## Summary of findings

1. Built entirely from the metrics JSON files. The three tables are machine-written by
   `src/phase16_comparison.py`. One stale doc (Phase 11 timings) was found and corrected
   to its file.
2. **Hardware audit passed: all 19 metrics files share one fingerprint.** PyTorch changed
   2.14 → 2.11 during the study; every PyTorch-dependent timing ran under 2.11, and
   classical timings do not use PyTorch.
3. **Protocol audit passed**: 9 timing rows, each ≥ 200 timed runs after ≥ 20 warm-ups,
   device recorded.
4. **Only fine-tuned DistilBERT clears the decision bar** (0.8978 ≥ 0.8879), and only it
   passes the rule — on GPU (p95 3.5 ms) and CPU (p95 18.3 ms).
5. **Both prompted Qwen configurations fail both criteria**: margins −0.447 and −0.515,
   p95 61 and 110 ms, 6,557× and 23,803× the SVM's cost.
6. **The pass/fail split falls exactly on whether the model was trained on the labelled
   data.**
7. **Pareto frontier: majority → (stratified) → linear SVM → fine-tuned DistilBERT.**
   Logistic regression, LightGBM and both Qwen configurations are dominated.
8. **At 1 M messages a month, moving from SVM to DistilBERT costs +$0.069 (GPU) or +$0.40
   (CPU) and avoids about 31,500 misclassifications** — about $10⁻⁵ per avoided error.
9. **The fine-tuned model's real cost is operational, not compute**: a 624× larger
   artefact, a PyTorch runtime, GPU retraining and slower cold starts.
10. Peak-memory figures differ in kind between the CPU table (process RSS) and the GPU
    table (GPU allocation), and are not compared across tables. The classical RSS figures
    overstate their serving memory, which biases against them, not for them.
