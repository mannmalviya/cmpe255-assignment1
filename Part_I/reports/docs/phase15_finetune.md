# Phase 15 — Fine-Tuned Transformer

CRISP-DM Phase 15 of 18.

**SPLITS USED.** *Training:* `train_clean` (15,923 rows). *Early stopping:* `val` —
the epoch with the best val macro F1 is kept. *Scoring:* `test`, **once**, with that
checkpoint. Two follow-up analyses (§15.5, §15.6) use only test predictions produced
under this identical protocol. **Neither makes any choice**, and the reported model
remains seed 42.

**Devices.** Training, GPU inference and GPU latency: **`cuda:0`**, NVIDIA GeForce RTX
4060 Laptop GPU, CUDA 12.8, PyTorch 2.11.0+cu128. CPU latency: **`cpu`**, AMD Ryzen 7
7435HS, 8 PyTorch threads, measured in a **fresh process**. Each device has its own
metrics file with its own hardware stamp: `phase15_finetune_gpu.json` and
`phase15_finetune_cpu.json`.

Artefacts: `src/phase15_finetune.py`, `src/phase15_significance.py`,
`reports/metrics/phase15_{finetune_gpu,finetune_cpu,seed_check,significance}.json`,
`data/processed/models/distilbert_finetuned/`.

---

## 15.1 Why this model, and what it isolates

Phases 13 and 14 lost for the same reason. The model never learned this dataset's
label conventions from enough examples — zero examples, then twelve noisy ones. The
linear SVM succeeded by learning from all 17,923 labelled development rows.

Fine-tuning removes exactly that cause: the pre-trained model learns from the whole
training split. It is the fair test of the LLM family's best case on this task.

**`distilbert-base-uncased`, 66.96 M parameters**, a 6-layer transformer encoder with a
6-way classification head. Three reasons for the choice:

- **An encoder outputs a class directly.** There is no generation, so there are no
  unparseable outputs — the failure that cost Phase 13 7% of its rows.
- **It trains in minutes** on the 8 GB GPU.
- **It is small enough that CPU serving is realistic.** That makes the CPU latency a
  real deployment option, not a curiosity.

A note on terminology. DistilBERT is a small open-weight *language model*, but not a
generative instruct *LLM* like Qwen in Phases 13–14. The research question asks about
"a small open-weight LLM", and the brief for this phase allowed "a small open-weight
encoder or decoder". The final report keeps the two families distinct: prompted
decoders (Phases 13–14) and a fine-tuned encoder (Phase 15). They behave very
differently, and merging them under one label would hide that.

## 15.2 Design, fixed before training

| Setting | Value | Reason |
|---|---|---|
| max length | 128 tokens | Phase 7: longest document is 87 WordPiece tokens — no truncation |
| batch | 32, dynamic padding | each batch pads only to its own longest row |
| optimiser | AdamW, lr 5×10⁻⁵, weight decay 0.01 | standard DistilBERT fine-tuning values |
| schedule | linear, 10% warm-up | |
| epochs | max 4, keep best val macro F1 | early stopping on val, never on test |
| loss | cross-entropy with **balanced class weights** from `train_clean` | Phase 12: balanced weighting was worth +2.8 macro F1 |
| precision | bf16 autocast on GPU; fp32 on CPU | |
| gradient clipping | 1.0 | |
| seed | 42 | |

No setting was changed after seeing a result.

## 15.3 Training

| Epoch | Train loss | Val macro F1 | Val accuracy | Seconds |
|---:|---:|---:|---:|---:|
| 1 | 0.7136 | 0.9032 | 0.9270 | 33.7 |
| 2 | 0.1685 | 0.9093 | 0.9330 | 32.1 |
| **3** | 0.1191 | **0.9163** | **0.9390** | 33.1 |
| 4 | 0.0781 | 0.9161 | 0.9390 | 33.2 |

**Best epoch: 3**, kept by val macro F1. Total training time **137 seconds**, peak GPU
memory **1,708 MB**.

**After a single epoch, val macro F1 is already 0.9032** — above the linear SVM's
cross-validated 0.8677, and above the 0.85 target. The pre-trained representation is
doing most of the work. Epochs 2–4 add only 1.3 points while training loss keeps
falling, which is the usual sign of approaching over-fitting, and the reason for
keeping the best val epoch rather than the last.

## 15.4 Results

Test, scored once, seed-42 checkpoint:

| | Fine-tuned DistilBERT | Linear SVM | Zero-shot Qwen | Few-shot Qwen |
|---|---:|---:|---:|---:|
| **Macro F1** | **0.8978** | 0.8579 | 0.4107 | 0.3427 |
| Accuracy | **0.9310** | 0.8995 | 0.4975 | 0.4750 |
| Errors (of 2,000) | **138** | 201 | 1,005 | 1,050 |
| Meets 0.85 target | Yes | Yes | No | No |
| Unparseable outputs | n/a (classifier) | n/a | 143 | 19 |

Per-class F1:

| Class | Train rows | DistilBERT | Linear SVM | Gain |
|---|---:|---:|---:|---:|
| sadness | 4,661 | **0.970** | 0.935 | +0.035 |
| joy | 5,340 | **0.948** | 0.927 | +0.021 |
| anger | 2,152 | **0.932** | 0.896 | +0.036 |
| fear | 1,923 | **0.888** | 0.864 | +0.024 |
| love | 1,283 | **0.853** | 0.811 | +0.042 |
| surprise | 564 | **0.795** | 0.714 | **+0.081** |

**The fine-tuned model beats the SVM on every class.** The largest gain is on
`surprise`, the smallest and hardest class: +8.1 points. Class size still orders the
per-class scores, but the gap between best and worst class has narrowed from 22.1
points under the SVM to 17.5.

**CPU and GPU give identical predictions.** The fp32 CPU run and the bf16 GPU run agree
on **all 2,000 test rows** (agreement 1.0000). So the reduced precision on GPU costs
nothing here.

### Where the remaining errors are

Confusion matrix (rows = true, columns = predicted):

|  | joy | sadness | anger | fear | love | surprise |
|---|---:|---:|---:|---:|---:|---:|
| **joy** | 634 | 2 | 4 | 0 | **48** | 7 |
| **sadness** | 3 | 554 | 17 | 7 | 0 | 0 |
| **anger** | 2 | 2 | 269 | 2 | 0 | 0 |
| **fear** | 0 | 3 | 10 | 187 | 0 | **24** |
| **love** | 4 | 0 | 1 | 0 | 154 | 0 |
| **surprise** | 0 | 0 | 1 | 1 | 0 | 64 |

| Confusion pair | DistilBERT errors | Share | SVM share | Annotator enrichment (Phase 5) |
|---|---:|---:|---:|---:|
| **joy ↔ love** | 52 | **37.7%** | 29.4% | 7.8× |
| **fear ↔ surprise** | 25 | **18.1%** | 10.9% | 12.3× |
| sadness ↔ anger | 19 | 13.8% | 15.4% | 0.8× |
| anger ↔ fear | 12 | 8.7% | 8.5% | 2.3× |

**As the model improves, its errors concentrate on the two pairs annotators
themselves disagree about.** Joy↔love and fear↔surprise are **55.8%** of DistilBERT's
errors, against 40.3% of the SVM's. The model has removed much of the avoidable error.
What remains is mostly the boundary Phase 5 showed humans draw inconsistently.

The direction of these errors is informative too. `love` recall is **96.9%** (154 of
159) and `surprise` recall is **97.0%** (64 of 66). The balanced class weights push the
model to find almost every tail-class row. Its remaining mistakes are **false
positives pulled from the large neighbouring class**: 48 `joy` rows called `love`, and
24 `fear` rows called `surprise`.

## 15.5 Is the margin real? — significance

The Phase 1 rule demands a margin of **at least 3 macro F1 points**. The observed
margin is **3.99**, close to the line, from one test set of 2,000 rows. So two paired
tests were run on the saved test predictions. **Nothing was refitted or re-selected.**

**Paired bootstrap on macro F1** (10,000 resamples; both models scored on the same
resampled rows each time):

| | |
|---|---:|
| Observed margin | **+0.0399** |
| 95% confidence interval | **[+0.0214, +0.0591]** |
| P(margin > 0) | **1.000** |
| P(margin ≥ 0.030) | **0.849** |
| CI lower bound ≥ 0.030? | **No** |

**McNemar's exact test** on per-row correctness:

| | SVM right | SVM wrong |
|---|---:|---:|
| **DistilBERT right** | 1,752 | **110** |
| **DistilBERT wrong** | **47** | 91 |

Exact p = **5.3 × 10⁻⁷**.

Two statements follow, and they should not be confused.

**That DistilBERT is better than the SVM is established.** Every one of 10,000
bootstrap resamples favours it, and McNemar rejects equal accuracy at p < 10⁻⁶. It
fixes 110 rows the SVM gets wrong and breaks only 47.

**That it is better by at least 3 points is likely, not established.** 85% of resamples
clear the threshold, but the 95% interval reaches down to +2.1 points.

## 15.6 Seed variance

The headline model is one training run. Two further seeds were trained under the
**identical** protocol (train on `train_clean`, best epoch on `val`) and scored on test.
**No choice was made from these runs.** They show only how far the result moves under a
different random initialisation and data order.

| Seed | Best epoch | Val macro F1 | **Test macro F1** | Test accuracy | Margin over SVM |
|---:|---:|---:|---:|---:|---:|
| **42** (reported) | 3 | 0.9163 | **0.8978** | 0.9310 | **+0.0399** |
| 7 | 4 | 0.9167 | 0.8895 | 0.9255 | +0.0316 |
| 123 | 4 | 0.9154 | 0.8913 | 0.9310 | +0.0334 |
| **Mean ± SD** | | | **0.8929 ± 0.0044** | | **+0.035** |

**Every seed beats the SVM, and every seed clears the 3-point threshold on its point
estimate.** But two of three clear it only narrowly (+3.2 and +3.3), and **the reported
seed-42 run is the most favourable of the three.** The mean margin, +3.5 points, is the
more representative figure, and it is the one Phase 16 should carry alongside the
seed-42 headline.

Two further observations, recorded rather than acted on:

- **Seeds 7 and 123 both selected the final epoch (4)**, with val F1 still rising. The
  4-epoch cap may have been binding for them, so a longer schedule might add a little.
  It was not tried: the design was fixed, and extending it after seeing test results
  would be tuning.
- **The val→test gap is about 2.3 points** (val ≈ 0.916, test ≈ 0.893), against about 1
  point for the SVM's CV→test gap. Early stopping on a single 2,000-row val split selects
  a slightly optimistic epoch. Test, which no choice touched, is the trustworthy number.

## 15.7 Cost and latency

End to end from raw text to label, through the shared harness: 20 warm-up runs
discarded, 200 timed single-row runs, then batched throughput. GPU timings call
`torch.cuda.synchronize()` so they measure finished work, not queued kernels.

| | DistilBERT **GPU** | DistilBERT **CPU** | Linear SVM (CPU) | Zero-shot Qwen (GPU) |
|---|---:|---:|---:|---:|
| Device | cuda:0 | cpu, 8 threads | cpu | cuda:0 |
| p50 single-row | **3.01 ms** | 14.04 ms | 0.220 ms | 43.8 ms |
| **p95 single-row** | **3.51 ms** | **18.26 ms** | 0.246 ms | 61.2 ms |
| p99 single-row | 3.63 ms | 20.16 ms | — | 70.2 ms |
| Within 50 ms p95 budget | **Yes** | **Yes** | Yes | No |
| Best throughput | **2,414 rows/s** (batch 32) | 133 rows/s (batch 8) | 104,222 rows/s | 49.7 rows/s |
| Cost / 1,000 predictions | **$6.90 × 10⁻⁵** | $4.02 × 10⁻⁴ | $5.12 × 10⁻⁷ | $3.36 × 10⁻³ |
| **Cost / month at 1 M** | **$0.069** | **$0.40** | $0.0005 | $3.36 |
| Model on disk | 256 MB | 256 MB | 0.41 MB | 2,955 MB |
| Peak memory | 1,485 MB GPU | 1,498 MB RSS | — | 4,540 MB GPU |
| Training | 137 s, 1.7 GB GPU | — | 20 s CV search | none |

Throughput by batch size:

| Batch | 1 | 8 | 32 | 128 |
|---|---:|---:|---:|---:|
| GPU rows/s | 358 | 1,865 | **2,414** | 2,205 |
| CPU rows/s | 80 | **133** | 109 | 90 |

**Relative to the SVM, the fine-tuned model is expensive.** On GPU it is 14× slower at
p95, 135× costlier per prediction, and 624× larger on disk. On CPU it is 74× slower and
785× costlier.

**In absolute terms, it is cheap.** The deciding fact is the scale Phase 1 fixed: at 1 M
messages a month, the fine-tuned model costs **$0.07 on GPU or $0.40 on CPU**, against
the SVM's $0.0005. Phase 1 §1.2 set the reading rule for this: a difference of $0.10 per
1,000 predictions is small, and $10 per 1,000 is decisive. The difference here is about
$0.0004 per 1,000 on CPU. **The ratio is enormous; the dollars are negligible.**

**Both devices are well inside the latency budget**: p95 3.5 ms on GPU, 18.3 ms on CPU,
against a 50 ms budget. The CPU figure matters most for deployment, because it means the
model can be served on the same commodity CPU instance as the SVM, with no accelerator.

The comparison with the prompted decoder is stark. The fine-tuned 67 M-parameter
encoder is **17× faster at p95 on the same GPU** as the 1.5 B-parameter zero-shot Qwen,
**49× cheaper per prediction**, **11.5× smaller on disk**, and **48.7 macro F1 points more
accurate**. Size and generality did not buy accuracy on this task. Task-specific
training did.

## 15.8 The verdict

The Phase 1 rule, applied as it was written, before any result existed:

| Criterion | Required | DistilBERT (seed 42) | Pass |
|---|---|---:|---|
| Macro F1 margin over best classical | ≥ +0.030 | **+0.0399** | **Yes** |
| p95 latency, GPU | ≤ 50 ms | 3.51 ms | **Yes** |
| p95 latency, CPU | ≤ 50 ms | 18.26 ms | **Yes** |
| **Worth it (rule as written)** | both | — | **Yes** |

**Under the pre-registered decision rule, the fine-tuned model is worth it.** It is the
only model in the study that passes. The rule compares point estimates, and it is
applied as written. It is not tightened now that the result is known, because moving
the goalposts after seeing the result is exactly what pre-registration exists to prevent.

The honest qualifications, which belong beside the verdict and not beneath it:

1. **"Better" is certain; "better by ≥ 3 points" is probable.** McNemar p = 5.3 × 10⁻⁷,
   but P(margin ≥ 0.03) = 0.85, and the 95% interval's lower bound is +2.1 points.
2. **The reported seed is the most favourable of three.** Across seeds the mean margin is
   +3.5 points (range +3.2 to +4.0). All three clear the threshold, two narrowly.
3. **The win comes from fine-tuning, not from generative capability.** The two prompted
   decoder configurations lost by 45 and 52 points. What succeeds is a small pre-trained
   *encoder* trained on the full labelled set, and "LLM" should not be read as covering
   both.
4. **2.8% of test rows are near-copies or contradicted copies of training rows**
   (Phases 5 and 9). Measured per row in Phase 17, their net effect on reported
   accuracy is about −0.3 points for both DistilBERT and the SVM, so the comparison
   between models is unaffected. *(An earlier estimate of "1.4 points" was wrong; see
   the Phase 9 correction.)*
5. **The margin is bounded by annotation ambiguity.** 55.8% of the remaining errors sit on
   the two label pairs humans disagree on, so further accuracy gains will be hard for any
   model to buy.

Phase 16 places every model in one table and tests the deployment-level reading of this
verdict. Phase 17 examines the errors, including the genre-B rows (Phase 6) where a
pre-trained model was predicted to have its structural advantage.

---

## Summary of findings

1. **Fine-tuned DistilBERT (67 M parameters): test macro F1 0.8978, accuracy 0.9310** —
   the best model in the study, beating the linear SVM (0.8579) **on every class**.
2. **Largest gain on `surprise`, the hardest class: +8.1 points** (0.714 → 0.795). The
   best-to-worst class gap narrows from 22.1 to 17.5 points.
3. **One epoch already reaches val macro F1 0.9032**. The pre-trained representation does
   most of the work; epoch 3 was kept by val.
4. **Errors concentrate on annotator-ambiguous pairs**: joy↔love 37.7% and fear↔surprise
   18.1% — **55.8% of all errors**, against 40.3% for the SVM. `love` and `surprise` recall
   are 96.9% and 97.0%; the remaining tail errors are false positives from larger
   neighbours.
5. **CPU fp32 and GPU bf16 predictions agree on all 2,000 rows.**
6. **Significance: better is established** (McNemar p = 5.3 × 10⁻⁷; 110 rows fixed vs 47
   broken; P(margin > 0) = 1.000). **Better by ≥ 3 points is probable, not established**
   (95% CI [+2.1, +5.9]; P = 0.849).
7. **Seed variance: 0.8929 ± 0.0044 across three seeds; every seed clears +3 points**, two
   narrowly. **Seed 42 is the most favourable**; the mean margin is +3.5.
8. Two seeds chose the final epoch, so the 4-epoch cap may have been binding. Not
   extended, because the design was fixed. The val→test gap (~2.3 points) exceeds the SVM's.
9. **p95 latency 3.51 ms on GPU and 18.26 ms on CPU — both inside the 50 ms budget.**
   Throughput 2,414 rows/s (GPU) and 133 rows/s (CPU).
10. **$0.069/month (GPU) or $0.40/month (CPU) at 1 M messages**, against $0.0005 for the
    SVM: **135–785× the ratio, negligible in dollars**. 256 MB on disk; ~1.5 GB peak memory.
11. **Against zero-shot Qwen on the same GPU: 17× faster, 49× cheaper, 11.5× smaller, and
    48.7 macro F1 points more accurate.** Task-specific training beat size and generality.
12. **Verdict under the pre-registered rule: worth it** — the only model in the study to
    pass. It is applied as written, and reported with its qualifications: a margin that is
    probable rather than certain, a favourable seed, and a win that belongs to a
    fine-tuned encoder, not to a prompted generative LLM.
