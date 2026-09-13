# Phase 14 — Few-Shot LLM

CRISP-DM Phase 14 of 18.

**SPLITS USED.** *Examples:* drawn from `train_clean` only. *Choice of shot count:*
made on `val`. *Scoring:* the chosen configuration scored **once** on `test`. A
follow-up robustness check (§14.6) used **`val` only** and did not score test again.

**Device: `cuda:0`, NVIDIA GeForce RTX 4060 Laptop GPU, CUDA 12.8, PyTorch
2.11.0+cu128** — identical to Phase 13. Full hardware block in each metrics JSON.

Artefacts: `src/phase14_fewshot.py`, `src/phase14_robustness.py`,
`reports/metrics/phase14_fewshot.json`, `reports/metrics/phase14_robustness.json`,
`data/processed/phase14_val_sweep.json`.

---

## 14.1 One change from Phase 13

Phase 13 diagnosed the zero-shot failure as a **convention problem**: the dataset's
label boundaries — above all joy versus love — are annotation conventions, and a
model that has never seen them draws its own. The direct remedy is to show the
model the convention.

So Phase 14 changes exactly one thing: **labelled examples are placed in the prompt**
as earlier user/assistant turns. The model, system prompt, greedy decoding and
strict parser are imported unchanged from `src/llm_common.py`. Any difference from
Phase 13 is attributable to the examples alone.

## 14.2 Design, fixed before any result

- **Candidate pool.** `train_clean` rows with no Phase 6 outlier flag and at most
  25 words: **10,710 rows**. Examples should show the typical convention, not an
  edge case, and short examples keep the prompt — and so the latency cost — bounded.
- **Selection.** Stratified random, seed 42, *k* examples **per class**, so every
  class, and the joy/love boundary in particular, is shown.
- **Order.** Round-robin across classes in a seeded, shuffled class order, so the
  last example — which small models over-weight — is not always the same class.
- **Choosing *k*.** Grid {1, 2, 4} per class (6, 12 or 24 examples), selected by
  **val** macro F1. Ties go to the smaller *k*, because examples cost latency.
  Choosing *k* on test would be tuning on test.

## 14.3 Two runs, and why the second exists

The first run **completed the val sweep and then crashed** with a CUDA
out-of-memory error. The crash came from the latency harness's batch-128
throughput test, not from scoring. `score_split` already caught OOM there and
halved the batch. Because results were written only at the very end, the sweep was
lost.

Two fixes. **An OOM in the throughput test is now recorded as a result** —
`"batch 128: out of memory"` — because it is a real limit on how far this
configuration can be batched on 8 GB. And **each val result is checkpointed** as it
completes.

**Integrity note.** The first run may have generated test predictions before it
crashed. If so, test was generated twice, with an identical configuration that val
had already fixed. Greedy decoding is deterministic. No choice depended on the
first pass, and its output was never seen, because it was lost with the crash. This
is not tuning on test, but it is recorded.

## 14.4 Results

### Choosing *k* on val

| Examples per class | Total examples | Prompt tokens | Val macro F1 | Val accuracy | Unparseable | Rows/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | ~243 | 0.3378 | 0.4480 | 37 | 21.9 |
| **2** | **12** | **~411** | **0.3592** | 0.4815 | 14 | 13.7 |
| 4 | 24 | ~701 | 0.3072 | 0.3790 | **0** | 8.1 |

*k* = 2 per class was chosen.

### Test, scored once

| | Few-shot (k=2) | Zero-shot (Phase 13) | Linear SVM (Phase 12) |
|---|---:|---:|---:|
| **Macro F1** | **0.3427** | 0.4107 | **0.8579** |
| Accuracy | 0.4750 | 0.4975 | 0.8995 |
| Weighted F1 | 0.4636 | 0.5122 | — |
| Balanced accuracy | 0.3921 | 0.4026 | — |
| Unparseable outputs | **19** (0.95%) | 143 (7.15%) | — |

Per-class F1:

| Class | Few-shot | Zero-shot | Change | Linear SVM |
|---|---:|---:|---:|---:|
| joy | **0.638** | 0.567 | +0.071 | 0.927 |
| sadness | 0.492 | 0.625 | −0.133 | 0.935 |
| anger | 0.438 | 0.501 | −0.063 | 0.896 |
| fear | **0.085** | 0.322 | **−0.237** | 0.864 |
| love | 0.346 | 0.295 | +0.051 | 0.811 |
| surprise | **0.058** | 0.154 | −0.096 | 0.714 |

**Few-shot prompting made the model worse.** Macro F1 fell by 6.8 points against
zero-shot, and it now trails the linear SVM by 51.5 points. The examples helped two
classes a little (`joy`, `love`) and badly damaged three (`sadness`, `fear`,
`surprise`). `fear` nearly disappeared.

## 14.5 What the examples actually taught

### They taught the format — reliably

Unparseable outputs fell from **143 to 19** on test. On val the decline was
monotonic in *k*: **37 → 14 → 0**. By 24 examples, the model never once answered
outside the six labels. The out-of-set words it used in Phase 13 (`stress`,
`anxiety`, `curiosity`) are gone. What remains is mostly `neutral` (6) and `pain`
(4). **Showing the model the answer format works.**

### They taught the labels — badly

On val, more examples gave **fewer format errors and worse classification**. At
*k* = 4, 0 unparseable outputs sat alongside the *lowest* macro F1 (0.3072). Format
compliance and classification quality moved in opposite directions.

Test confusion matrix, *k* = 2 (rows = true, columns = predicted):

|  | joy | sadness | anger | fear | love | surprise |
|---|---:|---:|---:|---:|---:|---:|
| **joy** | 417 | 40 | 88 | 0 | 147 | 0 |
| **sadness** | 73 | 235 | **200** | 1 | 65 | 0 |
| **anger** | 21 | 26 | 198 | 0 | 26 | 0 |
| **fear** | 36 | 47 | **111** | 10 | 16 | 1 |
| **love** | 38 | 20 | 12 | 0 | 88 | 0 |
| **surprise** | 28 | 7 | 20 | 0 | 8 | 2 |

| Label | True | Predicted | Ratio | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| joy | 695 | 613 | 0.88× | 0.680 | 0.600 |
| sadness | 581 | 375 | 0.65× | 0.627 | 0.405 |
| anger | 275 | 629 | **2.29×** | 0.315 | 0.720 |
| fear | 224 | 11 | **0.05×** | 0.909 | **0.045** |
| love | 159 | 350 | 2.20× | 0.251 | 0.553 |
| surprise | 66 | 3 | **0.05×** | 0.667 | **0.030** |

In Phase 13, `sadness` absorbed the negative classes. **Now `anger` does.** It is
predicted 2.29× more often than it occurs. **200 of 581 `sadness` rows** and **111
of 224 `fear` rows** are called `anger`. And `fear` is predicted just 11 times in
2,000.

The twelve examples explain this directly:

| # | Label | Text |
|---:|---|---|
| 1 | love | im supposed to feel compassionate towards that little girl but i feel like she never existed |
| 2 | joy | im also feeling more energetic and able to keep going for a better part of the day |
| 3 | anger | ive test tried dropping it … if something happened to my phone i would feel so fucked up |
| 4 | fear | i was told to do it continues and the fact i feel fear frightened correction terrified of what is next |
| 5 | surprise | i feel less weird about my premature graying that started |
| 6 | sadness | i am not giving up but i am feeling discouraged |
| 7 | love | i feel so blessed to have met each and every one of them |
| 8 | joy | i feel fucking fantastic and wanted to share the news with you |
| 9 | anger | **i just feel cold and drained all the time im either hungry or tired or cold at the moment and it sort of sucks** |
| 10 | fear | i dont know that i am feeling fearful |
| 11 | surprise | i returned to the ground floor feeling dazed |
| 12 | sadness | i feel very isolated from my family so it is really important to me to meet people |

**Example 9 is labelled `anger`, and it describes tiredness.** "Cold and drained
all the time … it sort of sucks" is, to most readers, sadness or fatigue. With two
examples per class, that one sentence is **half of the model's definition of
anger**. The model learned what it was shown — *low-energy negativity is anger* —
and applied it to hundreds of `sadness` and `fear` rows.

**Both `fear` examples contain the literal word** (`fear frightened terrified`,
`feeling fearful`), and one wraps it in a negation. The model appears to have
inferred that `fear` requires the word itself, and it withheld the label almost
everywhere else. **Example 5**, the only surprise example other than "dazed", is
barely surprise at all.

This is Phase 5's label noise coming back through the prompt. The dataset contains
ambiguous and arguably mislabelled rows. **A random sample of 12 rows is not a
teacher — it is a sample of that noise**, and at two examples per class a single
noisy row redefines a class. A supervised model averages over 17,923 rows and is
robust to it. A few-shot prompt amplifies it.

## 14.6 Robustness: is this one unlucky draw?

The test result rests on **one** seeded draw of 12 examples, and the val grid did not
include *k* = 0. Both gaps were found after the result. So both were checked — **on
val only**. Test was not scored again and no choice was re-made. The Phase 14 test
result stands exactly as reported above.

| Configuration | Val macro F1 | Unparseable | Predicted `love` | Predicted `joy` |
|---|---:|---:|---:|---:|
| **Zero-shot (k = 0)** | **0.4397** | 118 | 332 | 432 |
| k = 2, seed 42 (Phase 14) | 0.3592 | 14 | — | — |
| k = 2, seed 7 | 0.4274 | 15 | 659 | 408 |
| k = 2, seed 123 | 0.4272 | 48 | 465 | 312 |
| k = 2, seed 2024 | 0.3035 | 5 | **1,225** | **18** |

| Summary | |
|---|---:|
| Mean val macro F1, k = 2, four draws | 0.3793 |
| Standard deviation | 0.0599 |
| Range | **0.1239** |
| Draws that beat zero-shot on val | **0 of 4** |
| Would val have chosen k = 0, had it been in the grid? | **Yes** |

**The finding is robust: no draw of 12 examples beat zero-shot, on val, in four
tries.** It is not one unlucky sample.

**The procedure is extremely brittle.** Changing only *which* 12 training rows are
shown moves val macro F1 across a **12.4-point range**. Seed 2024 is the extreme
case: the model called **1,225 of 2,000 rows `love` and only 18 `joy`**. One draw
turned it into a love classifier. A deployment built on few-shot prompting would
carry that variance invisibly: its quality would depend on which examples happened
to be pasted into the prompt.

**A design flaw, stated plainly.** The val grid should have included *k* = 0. Had it
done so, val would have selected **zero-shot**, and the few-shot configuration would
never have reached test. The pre-registered design was followed and its test result
is reported unaltered. But the correct operational conclusion is not "use *k* = 2"
— it is **"on this model and benchmark, adding random examples does not help; use
zero-shot if an LLM is used at all."**

## 14.7 Cost and latency

Same harness, same GPU, same protocol as Phase 13.

| | Few-shot (k=2) | Zero-shot | Linear SVM |
|---|---:|---:|---:|
| Device | cuda:0 | cuda:0 | cpu |
| Prompt tokens (val row 0, measured) | 411 | 91 | — |
| p50 single-row | **91.1 ms** | 43.8 ms | 0.220 ms |
| **p95 single-row** | **109.7 ms** | 61.2 ms | 0.246 ms |
| p99 single-row | 112.2 ms | 70.2 ms | — |
| Within 50 ms p95 budget | **No** | No | Yes |
| Best throughput | 13.7 rows/s (batch 8) | 49.7 rows/s | 104,222 rows/s |
| Batch 128 | **out of memory** | fits | — |
| Cost / 1,000 | **$0.01218** | $0.003355 | $5.12 × 10⁻⁷ |
| Cost / month at 1 M | **$12.18** | $3.36 | $0.0005 |
| Peak GPU memory | **6,800 MB** | 4,540 MB | — |
| Model on disk | 2,955 MB | 2,955 MB | 0.411 MB |

**Examples are paid for on every single request.** The 12 examples add about 320
tokens to every prompt, and `transformers` `generate()` re-encodes them every time,
because it keeps no prompt-prefix cache. The result:

- **p95 latency up 1.79×** over zero-shot, to **110 ms** — more than twice the budget,
  and **445×** the SVM.
- **Throughput down 3.6×**, peaking at a batch size of only 8.
- **Batch 128 no longer fits in 8 GB.** Peak memory rose from 4,540 to 6,800 MB,
  because the KV cache grows with prompt length.
- **Cost up 3.6× over zero-shot, and 23,803× the SVM.**

On val, the pattern held across *k*: throughput fell from 21.9 to 13.7 to 8.1 rows/s
as examples went from 6 to 12 to 24.

A serving engine with prefix caching would encode the shared example block once and
reuse it, recovering much of this cost. That is a fair caveat. But it would reduce
the cost of a configuration that is **less accurate** than the one without examples,
so it cannot change the verdict.

## 14.8 The verdict

| Criterion | Required | Few-shot (k=2) | Pass |
|---|---|---:|---|
| Macro F1 margin over best classical | ≥ +0.030 | **−0.515** | **No** |
| Macro F1 needed to pass | ≥ 0.8879 | 0.3427 | **No** |
| p95 latency | ≤ 50 ms | 109.7 ms | **No** |
| Improvement over zero-shot | — | **−0.068** | — |
| **Worth it** | both | — | **No** |

**Few-shot prompting fails every criterion, and it is worse than zero-shot on both
accuracy and cost.** This is a negative result, reported as such.

### What this adds to the research question

Phase 13 showed a small LLM cannot follow this dataset's label conventions without
seeing them. Phase 14 shows that **seeing a handful of them does not fix it**:

1. **A few examples teach format, not meaning.** Parse failures went to zero while
   classification got worse.
2. **Random examples carry the dataset's label noise into the prompt, and a small
   model copies it.** One mislabelled example (tiredness labelled `anger`) redefined
   a class for hundreds of test rows.
3. **The result depends wildly on which examples are drawn**: a 12.4-point range, and
   one draw producing 1,225 `love` predictions out of 2,000.
4. **Every example is a recurring cost**, paid in latency, memory and money on every
   request.

The supervised linear SVM succeeds precisely where few-shot fails: it learns the
convention from **all** 17,923 labelled rows, so no single noisy example can move it.
That points to the last LLM configuration worth testing — **fine-tuning**, which lets
a pre-trained model learn from the whole training set, as the SVM does.

---

## Summary of findings

1. **Few-shot (12 examples, k = 2 per class, chosen on val): test macro F1 0.3427,
   accuracy 0.4750** — **6.8 points worse than zero-shot** and 51.5 below the SVM.
2. **Examples fixed the format**: unparseable outputs fell 143 → 19 on test and
   37 → 14 → 0 on val as *k* rose.
3. **Examples damaged the classification**: on val, more examples meant fewer parse
   failures *and* lower macro F1. At *k* = 4: 0 unparseable, lowest F1 (0.3072).
4. **`anger` became the catch-all** (2.29× over-predicted; 200 of 581 `sadness` rows
   and 111 of 224 `fear` rows). **`fear` collapsed** to 11 predictions (recall 0.045);
   `surprise` to 3 (recall 0.030).
5. **Traced to specific examples**: an `anger` example describing tiredness ("cold
   and drained … it sort of sucks"); both `fear` examples containing the literal word.
   At two per class, one noisy example is half a class's definition.
6. **Robust (val only, test not re-scored): 0 of 4 random draws of 12 examples beat
   zero-shot.**
7. **Extremely brittle**: val macro F1 ranged 0.3035–0.4274 across draws (σ = 0.060);
   one draw predicted `love` for 1,225 of 2,000 rows.
8. **Design flaw stated**: *k* = 0 was absent from the val grid. Had it been present,
   val would have chosen zero-shot. The pre-registered test result is reported
   unaltered.
9. **p95 109.7 ms (1.79× zero-shot, 445× the SVM)**; throughput 13.7 rows/s; **batch 128
   out of memory**; peak GPU memory 6,800 MB; **$0.01218 per 1,000 (23,803× the SVM)**.
10. First run lost to an OOM in the throughput test; fixed by recording OOM as a
    result and checkpointing the val sweep. Possible duplicate test generation with an
    identical, val-fixed configuration is recorded.
11. **Verdict: not worth it**, and dominated by zero-shot. The remaining LLM
    configuration — fine-tuning on the full training set — is Phase 15.
