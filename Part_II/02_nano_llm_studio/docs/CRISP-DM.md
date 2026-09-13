# PocketLM CRISP-DM Record

## 1. Business understanding

The objective is educational: build an inspectable language model and chatbot that can be trained on a student laptop without an external model provider. Success means the full path from conversational data to local inference is visible, reproducible, and usable through one operations interface. It is not intended to compete with general-purpose foundation models.

## 2. Data understanding

The source is `python/data/conversations.jsonl`. Each row contains `system`, `user`, and `assistant` text. The included corpus is intentionally small and focused on data science, Python, and Transformer concepts. Training records a truncated SHA-256 digest so dashboard metrics can be tied back to the exact corpus.

Risks include narrow coverage, repeated wording, factual omissions, and optimistic validation scores because the dataset is small. New data should be reviewed for correctness, duplication, private information, harmful content, and licensing before it is added.

## 3. Data preparation

Text is encoded directly as UTF-8 bytes. Six reserved tokens represent padding, beginning/end of sequence, and chat roles. This produces a fixed vocabulary of 262 tokens and eliminates unknown-token failures. Each example follows:

```text
<bos><system>instructions\n<user>question\n<assistant>answer<eos>
```

Prompt and padding labels are set to `-100`, so cross-entropy is optimized only on the assistant response. A seeded 85/15 split is used for training and validation.

## 4. Modeling

PocketLM is a decoder-only Transformer. The default `nano` profile uses four layers, a 128-wide embedding, four query heads, two key-value heads, a 384-wide SwiGLU feed-forward block, and a 256-token context. The `micro` profile reduces depth, width, and context for slower laptops.

The model includes pre-RMSNorm, Rotary Position Embeddings, grouped-query attention, native scaled-dot-product attention, SwiGLU, tied input/output embeddings, and cached keys and values during decoding. Training uses AdamW, warmup plus cosine decay, gradient clipping, deterministic splitting, and CUDA mixed precision when available.

## 5. Evaluation

Every epoch records training cross-entropy, held-out cross-entropy, held-out perplexity, and learning rate. The dashboard plots training and validation curves together. A lower validation loss is useful only as a within-corpus comparison; with such a small dataset it is not evidence of broad language capability.

Before expanding the project, add a manually reviewed prompt set and measure exact or rubric-based response quality, latency, peak memory, and safety failures. Never select hyperparameters by repeatedly looking at a final test set.

## 6. Deployment

FastAPI loads `artifacts/pocketlm.pt` onto CUDA, Apple MPS, or CPU and exposes health, chat, tokenizer, metrics, lifecycle, and background-training endpoints. Next.js provides the user-facing chatbot and data-science administration console. Both processes run locally; no container, cloud service, account, or API key is required.

Monitoring is deliberately lightweight: checkpoint status, device, throughput, latency, training progress, curves, parameter count, file size, corpus digest, and CRISP-DM phase status. Re-training writes a temporary checkpoint first and then atomically activates it.

