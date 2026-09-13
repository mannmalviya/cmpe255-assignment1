# PocketLM Studio

A small language model, chatbot, and data-science administration dashboard built with Python, PyTorch, FastAPI, Next.js, and TypeScript. It follows all six CRISP-DM phases and uses modern Transformer primitives while remaining small enough for a laptop GPU or CPU.

This is an original implementation inspired by the architecture topics in [dlmastery's nano LLM example](https://github.com/dlmastery/data_science_examples/tree/main/02_nano_llm_transformer). The code, training flow, visual design, copy, API contract, and CRISP-DM artifacts here are independent.

## Run it

Use two terminals from this directory.

```bash
# Terminal 1 — Python API
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn python.api:app --reload --host 127.0.0.1 --port 8000
```

```bash
# Terminal 2 — Next.js
npm install
npm run dev
```

Open `http://localhost:3000`, choose **Model ops**, and start a `micro` or `nano` training run. The chatbot becomes available when the checkpoint is loaded. For CLI training instead:

```bash
python -m python.train --profile nano --epochs 20
```

## Laptop profiles

| Profile | Width | Layers | Heads / KV heads | Context | Best for |
|---|---:|---:|---:|---:|---|
| micro | 96 | 3 | 4 / 2 | 192 | CPU and quick iteration |
| nano | 128 | 4 | 4 / 2 | 256 | Small laptop GPUs |

The runtime automatically chooses CUDA, Apple MPS, or CPU. CUDA training uses mixed precision. Generation uses an autoregressive KV cache.

## Project map

```text
app/            Next.js application shell and visual system
components/     Chat, admin, CRISP-DM, telemetry, and tokenizer UI
lib/            Typed frontend API helper
python/         Model, tokenizer, dataset, trainer, and FastAPI service
artifacts/      Generated checkpoint, tokenizer metadata, and metrics
docs/           CRISP-DM record and model card
```

See [docs/CRISP-DM.md](docs/CRISP-DM.md) for the lifecycle rationale and [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for appropriate-use boundaries.

