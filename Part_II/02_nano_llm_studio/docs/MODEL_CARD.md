# PocketLM Model Card

PocketLM is a from-scratch educational decoder Transformer. It demonstrates modern language-model components at a scale that can run on a laptop.

- Intended use: coursework, architecture inspection, tiny-corpus experiments, and local UI demonstrations.
- Not intended for: factual decision-making, medical/legal/financial advice, production support, safety-critical use, or sensitive data.
- Training data: the bundled, hand-authored conversational JSONL unless the operator changes it.
- Primary metrics: assistant-token validation loss and perplexity. These do not measure factuality or general reasoning.
- Main limitations: tiny dataset, byte-level sequences, short context, no alignment evaluation, no retrieval, and unpredictable generations outside the training distribution.
- Privacy: local by design, but anything added to the corpus is serialized into a model checkpoint and may be memorized.

The dashboard must not be interpreted as evidence that the model is production-ready. It is an observability surface for a learning artifact.

