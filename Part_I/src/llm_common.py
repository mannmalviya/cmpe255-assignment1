"""Shared LLM machinery for Phases 13 (zero-shot) and 14 (few-shot).

Kept in one place so the two phases differ ONLY in their prompt. If prompting and
parsing were reimplemented per phase, a difference in the harness could be
mistaken for a difference in prompting strategy.

THE PROMPT AND THE PARSER ARE FIXED HERE, BEFORE ANY RESULT IS SEEN. They are not
revised after looking at scores. Any change would be prompt-tuning on the test
split, which is the same violation as tuning a hyper-parameter on it.
"""

import re
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from config import LABELS

# A 1.5B instruct model in bf16 is ~3.1 GB of weights. The RTX 4060 Laptop has
# 8,188 MiB, so this leaves room for a KV cache at batch sizes that matter. A 3B
# model in bf16 (~6.2 GB) would fit the weights but not comfortable batching, and
# the study needs batched throughput as much as single-row latency.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Generation is deterministic: greedy, no sampling. A sampled model would give a
# different answer on re-run and make the reported score irreproducible.
MAX_NEW_TOKENS = 6
GEN_KWARGS = dict(do_sample=False, temperature=None, top_p=None, top_k=None)

LABEL_SET = set(LABELS)
_LABEL_RE = re.compile(r"\b(" + "|".join(LABELS) + r")\b")


# ------------------------------------------------------------------ prompt

SYSTEM_PROMPT = (
    "You are an emotion classifier. You are given one short first-person "
    "message. Reply with exactly one word: the single emotion the writer is "
    "expressing.\n"
    "The only allowed answers are: joy, sadness, anger, fear, love, surprise.\n"
    "Reply with the word alone. No punctuation, no explanation, no other text."
)


def build_messages(text: str, shots: list = None) -> list:
    """Chat messages for one input. `shots` is used by Phase 14 only."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot_text, shot_label in (shots or []):
        messages.append({"role": "user", "content": shot_text})
        messages.append({"role": "assistant", "content": shot_label})
    messages.append({"role": "user", "content": text})
    return messages


# ------------------------------------------------------------------ parsing

@dataclass
class ParseResult:
    strict: str | None          # exact match on the whole trimmed output
    lenient: str | None         # first label word found anywhere
    raw: str


def parse_label(generated: str) -> ParseResult:
    """Two-tier parsing, both reported.

    STRICT is the primary rule and the one the reported score uses: the entire
    generated string, lowercased and stripped of surrounding whitespace and
    trailing punctuation, must equal one of the six labels. Anything else is
    UNPARSEABLE and counts as an error.

    LENIENT is reported alongside as a diagnostic: the first label word appearing
    anywhere in the output. The gap between the two measures how much of the
    model's apparent weakness is formatting rather than classification.

    Silently retrying until the model complies is NOT done. That would measure a
    system we are not deploying, and it would hide the real per-prediction cost.
    """
    raw = generated
    cleaned = generated.strip().strip(".,!?;:'\"").strip().lower()
    strict = cleaned if cleaned in LABEL_SET else None
    m = _LABEL_RE.search(generated.lower())
    return ParseResult(strict=strict, lenient=m.group(1) if m else None, raw=raw)


# ------------------------------------------------------------------ runner

@dataclass
class LLMRunner:
    """Wraps a causal LM as a text -> label callable, with generation logging."""
    model_id: str = MODEL_ID
    device: str = "cuda:0"
    dtype: object = torch.bfloat16
    shots: list = field(default_factory=list)
    model: object = None
    tokenizer: object = None
    unparseable: list = field(default_factory=list)
    n_calls: int = 0

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Decoder-only batching requires LEFT padding: right padding would put
        # pad tokens between the prompt and the generated continuation.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=self.dtype, device_map=None).to(self.device)
        self.model.eval()
        return self

    def _render(self, texts: list) -> list:
        return [self.tokenizer.apply_chat_template(
            build_messages(t, self.shots), tokenize=False,
            add_generation_prompt=True) for t in texts]

    @torch.inference_mode()
    def generate(self, texts: list) -> list:
        prompts = self._render(texts)
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             add_special_tokens=False).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                  pad_token_id=self.tokenizer.pad_token_id,
                                  **GEN_KWARGS)
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def predict_batch(self, texts: list) -> list:
        """Returns strict-parsed labels; unparseable becomes the sentinel string."""
        gens = self.generate(list(texts))
        labels = []
        for text, gen in zip(texts, gens):
            p = parse_label(gen)
            self.n_calls += 1
            if p.strict is None:
                self.unparseable.append({"text": text, "raw_output": p.raw,
                                         "lenient_would_give": p.lenient})
                labels.append("UNPARSEABLE")
            else:
                labels.append(p.strict)
        return labels

    def predict_one(self, text: str) -> str:
        return self.predict_batch([text])[0]

    def gpu_memory_mb(self) -> dict:
        if not str(self.device).startswith("cuda"):
            return {}
        return {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        }

    def parameters_billions(self) -> float:
        return round(sum(p.numel() for p in self.model.parameters()) / 1e9, 3)


def run_full_split(runner: LLMRunner, texts: list, batch_size: int = 32,
                   log_every: int = 10) -> tuple:
    """Score an entire split, returning (labels, raw_generations, seconds)."""
    labels, raws = [], []
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        gens = runner.generate(chunk)
        for text, gen in zip(chunk, gens):
            p = parse_label(gen)
            raws.append({"text": text, "raw": p.raw, "strict": p.strict,
                         "lenient": p.lenient})
            labels.append(p.strict if p.strict is not None else "UNPARSEABLE")
        if (i // batch_size) % log_every == 0:
            done = min(i + batch_size, len(texts))
            print(f"    {done}/{len(texts)} rows "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return labels, raws, time.perf_counter() - t0


def parse_report(raws: list) -> dict:
    """Every unparseable output, counted and characterised."""
    n = len(raws)
    strict_ok = sum(1 for r in raws if r["strict"] is not None)
    lenient_ok = sum(1 for r in raws if r["lenient"] is not None)
    bad = [r for r in raws if r["strict"] is None]
    rescued = [r for r in bad if r["lenient"] is not None]

    from collections import Counter
    shapes = Counter(r["raw"].strip()[:60] for r in bad)
    return {
        "n_outputs": n,
        "strict_parse_rate": round(strict_ok / n, 5),
        "lenient_parse_rate": round(lenient_ok / n, 5),
        "n_unparseable_strict": len(bad),
        "n_unparseable_but_lenient_finds_a_label": len(rescued),
        "n_unparseable_with_no_label_at_all": len(bad) - len(rescued),
        "policy": "strict parse is primary; an unparseable output counts as an "
                  "ERROR. No retries, no repair, no constrained decoding.",
        "most_common_unparseable_outputs": [[k, v] for k, v in shapes.most_common(15)],
        "all_unparseable": bad,
    }
