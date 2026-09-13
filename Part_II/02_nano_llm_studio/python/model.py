import math
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def rope_cache(head_dim: int, length: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(length).float(), inverse)
    return torch.repeat_interleave(angles.cos(), 2, -1), torch.repeat_interleave(angles.sin(), 2, -1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, :, None, :].to(device=x.device, dtype=x.dtype)
    sin = sin[None, :, None, :].to(device=x.device, dtype=x.dtype)
    return x * cos + rotate_half(x) * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.dim % config.n_heads or config.n_heads % config.n_kv_heads:
            raise ValueError("dim must divide n_heads and n_heads must divide n_kv_heads")
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.groups = config.n_heads // config.n_kv_heads
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, sequence, _ = x.shape
        q = self.q_proj(x).view(batch, sequence, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, sequence, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, sequence, self.n_kv_heads, self.head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        cache = (k, v) if use_cache else None
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        q = q.transpose(1, 2)
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=past is None and sequence > 1,
        )
        output = attention.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.out_proj(output), cache


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.feed_forward = SwiGLU(config)

    def forward(self, x, cos, sin, past=None, use_cache=False):
        attended, cache = self.attention(self.attention_norm(x), cos, sin, past, use_cache)
        x = x + attended
        return x + self.feed_forward(self.ffn_norm(x)), cache


class PocketTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        cos, sin = rope_cache(config.dim // config.n_heads, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, tokens, targets=None, past_key_values=None, use_cache=False):
        past_length = 0 if not past_key_values else past_key_values[0][0].shape[2]
        sequence = tokens.shape[1]
        if past_length + sequence > self.config.max_seq_len:
            raise ValueError("Sequence exceeds configured context window")
        cos = self.rope_cos[past_length : past_length + sequence]
        sin = self.rope_sin[past_length : past_length + sequence]
        x = self.embedding(tokens)
        caches = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            x, cache = layer(x, cos, sin, past, use_cache)
            if use_cache:
                caches.append(cache)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        return logits, loss, caches if use_cache else None

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=96, temperature=0.8, top_k=40, top_p=0.9):
        self.eval()
        device = next(self.parameters()).device
        prompt = prompt[-self.config.max_seq_len :]
        tokens = torch.tensor([prompt], dtype=torch.long, device=device)
        logits, _, cache = self(tokens, use_cache=True)
        output: list[int] = []
        budget = min(max_new_tokens, self.config.max_seq_len - len(prompt))
        for _ in range(max(0, budget)):
            scores = logits[:, -1, :]
            if temperature <= 0:
                next_token = scores.argmax(-1, keepdim=True)
            else:
                scores = scores / max(temperature, 1e-5)
                if top_k > 0:
                    threshold = torch.topk(scores, min(top_k, scores.shape[-1])).values[:, -1:]
                    scores = scores.masked_fill(scores < threshold, -torch.inf)
                if top_p < 1.0:
                    sorted_scores, sorted_ids = scores.sort(descending=True)
                    cumulative = sorted_scores.softmax(-1).cumsum(-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    scores.scatter_(1, sorted_ids, sorted_scores.masked_fill(remove, -torch.inf))
                next_token = torch.multinomial(scores.softmax(-1), 1)
            token_id = int(next_token.item())
            if token_id == 2:
                break
            output.append(token_id)
            logits, _, cache = self(next_token, past_key_values=cache, use_cache=True)
        return output

    def checkpoint(self) -> dict:
        return {"config": asdict(self.config), "model": self.state_dict()}

