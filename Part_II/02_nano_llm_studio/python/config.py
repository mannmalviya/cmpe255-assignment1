from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 262
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    hidden_dim: int = 384
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10_000.0

    def to_dict(self) -> dict:
        return asdict(self)


PROFILES = {
    "micro": ModelConfig(dim=96, n_layers=3, hidden_dim=256, max_seq_len=192),
    "nano": ModelConfig(),
}

