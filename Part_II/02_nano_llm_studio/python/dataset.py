import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


class ChatDataset(Dataset):
    def __init__(self, path: Path, tokenizer: ByteTokenizer, max_seq_len: int, repeat: int = 8):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.examples = rows * repeat
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.examples[index]
        sequence, assistant_start = self.tokenizer.training_example(
            row.get("system", "You are a concise, helpful assistant."),
            row["user"],
            row["assistant"],
        )
        sequence = sequence[: self.max_seq_len + 1]
        inputs, labels = sequence[:-1], sequence[1:]
        labels = [token if position + 1 >= assistant_start else -100 for position, token in enumerate(labels)]
        padding = self.max_seq_len - len(inputs)
        inputs.extend([self.tokenizer.SPECIAL["<pad>"]] * padding)
        labels.extend([-100] * padding)
        return torch.tensor(inputs), torch.tensor(labels)

