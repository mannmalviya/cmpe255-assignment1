import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Callable

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, random_split

from .config import PROFILES
from .dataset import ChatDataset
from .model import PocketTransformer
from .tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(__file__).parent / "data" / "conversations.jsonl"
ARTIFACTS = ROOT / "artifacts"


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    profile: str = "nano",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    seed: int = 42,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = PROFILES[profile]
    tokenizer = ByteTokenizer()
    dataset = ChatDataset(DATA_PATH, tokenizer, config.max_seq_len, repeat=1)
    validation_size = max(1, int(len(dataset) * 0.15))
    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    training_data, validation_data = random_split(dataset, [training_size, validation_size], generator=generator)
    # Repeat only after splitting so identical rows cannot leak into validation.
    repeated_training_data = ConcatDataset([training_data] * 8)
    training_loader = DataLoader(repeated_training_data, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)

    device = select_device()
    model = PocketTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = max(1, epochs * len(training_loader))
    warmup_steps = max(1, int(total_steps * 0.08))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict] = []
    started = time.perf_counter()
    global_step = 0

    def learning_rate_at(step: int) -> float:
        if step < warmup_steps:
            return learning_rate * (step + 1) / warmup_steps
        ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return learning_rate * (0.1 + 0.45 * (1 + math.cos(math.pi * ratio)))

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        for inputs, labels in training_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            rate = learning_rate_at(global_step)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, loss, _ = model(inputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            training_loss += float(loss.item())
            global_step += 1

        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for inputs, labels in validation_loader:
                _, loss, _ = model(inputs.to(device), labels.to(device))
                validation_loss += float(loss.item())

        record = {
            "epoch": epoch,
            "train_loss": round(training_loss / len(training_loader), 4),
            "validation_loss": round(validation_loss / len(validation_loader), 4),
            "learning_rate": round(rate, 8),
        }
        record["perplexity"] = round(math.exp(min(record["validation_loss"], 20)), 2)
        history.append(record)
        if progress:
            progress({"status": "training", "epoch": epoch, "epochs": epochs, "latest": record})

    ARTIFACTS.mkdir(exist_ok=True)
    checkpoint_path = ARTIFACTS / "pocketlm.pt"
    temporary_path = ARTIFACTS / "pocketlm.tmp"
    torch.save(model.checkpoint(), temporary_path)
    temporary_path.replace(checkpoint_path)
    tokenizer.save(ARTIFACTS / "tokenizer.json")
    metrics = {
        "status": "ready",
        "profile": profile,
        "device": str(device),
        "parameters": model.parameter_count,
        "model_size_mb": round(checkpoint_path.stat().st_size / 1_048_576, 2),
        "epochs": epochs,
        "batch_size": batch_size,
        "training_examples": training_size,
        "validation_examples": validation_size,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()[:12],
        "history": history,
        "final": history[-1],
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PocketLM on a laptop GPU or CPU.")
    parser.add_argument("--profile", choices=PROFILES, default="nano")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    print(json.dumps(train(args.profile, args.epochs, args.batch_size, args.learning_rate), indent=2))
