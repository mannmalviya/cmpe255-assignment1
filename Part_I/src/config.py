"""Frozen study configuration. Import this everywhere. Do not redefine constants."""

import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "reports" / "figures"
METRICS = PROJECT_ROOT / "reports" / "metrics"
DOCS = PROJECT_ROOT / "reports" / "docs"

SEED = 42

LABELS = ["joy", "sadness", "anger", "fear", "love", "surprise"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

# The split is the dataset's own split. It is frozen. It is never re-drawn.
# train + val may be used for tuning. test is touched only for final scoring.
SPLITS = {"train": DATA_RAW / "train.txt",
          "val": DATA_RAW / "val.txt",
          "test": DATA_RAW / "test.txt"}

EXPECTED_ROWS = {"train": 16000, "val": 2000, "test": 2000}
EXPECTED_TRAIN_COUNTS = {"joy": 5362, "sadness": 4666, "anger": 2159,
                         "fear": 1937, "love": 1304, "surprise": 572}

# Latency protocol (hard rule).
LATENCY_WARMUP_RUNS = 20
LATENCY_MIN_RUNS = 200

# Success thresholds set in Phase 1 (business understanding).
TARGET_MACRO_F1 = 0.85
LATENCY_BUDGET_P95_MS = 50.0


def set_all_seeds(seed: int = SEED) -> None:
    """Set every seed we can reach. Call at the top of every script."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs() -> None:
    for d in (DATA_PROCESSED, FIGURES, METRICS, DOCS):
        d.mkdir(parents=True, exist_ok=True)
