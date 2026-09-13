"""Canonical data loader. Every phase loads the split through this module.

The split is the dataset's own three-way split. It is frozen here and is never
re-drawn. `train` and `val` may be used for fitting and tuning. `test` is scored
once, at the end, by each model.
"""

import pandas as pd

from config import DATA_PROCESSED, EXPECTED_ROWS, EXPECTED_TRAIN_COUNTS, LABELS, SPLITS


def load_split(name: str) -> pd.DataFrame:
    """Read one raw split file into a DataFrame with columns [text, label, split].

    The raw format is one row per line: `text;label`. The text itself may contain
    a semicolon, so we split on the LAST semicolon, not the first.
    """
    path = SPLITS[name]
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        text, sep, label = line.rpartition(";")
        if not sep:
            raise ValueError(f"{path.name}:{lineno} has no ';' separator: {line!r}")
        rows.append({"text": text, "label": label.strip(), "split": name})
    return pd.DataFrame(rows)


def load_all() -> pd.DataFrame:
    """Return train+val+test as one frame, with a stable `row_id` per split."""
    frames = []
    for name in ("train", "val", "test"):
        df = load_split(name)
        df = df.reset_index(drop=True)
        df.insert(0, "row_id", [f"{name}_{i}" for i in range(len(df))])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def verify(df: pd.DataFrame) -> None:
    """Fail loudly if the data is not exactly what the study specified."""
    for name, expected in EXPECTED_ROWS.items():
        got = int((df["split"] == name).sum())
        if got != expected:
            raise ValueError(f"split {name}: expected {expected} rows, got {got}")

    seen = set(df["label"].unique())
    if seen != set(LABELS):
        raise ValueError(f"labels differ. expected {sorted(LABELS)}, got {sorted(seen)}")

    counts = df[df["split"] == "train"]["label"].value_counts().to_dict()
    if counts != EXPECTED_TRAIN_COUNTS:
        raise ValueError(f"train label counts differ.\n  expected {EXPECTED_TRAIN_COUNTS}"
                         f"\n  got      {counts}")


def freeze(df: pd.DataFrame) -> None:
    """Persist the frozen split to parquet. Downstream phases read these files."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        part = df[df["split"] == name].reset_index(drop=True)
        part.to_parquet(DATA_PROCESSED / f"{name}.parquet", index=False)


def load_frozen(name: str) -> pd.DataFrame:
    """Read a frozen split. Use this in every phase after Phase 2."""
    return pd.read_parquet(DATA_PROCESSED / f"{name}.parquet")
