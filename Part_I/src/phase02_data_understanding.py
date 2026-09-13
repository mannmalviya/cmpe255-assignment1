"""Phase 2 — Data understanding.

Loads all three splits, verifies them against the frozen specification, describes
their structure, and freezes the split to parquet.

This phase DESCRIBES only. It does not clean, drop, or alter a single row.
Splits used: train, val and test are all READ for description. No model sees
test here, and nothing is fitted.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

import data as dataio
from config import LABELS, METRICS, SPLITS, ensure_dirs, set_all_seeds
from hardware import save_metrics

NON_ASCII = re.compile(r"[^\x00-\x7F]")
DIGIT = re.compile(r"\d")
PUNCT = re.compile(r"[^\w\s]")


def structure_report(df: pd.DataFrame) -> dict:
    """Shape, dtypes, null and blank counts. The 'is this a table' check."""
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_cells": {c: int(df[c].isna().sum()) for c in df.columns},
        "blank_text_rows": int((df["text"].str.strip() == "").sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
    }


def label_report(df: pd.DataFrame) -> dict:
    """Per-split class balance, in counts and in shares."""
    out = {}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        counts = part["label"].value_counts()
        out[name] = {
            "n": int(len(part)),
            "counts": {lab: int(counts.get(lab, 0)) for lab in LABELS},
            "share": {lab: round(float(counts.get(lab, 0)) / len(part), 5)
                      for lab in LABELS},
            "majority_class": str(counts.idxmax()),
            "majority_share": round(float(counts.max()) / len(part), 5),
            "imbalance_ratio_max_over_min": round(float(counts.max() / counts.min()), 3),
        }
    return out


def drift_report(labels: dict) -> dict:
    """Does val/test carry the same class mix as train?

    Total variation distance between two label distributions p and q is
    0.5 * sum |p_i - q_i|. It is 0 for identical mixes and 1 for disjoint ones.
    A large value would mean the splits were not drawn from one population, and
    every held-out score would then be measuring drift as well as skill.
    """
    train = labels["train"]["share"]
    out = {}
    for name in ("val", "test"):
        other = labels[name]["share"]
        tvd = 0.5 * sum(abs(train[lab] - other[lab]) for lab in LABELS)
        out[f"tvd_train_vs_{name}"] = round(tvd, 5)
        out[f"max_abs_share_diff_train_vs_{name}"] = round(
            max(abs(train[lab] - other[lab]) for lab in LABELS), 5)
    return out


def character_report(df: pd.DataFrame) -> dict:
    """What is actually inside the strings? Casing, digits, punctuation, unicode.

    This decides how much normalization Phase 5 needs. If the corpus is already
    lowercase ASCII with no punctuation, aggressive cleaning would be wasted work.
    """
    text = df["text"]
    all_chars = Counter("".join(text.tolist()))
    return {
        "rows_with_uppercase": int(text.str.contains(r"[A-Z]", regex=True).sum()),
        "rows_with_digit": int(text.str.contains(DIGIT, regex=True).sum()),
        "rows_with_punctuation": int(text.str.contains(PUNCT, regex=True).sum()),
        "rows_with_non_ascii": int(text.str.contains(NON_ASCII, regex=True).sum()),
        "rows_with_leading_or_trailing_space": int((text != text.str.strip()).sum()),
        "rows_with_double_space": int(text.str.contains("  ", regex=False).sum()),
        "rows_with_semicolon_in_text": int(text.str.contains(";", regex=False).sum()),
        "distinct_characters": len(all_chars),
        "character_classes": {
            "lowercase_letters": sum(v for k, v in all_chars.items() if k.islower()),
            "uppercase_letters": sum(v for k, v in all_chars.items() if k.isupper()),
            "digits": sum(v for k, v in all_chars.items() if k.isdigit()),
            "spaces": all_chars.get(" ", 0),
            "other": sum(v for k, v in all_chars.items()
                         if not (k.islower() or k.isupper() or k.isdigit() or k == " ")),
        },
        "top_20_characters": [[k, int(v)] for k, v in all_chars.most_common(20)],
    }


def coarse_size_report(df: pd.DataFrame) -> dict:
    """Headline size numbers only. Full length analysis belongs to Phase 3."""
    chars = df["text"].str.len()
    words = df["text"].str.split().str.len()
    return {
        "chars": {"min": int(chars.min()), "median": float(chars.median()),
                  "mean": round(float(chars.mean()), 2), "max": int(chars.max())},
        "words": {"min": int(words.min()), "median": float(words.median()),
                  "mean": round(float(words.mean()), 2), "max": int(words.max())},
    }


def main() -> None:
    set_all_seeds()
    ensure_dirs()

    df = dataio.load_all()
    dataio.verify(df)          # raises if the data is not the specified data
    dataio.freeze(df)          # the split is now frozen on disk

    labels = label_report(df)
    payload = {
        "phase": "02_data_understanding",
        "splits_used": ["train", "val", "test"],
        "split_policy": ("dataset's own three-way split, frozen to "
                         "data/processed/{train,val,test}.parquet. Never re-drawn. "
                         "train+val for fitting and tuning; test scored once at the end."),
        "source_files": {k: {"path": str(v), "bytes": v.stat().st_size}
                         for k, v in SPLITS.items()},
        "verification": "PASSED: row counts, label set and train label counts "
                        "match the study specification exactly",
        "structure": {
            "combined": structure_report(df),
            "per_split": {n: structure_report(df[df["split"] == n])
                          for n in ("train", "val", "test")},
        },
        "labels": labels,
        "distribution_drift": drift_report(labels),
        "characters": {
            "combined": character_report(df),
            "per_split": {n: character_report(df[df["split"] == n])
                          for n in ("train", "val", "test")},
        },
        "coarse_size": {n: coarse_size_report(df[df["split"] == n])
                        for n in ("train", "val", "test")},
        "duplicate_preview": {
            "note": "counted only, not removed. Phase 5 handles cleaning and leakage.",
            "exact_duplicate_texts_within_train": int(
                df[df["split"] == "train"]["text"].duplicated().sum()),
            "exact_duplicate_texts_across_all_splits": int(df["text"].duplicated().sum()),
        },
    }

    path = save_metrics("phase02_data_understanding", payload, device="cpu")
    print(json.dumps({k: payload[k] for k in
                      ("verification", "labels", "distribution_drift",
                       "duplicate_preview")}, indent=2))
    print(f"\nwrote {path}")
    print(f"froze split to data/processed/*.parquet")


if __name__ == "__main__":
    main()
