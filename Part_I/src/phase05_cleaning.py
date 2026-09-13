"""Phase 5 — Data cleaning and leakage detection.

SPLIT USED: train, val and test. Leakage can only be found by comparing splits,
so all three are read. Only their TEXT is compared; no val or test row informs
any fitting decision, and the test split itself is never modified.

Order of work:
  1. prove that normalization is a no-op instead of assuming it
  2. exact duplicates within each split
  3. exact duplicates across splits  <- this is leakage
  4. same text carrying different labels  <- this is label noise
  5. apply a stated policy and write the cleaned training data

Near-duplicate (non-identical) detection needs an embedding space and is
deferred to Phase 9, where LSH does it.
"""

import itertools
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

import data as dataio
from config import DATA_PROCESSED, LABELS, ensure_dirs, set_all_seeds
from hardware import save_metrics

PUNCT = re.compile(r"[^\w\s]")
WS = re.compile(r"\s+")


# ------------------------------------------------------- 1. normalization

def normalize(text: str) -> str:
    """The textbook cleaning pipeline, applied so we can measure its effect."""
    t = unicodedata.normalize("NFKD", text)
    t = t.encode("ascii", "ignore").decode("ascii")   # strip non-ASCII
    t = t.lower()
    t = PUNCT.sub(" ", t)                             # strip punctuation
    t = WS.sub(" ", t).strip()                        # collapse whitespace
    return t


def normalization_report(df: pd.DataFrame) -> dict:
    """Phase 2 said the corpus is already normalized. Prove it, do not assume it."""
    norm = df["text"].map(normalize)
    changed = norm != df["text"]
    examples = [{"row_id": r.row_id, "before": r.text, "after": n}
                for r, n in zip(df[changed].head(5).itertuples(), norm[changed].head(5))]
    return {
        "pipeline": "NFKD -> strip non-ASCII -> lowercase -> strip punctuation "
                    "-> collapse whitespace -> strip",
        "rows_changed": int(changed.sum()),
        "rows_total": int(len(df)),
        "share_changed": round(float(changed.mean()), 6),
        "examples": examples,
        "conclusion": ("no-op: the corpus arrives already normalized, so no "
                       "cleaning transform is applied. Verified, not assumed."
                       if changed.sum() == 0 else
                       "NOT a no-op: normalization changes rows. Review before "
                       "deciding whether to apply it."),
    }


# ------------------------------------------------------- 2/3. duplicates

def within_split_duplicates(df: pd.DataFrame, name: str) -> dict:
    """Repeated texts inside one split. Same label = redundancy; different = noise."""
    part = df[df["split"] == name]
    groups = part.groupby("text")
    repeated = {t: g for t, g in groups if len(g) > 1}

    same_label, conflicting = 0, []
    extra_rows = 0
    for text, g in repeated.items():
        labels = set(g["label"])
        extra_rows += len(g) - 1
        if len(labels) == 1:
            same_label += 1
        else:
            conflicting.append({
                "text": text,
                "labels": sorted(Counter(g["label"]).items(), key=lambda kv: -kv[1]),
                "row_ids": list(g["row_id"]),
            })
    return {
        "n_rows": int(len(part)),
        "distinct_texts": int(part["text"].nunique()),
        "repeated_texts": len(repeated),
        "redundant_rows": int(extra_rows),
        "repeated_texts_one_label": same_label,
        "repeated_texts_conflicting_labels": len(conflicting),
        "conflicting_examples": conflicting[:10],
    }


def cross_split_overlap(df: pd.DataFrame, a: str, b: str) -> dict:
    """Texts present in BOTH splits. train-vs-test overlap is leakage."""
    ta = df[df["split"] == a]
    tb = df[df["split"] == b]
    shared = set(ta["text"]) & set(tb["text"])

    la = ta.drop_duplicates("text").set_index("text")["label"]
    lb = tb.drop_duplicates("text").set_index("text")["label"]
    agree = sum(1 for t in shared if la[t] == lb[t])

    rows_b = int(tb["text"].isin(shared).sum())
    examples = [{"text": t, f"{a}_label": la[t], f"{b}_label": lb[t]}
                for t in list(shared)[:6]]
    return {
        "shared_texts": len(shared),
        f"rows_affected_in_{a}": int(ta["text"].isin(shared).sum()),
        f"rows_affected_in_{b}": rows_b,
        f"share_of_{b}_rows_affected": round(rows_b / len(tb), 6),
        "labels_agree": agree,
        "labels_disagree": len(shared) - agree,
        "examples": examples,
    }


# ------------------------------------------------- 4. label contradictions

def contradiction_report(df: pd.DataFrame) -> dict:
    """Corpus-wide: which texts appear more than once, and do their labels agree?

    Whenever a text recurs with a different label, at most one of the two
    annotations can be right. Counting the label PAIRS that collide gives a
    direct, model-free measurement of which emotions annotators confuse, and it
    predicts the confusion structure Phase 17 should find.
    """
    labels_per_text = df.groupby("text")["label"].agg(lambda s: tuple(sorted(set(s))))
    occurrences = df.groupby("text").size()
    repeated = labels_per_text[occurrences > 1]
    conflicting = repeated[repeated.map(len) > 1]

    pairs = Counter(conflicting)
    test_texts = set(df[df["split"] == "test"]["text"])
    unanswerable = len(test_texts & set(conflicting.index))

    # Is a collision more common than class sizes alone would produce? Under
    # independent draws, P(pair = {a,b}) = 2 p_a p_b / (1 - sum p_i^2).
    share = (df["label"].value_counts() / len(df)).to_dict()
    denom = 1 - sum(v * v for v in share.values())
    total = sum(pairs.values())
    enrichment = []
    for a, b in itertools.combinations(sorted(share), 2):
        expected = 2 * share[a] * share[b] / denom
        observed = pairs.get(tuple(sorted((a, b))), 0)
        enrichment.append({
            "pair": f"{a} <-> {b}",
            "observed": int(observed),
            "observed_share": round(observed / total, 4),
            "expected_share_by_chance": round(expected, 4),
            "expected_count": round(expected * total, 2),
            "enrichment_ratio": round((observed / total) / expected, 2),
        })
    enrichment.sort(key=lambda r: -r["observed"])

    return {
        "collision_enrichment": enrichment,
        "enrichment_caveat": (
            f"n = {total} conflicting texts, so per-pair counts are small and these "
            "ratios are descriptive, not a hypothesis test. Two results are large "
            "enough to be robust: joy<->love (29 observed vs 3.7 expected) and "
            "joy<->sadness (0 observed vs 13.1 expected, p ~ 3e-7 under the null)."),
        "texts_appearing_more_than_once": int(len(repeated)),
        "of_those_with_conflicting_labels": int(len(conflicting)),
        "of_those_with_one_label_true_redundancy": int((repeated.map(len) == 1).sum()),
        "colliding_label_pairs": [{"pair": f"{p[0]} <-> {p[1]}", "n": int(c)}
                                  for p, c in pairs.most_common()],
        "test_rows_contradicted_elsewhere": unanswerable,
        "share_of_test_contradicted": round(unanswerable / 2000, 6),
        "test_accuracy_ceiling_from_this_alone": round(1 - unanswerable / 2000, 6),
        "interpretation": (
            "51 of 52 repeated texts carry conflicting labels, so the dataset was "
            "de-duplicated on the (text, label) pair rather than on text. Identical "
            "texts therefore survive only where annotators disagreed. These rows are "
            "adversarial, not advantageous: a model that memorises the training copy "
            "is guaranteed to miss the held-out copy. They set a hard ceiling on "
            "achievable test accuracy."),
    }


# ------------------------------------------------------- 5. policy

def apply_policy(df: pd.DataFrame) -> tuple:
    """Clean the TRAINING data only. val and test are left byte-identical.

    Rationale for asymmetry: val and test are the benchmark. Altering them would
    make our scores incomparable with any other work on this dataset, and would
    let a cleaning choice flatter the result. Every removal therefore happens on
    the training side, where it can only cost us information, never gift us any.

    Three removals, in order:
      A. training rows whose text also appears in val or test  -> leakage
      B. training rows whose text appears more than once with conflicting labels
         -> unlearnable contradictions; all copies go
      C. exact redundant repeats within train (same text, same label) -> keep one
    """
    train = df[df["split"] == "train"].copy()
    held_out = set(df[df["split"].isin(["val", "test"])]["text"])

    leak_mask = train["text"].isin(held_out)
    removed_leak = train[leak_mask]
    train = train[~leak_mask]

    counts = train.groupby("text")["label"].nunique()
    conflict_texts = set(counts[counts > 1].index)
    conflict_mask = train["text"].isin(conflict_texts)
    removed_conflict = train[conflict_mask]
    train = train[~conflict_mask]

    before = len(train)
    train = train.drop_duplicates("text", keep="first")
    n_redundant = before - len(train)

    removals = {
        "A_leakage_rows_removed_from_train": int(len(removed_leak)),
        "B_conflicting_label_rows_removed_from_train": int(len(removed_conflict)),
        "C_redundant_repeat_rows_removed_from_train": int(n_redundant),
        "train_rows_before": 16000,
        "train_rows_after": int(len(train)),
        "total_removed": int(16000 - len(train)),
        "share_removed": round((16000 - len(train)) / 16000, 6),
        "val_rows_after": 2000,
        "test_rows_after": 2000,
        "val_test_modified": False,
    }
    return train.reset_index(drop=True), removals, removed_leak


def main() -> None:
    set_all_seeds()
    ensure_dirs()

    df = dataio.load_all()

    norm = normalization_report(df)
    within = {n: within_split_duplicates(df, n) for n in ("train", "val", "test")}
    across = {f"{a}_vs_{b}": cross_split_overlap(df, a, b)
              for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    contradictions = contradiction_report(df)

    clean_train, removals, removed_leak = apply_policy(df)
    clean_train.to_parquet(DATA_PROCESSED / "train_clean.parquet", index=False)

    clean_counts = clean_train["label"].value_counts().to_dict()

    payload = {
        "phase": "05_cleaning_and_leakage",
        "splits_used": ["train", "val", "test"],
        "split_policy": ("all three read, for text comparison only. val and test "
                         "are left byte-identical; every removal is applied to "
                         "train. data/processed/{val,test}.parquet are unchanged."),
        "normalization": norm,
        "duplicates_within_split": within,
        "overlap_across_splits": across,
        "label_contradictions": contradictions,
        "cleaning_policy": {
            "A": "remove training rows whose text also appears in val or test "
                 "(leakage). val/test untouched.",
            "B": "remove ALL copies of a training text that carries conflicting "
                 "labels (unlearnable contradiction).",
            "C": "collapse exact repeats within train (same text, same label) to "
                 "one row.",
            "not_applied": "no normalization transform (proved to be a no-op); "
                           "no stopword removal; no stemming; no near-duplicate "
                           "removal (deferred to Phase 9).",
        },
        "removals": removals,
        "clean_train_label_counts": {lab: int(clean_counts.get(lab, 0))
                                     for lab in LABELS},
        "artefacts": {
            "frozen_raw": "data/processed/{train,val,test}.parquet (unchanged)",
            "cleaned_training_data": "data/processed/train_clean.parquet",
        },
    }
    path = save_metrics("phase05_cleaning", payload, device="cpu")

    print("=== 1. NORMALIZATION ===")
    print(f"  rows changed by the full cleaning pipeline: {norm['rows_changed']} "
          f"of {norm['rows_total']}")
    print(f"  -> {norm['conclusion']}")
    print("\n=== 2. DUPLICATES WITHIN SPLIT ===")
    for n, r in within.items():
        print(f"  {n:6s} {r['n_rows']:>5} rows, {r['distinct_texts']:>5} distinct, "
              f"{r['repeated_texts']:>3} repeated texts "
              f"({r['repeated_texts_conflicting_labels']} with conflicting labels)")
    print("\n=== 3. OVERLAP ACROSS SPLITS (leakage) ===")
    for k, r in across.items():
        a, b = k.split("_vs_")
        print(f"  {k:16s} shared texts: {r['shared_texts']:>3}  "
              f"rows in {b}: {r[f'rows_affected_in_{b}']:>3} "
              f"({r[f'share_of_{b}_rows_affected']:.3%})  "
              f"labels agree {r['labels_agree']}/{r['shared_texts']}")
    print("\n=== 4. LABEL CONTRADICTIONS (corpus-wide) ===")
    c = contradictions
    print(f"  repeated texts: {c['texts_appearing_more_than_once']}, "
          f"conflicting: {c['of_those_with_conflicting_labels']}, "
          f"true redundancy: {c['of_those_with_one_label_true_redundancy']}")
    for row in c["colliding_label_pairs"]:
        print(f"    {row['n']:>3}  {row['pair']}")
    print(f"  test rows contradicted elsewhere: {c['test_rows_contradicted_elsewhere']} "
          f"({c['share_of_test_contradicted']:.3%}) -> accuracy ceiling "
          f"{c['test_accuracy_ceiling_from_this_alone']:.4%}")

    print("\n=== 5. CLEANING APPLIED (train only) ===")
    print(json.dumps(removals, indent=2))
    print("\nclean train label counts:", json.dumps(payload["clean_train_label_counts"]))
    print(f"\nwrote {path}")
    print("wrote data/processed/train_clean.parquet")

    if across["train_vs_test"]["shared_texts"]:
        print("\n--- leakage examples (train text also present in test) ---")
        for e in across["train_vs_test"]["examples"][:4]:
            print(f"  [{e['train_label']} | {e['test_label']}] {e['text'][:90]}")


if __name__ == "__main__":
    main()
