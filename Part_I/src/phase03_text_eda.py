"""Phase 3 — Text EDA. Numbers only; Phase 4 draws them.

SPLIT USED: train ONLY (16,000 rows).

Vocabulary statistics computed on val or test would leak information about the
held-out distribution into feature and model decisions, so they are not touched
here. Length statistics for val/test were already reported coarsely in Phase 2.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import data as dataio
from config import LABELS, ensure_dirs, set_all_seeds
from hardware import save_metrics

PCTS = [1, 5, 25, 50, 75, 90, 95, 99]


# ---------------------------------------------------------------- lengths

def describe_lengths(values: pd.Series) -> dict:
    v = values.to_numpy()
    mean, med = float(v.mean()), float(np.median(v))
    std = float(v.std(ddof=1))
    # Pearson's second skewness coefficient: 3*(mean - median)/std.
    # Positive means a right tail (a few unusually long documents).
    skew = 3.0 * (mean - med) / std if std > 0 else 0.0
    return {
        "n": int(v.size),
        "min": int(v.min()), "max": int(v.max()),
        "mean": round(mean, 3), "std": round(std, 3),
        "percentiles": {str(p): float(np.percentile(v, p)) for p in PCTS},
        "skewness_pearson2": round(skew, 4),
    }


def length_report(df: pd.DataFrame) -> dict:
    words = df["text"].str.split().str.len()
    chars = df["text"].str.len()
    out = {
        "overall": {"words": describe_lengths(words), "chars": describe_lengths(chars)},
        "per_class": {},
    }
    for lab in LABELS:
        m = df["label"] == lab
        out["per_class"][lab] = {
            "words": describe_lengths(words[m]),
            "chars": describe_lengths(chars[m]),
        }
    # Does length itself carry label signal? If mean word counts per class were
    # far apart, a trivial length feature would predict the label.
    means = {lab: out["per_class"][lab]["words"]["mean"] for lab in LABELS}
    out["length_signal"] = {
        "mean_words_per_class": means,
        "spread_max_minus_min": round(max(means.values()) - min(means.values()), 3),
        "overall_std_words": out["overall"]["words"]["std"],
    }
    return out


# ---------------------------------------------------------------- vocabulary

def vocabulary_report(df: pd.DataFrame) -> dict:
    tokens = [t for text in df["text"] for t in text.split()]
    counts = Counter(tokens)
    n_tokens, n_types = len(tokens), len(counts)

    freqs = np.array(sorted(counts.values(), reverse=True))
    cum = np.cumsum(freqs) / n_tokens
    coverage = {f"types_covering_{int(p*100)}pct_tokens": int(np.searchsorted(cum, p) + 1)
                for p in (0.5, 0.8, 0.9, 0.95, 0.99)}

    hapax = sum(1 for c in counts.values() if c == 1)
    dis = sum(1 for c in counts.values() if c == 2)

    # Zipf check: log-frequency against log-rank should be near-linear with
    # slope about -1 for natural language.
    ranks = np.arange(1, n_types + 1)
    slope, intercept = np.polyfit(np.log(ranks), np.log(freqs), 1)
    pred = intercept + slope * np.log(ranks)
    ss_res = float(((np.log(freqs) - pred) ** 2).sum())
    ss_tot = float(((np.log(freqs) - np.log(freqs).mean()) ** 2).sum())

    doc_freq = Counter()
    for text in df["text"]:
        doc_freq.update(set(text.split()))

    return {
        "n_tokens": n_tokens,
        "n_types": n_types,
        "type_token_ratio": round(n_types / n_tokens, 6),
        "hapax_legomena": hapax,
        "hapax_share_of_types": round(hapax / n_types, 5),
        "dis_legomena": dis,
        "mean_tokens_per_doc": round(n_tokens / len(df), 3),
        "coverage": coverage,
        "zipf_fit": {"slope": round(float(slope), 4),
                     "r_squared": round(1 - ss_res / ss_tot, 5)},
        "top_30_tokens": [[w, int(c)] for w, c in counts.most_common(30)],
        "top_30_by_document_frequency": [
            [w, int(c), round(c / len(df), 4)] for w, c in doc_freq.most_common(30)],
        "types_in_at_least_1pct_of_docs": int(
            sum(1 for c in doc_freq.values() if c >= 0.01 * len(df))),
    }


# ---------------------------------------------------------------- per class words

def log_odds_dirichlet(class_counts: Counter, rest_counts: Counter,
                       prior: Counter, top_k: int = 15) -> list:
    """Monroe, Colaresi & Quinn (2008) log-odds ratio with informative Dirichlet prior.

    Plain frequency ranks stopwords first; plain ratios rank rare noise first.
    This estimator shrinks each word toward the corpus-wide rate and divides by
    its own standard error, so the score is a z-statistic: how confidently is
    this word over-used by this class relative to the rest of the corpus?
    """
    a0 = sum(prior.values())
    n_i, n_j = sum(class_counts.values()), sum(rest_counts.values())
    scored = []
    for w, aw in prior.items():
        yi, yj = class_counts.get(w, 0), rest_counts.get(w, 0)
        num_i, num_j = yi + aw, yj + aw
        den_i = n_i + a0 - num_i
        den_j = n_j + a0 - num_j
        if den_i <= 0 or den_j <= 0:
            continue
        delta = math.log(num_i / den_i) - math.log(num_j / den_j)
        z = delta / math.sqrt(1.0 / num_i + 1.0 / num_j)
        scored.append((w, z, yi))
    scored.sort(key=lambda r: r[1], reverse=True)
    return [[w, round(z, 3), int(c)] for w, z, c in scored[:top_k]]


def per_class_words(df: pd.DataFrame) -> dict:
    per_class = {lab: Counter(t for text in df[df["label"] == lab]["text"]
                              for t in text.split()) for lab in LABELS}
    corpus = Counter()
    for c in per_class.values():
        corpus.update(c)

    out = {}
    for lab in LABELS:
        rest = Counter(corpus)
        rest.subtract(per_class[lab])
        rest = Counter({w: c for w, c in rest.items() if c > 0})
        out[lab] = {
            "n_tokens": int(sum(per_class[lab].values())),
            "n_types": len(per_class[lab]),
            "top_15_raw_frequency": [[w, int(c)] for w, c in per_class[lab].most_common(15)],
            "top_15_distinctive_log_odds_z": log_odds_dirichlet(
                per_class[lab], rest, corpus, top_k=15),
        }
    return out, per_class


# ---------------------------------------------------------------- overlap

def js_divergence(p: Counter, q: Counter, vocab: list) -> float:
    """Jensen-Shannon divergence in bits. 0 = identical, 1 = no shared support."""
    np_, nq = sum(p.values()), sum(q.values())
    total = 0.0
    for w in vocab:
        pi, qi = p.get(w, 0) / np_, q.get(w, 0) / nq
        m = 0.5 * (pi + qi)
        if pi > 0:
            total += 0.5 * pi * math.log2(pi / m)
        if qi > 0:
            total += 0.5 * qi * math.log2(qi / m)
    return total


def overlap_report(per_class: dict) -> dict:
    vocab = sorted({w for c in per_class.values() for w in c})
    sets = {lab: set(per_class[lab]) for lab in LABELS}

    jaccard, js = {}, {}
    for i, a in enumerate(LABELS):
        for b in LABELS[i + 1:]:
            key = f"{a}|{b}"
            inter = len(sets[a] & sets[b])
            jaccard[key] = round(inter / len(sets[a] | sets[b]), 5)
            js[key] = round(js_divergence(per_class[a], per_class[b], vocab), 5)

    # How many classes does each word appear in? Words in all six carry no
    # discriminative signal on their own.
    spread = Counter()
    for w in vocab:
        spread[sum(1 for lab in LABELS if w in sets[lab])] += 1

    most_similar = min(js, key=js.get)
    most_different = max(js, key=js.get)
    return {
        "vocabulary_jaccard": jaccard,
        "jensen_shannon_divergence_bits": js,
        "most_similar_pair": [most_similar, js[most_similar]],
        "most_different_pair": [most_different, js[most_different]],
        "types_by_number_of_classes_they_appear_in": {
            str(k): int(spread[k]) for k in range(1, 7)},
        "share_of_types_unique_to_one_class": round(spread[1] / len(vocab), 5),
        "share_of_types_in_all_six_classes": round(spread[6] / len(vocab), 5),
    }


# ---------------------------------------------------------------- main

def main() -> None:
    set_all_seeds()
    ensure_dirs()

    train = dataio.load_frozen("train")
    assert len(train) == 16000

    lengths = length_report(train)
    vocab = vocabulary_report(train)
    classes, per_class_counters = per_class_words(train)
    overlap = overlap_report(per_class_counters)

    payload = {
        "phase": "03_text_eda",
        "splits_used": ["train"],
        "split_policy": "train only. val and test were not read in this phase.",
        "tokenization": "whitespace split. The corpus is pre-normalized to 27 "
                        "characters (lowercase letters + space), so whitespace "
                        "tokens equal words exactly. Sub-word token counts are "
                        "deferred to Phase 7, where the model tokenizer is loaded.",
        "lengths": lengths,
        "vocabulary": vocab,
        "per_class_words": classes,
        "class_overlap": overlap,
    }
    path = save_metrics("phase03_text_eda", payload, device="cpu")

    print("=== LENGTHS (words, train) ===")
    print(json.dumps(lengths["overall"]["words"], indent=2))
    print("\nmean words per class:",
          json.dumps(lengths["length_signal"], indent=2))
    print("\n=== VOCABULARY ===")
    print(json.dumps({k: v for k, v in vocab.items()
                      if k not in ("top_30_tokens", "top_30_by_document_frequency")},
                     indent=2))
    print("\ntop 15 tokens:", vocab["top_30_tokens"][:15])
    print("\n=== DISTINCTIVE WORDS PER CLASS (log-odds z) ===")
    for lab in LABELS:
        words = [w for w, _z, _c in classes[lab]["top_15_distinctive_log_odds_z"]]
        print(f"  {lab:9s} {' '.join(words)}")
    print("\n=== CLASS OVERLAP ===")
    print(json.dumps(overlap, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
