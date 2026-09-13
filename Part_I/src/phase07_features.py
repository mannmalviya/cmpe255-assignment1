"""Phase 7 — Feature engineering: TF-IDF and sentence embeddings.

SPLIT USED: every fit is on train_clean ONLY (15,923 rows). val and test are
transformed with the already-fitted objects and never influence a vocabulary, an
IDF weight, or a model parameter.

Two independent representations are built, because the study compares two model
families that consume different inputs:
  A. TF-IDF  - sparse, lexical, fitted to this corpus. Feeds Phase 12.
  B. Sentence embeddings - dense, semantic, pre-trained elsewhere. Feeds
     Phases 8 (clustering) and 9 (LSH).

Device note: embeddings are computed on CPU. They are an ANALYSIS input here,
not a timed model, so the device does not enter any latency comparison. Any
embedding-based model that reaches the Phase 16 table is re-timed there under
the full latency protocol, on a stated device.
"""

import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DATA_PROCESSED, LABELS, SEED, ensure_dirs, set_all_seeds
from hardware import save_metrics

ARTIFACTS = DATA_PROCESSED / "features"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ = 128          # the budget Phase 2 predicted would be sufficient


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2


# ------------------------------------------------------------- A. TF-IDF

def build_tfidf(train, val, test) -> dict:
    """Fit three TF-IDF configurations on train_clean and compare their cost.

    The winner is not chosen here. Phase 12 cross-validates the choice on
    train+val; this phase only measures what each option costs.
    """
    configs = {
        "unigram_mindf1": dict(ngram_range=(1, 1), min_df=1),
        "unigram_mindf2": dict(ngram_range=(1, 1), min_df=2),
        "bigram_mindf2": dict(ngram_range=(1, 2), min_df=2),
    }
    out = {}
    for name, kw in configs.items():
        vec = TfidfVectorizer(sublinear_tf=True, lowercase=False,
                              token_pattern=r"\S+", **kw)
        t0 = time.perf_counter()
        Xtr = vec.fit_transform(train["text"])
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        Xva = vec.transform(val["text"])
        Xte = vec.transform(test["text"])
        transform_s = time.perf_counter() - t0

        path = ARTIFACTS / f"tfidf_{name}.joblib"
        joblib.dump(vec, path, compress=3)

        # How much of the held-out vocabulary is simply absent from the fitted
        # space? Rows that land on an all-zero vector cannot be classified at all.
        oov_rows_test = int((Xte.getnnz(axis=1) == 0).sum())

        out[name] = {
            "params": {**kw, "sublinear_tf": True, "lowercase": False,
                       "token_pattern": r"\S+"},
            "vocabulary_size": int(len(vec.vocabulary_)),
            "train_shape": list(Xtr.shape),
            "train_nnz": int(Xtr.nnz),
            "train_density": round(Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1]), 8),
            "mean_nonzeros_per_row_train": round(Xtr.nnz / Xtr.shape[0], 2),
            "mean_nonzeros_per_row_test": round(Xte.nnz / Xte.shape[0], 2),
            "fit_transform_seconds_train": round(fit_s, 3),
            "transform_seconds_val_plus_test": round(transform_s, 3),
            "vectorizer_disk_mb": round(path.stat().st_size / 1024**2, 3),
            "train_matrix_memory_mb": round(
                (Xtr.data.nbytes + Xtr.indices.nbytes + Xtr.indptr.nbytes) / 1024**2, 3),
            "test_rows_with_all_zero_vector": oov_rows_test,
        }
        if name == "bigram_mindf2":
            joblib.dump({"train": Xtr, "val": Xva, "test": Xte},
                        ARTIFACTS / "tfidf_bigram_mindf2_matrices.joblib", compress=3)
    return out


# --------------------------------------------------------- B. embeddings

def build_embeddings(train, val, test) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(SEED)
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    model.max_seq_length = MAX_SEQ
    tok = model.tokenizer

    # Verify the Phase 2 claim against the real sub-word tokenizer, rather than
    # trusting a whitespace-word estimate.
    lens = np.array([len(tok.encode(t, add_special_tokens=True))
                     for t in train["text"]])
    token_stats = {
        "tokenizer": tok.__class__.__name__,
        "max_seq_length_set": MAX_SEQ,
        "train_subword_tokens": {
            "min": int(lens.min()), "median": float(np.median(lens)),
            "mean": round(float(lens.mean()), 2),
            "p95": float(np.percentile(lens, 95)),
            "p99": float(np.percentile(lens, 99)),
            "max": int(lens.max()),
        },
        "rows_exceeding_max_seq": int((lens > MAX_SEQ).sum()),
        "subword_to_word_ratio": round(float(lens.mean()) /
                                       float(train["text"].str.split().str.len().mean()), 3),
    }

    out = {"model": EMBED_MODEL, "device": "cpu", "tokenization": token_stats,
           "splits": {}}
    for name, df in (("train_clean", train), ("val", val), ("test", test)):
        t0 = time.perf_counter()
        emb = model.encode(df["text"].tolist(), batch_size=64,
                           convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False)
        secs = time.perf_counter() - t0
        np.save(ARTIFACTS / f"emb_{name}.npy", emb.astype(np.float32))
        out["splits"][name] = {
            "n_rows": int(emb.shape[0]), "dim": int(emb.shape[1]),
            "encode_seconds": round(secs, 2),
            "rows_per_second": round(len(df) / secs, 1),
            "npy_disk_mb": round(emb.nbytes / 1024**2, 2),
            "l2_normalized": bool(np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4)),
        }

    cache = Path.home() / ".cache" / "huggingface"
    out["model_disk_mb"] = round(dir_size_mb(cache), 1) if cache.exists() else None
    out["parameters_millions"] = round(
        sum(p.numel() for p in model.parameters()) / 1e6, 2)
    return out


def embedding_separability(train) -> dict:
    """Do the embeddings already know about the labels, before any fitting?

    Mean cosine similarity within a class against the corpus mean. A ratio near
    1.0 would mean the space carries no class structure; well above 1.0 means a
    classifier has something to work with. This previews Phase 8.
    """
    rng = np.random.default_rng(SEED)
    emb = np.load(ARTIFACTS / "emb_train_clean.npy")
    labels = train["label"].to_numpy()

    idx = rng.choice(len(emb), size=4000, replace=False)   # O(n^2), so subsample
    e, y = emb[idx], labels[idx]
    sim = e @ e.T
    np.fill_diagonal(sim, np.nan)

    overall = float(np.nanmean(sim))
    per_class, nearest = {}, {}
    for lab in LABELS:
        m = y == lab
        if m.sum() < 2:
            continue
        within = float(np.nanmean(sim[np.ix_(m, m)]))
        per_class[lab] = {
            "n_sampled": int(m.sum()),
            "mean_within_class_cosine": round(within, 4),
            "ratio_to_corpus_mean": round(within / overall, 3),
        }
        others = {o: float(np.nanmean(sim[np.ix_(m, y == o)]))
                  for o in LABELS if o != lab and (y == o).sum() > 0}
        best = max(others, key=others.get)
        nearest[lab] = {"nearest_other_class": best,
                        "mean_cosine": round(others[best], 4),
                        "gap_within_minus_nearest": round(within - others[best], 4)}

    # Mean cosine compares CENTROIDS and is a weak statistic in high dimensions.
    # What a classifier actually exploits is the LOCAL neighbourhood, so measure
    # that directly: of a point's k nearest neighbours, how many share its label?
    k = 10
    order = np.argsort(-np.nan_to_num(sim, nan=-np.inf), axis=1)[:, :k]
    neigh = y[order]
    purity = (neigh == y[:, None]).mean(axis=1)
    knn_vote = np.array([max(set(row), key=list(row).count) for row in neigh])

    per_class_purity = {}
    for lab in LABELS:
        m = y == lab
        if m.sum() < 2:
            continue
        per_class_purity[lab] = {
            "mean_knn_purity": round(float(purity[m].mean()), 4),
            "knn_vote_accuracy": round(float((knn_vote[m] == lab).mean()), 4),
        }

    chance = float(sum((y == lab).mean() ** 2 for lab in LABELS))
    return {
        "sample_size": len(idx),
        "mean_cosine_all_pairs": round(overall, 4),
        "per_class": per_class,
        "nearest_confusable_class": nearest,
        "class_mean_cosine_matrix": {
            a: {b: round(float(np.nanmean(sim[np.ix_(y == a, y == b)])), 4)
                for b in LABELS if (y == b).sum() > 0}
            for a in LABELS if (y == a).sum() > 0},
        "local_neighbourhood": {
            "k": k,
            "mean_knn_purity_overall": round(float(purity.mean()), 4),
            "knn_vote_accuracy_overall": round(float((knn_vote == y).mean()), 4),
            "purity_expected_by_chance": round(chance, 4),
            "per_class": per_class_purity,
            "note": ("Measured on the raw pre-trained embeddings with no fitting "
                     "of any kind. It is an upper-bound-free preview of how much "
                     "class structure the representation already carries locally, "
                     "and it is computed on the 4,000-row sample, not on test."),
        },
    }


def main() -> None:
    set_all_seeds()
    ensure_dirs()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
    val = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    assert len(train) == 15923 and len(val) == 2000 and len(test) == 2000

    tfidf = build_tfidf(train, val, test)
    emb = build_embeddings(train, val, test)
    sep = embedding_separability(train)

    payload = {
        "phase": "07_features",
        "splits_used": ["train_clean (fit)", "val (transform only)",
                        "test (transform only)"],
        "split_policy": "every fit uses train_clean only. val and test are "
                        "transformed with the fitted objects; they never inform a "
                        "vocabulary, an IDF weight or a model parameter.",
        "tfidf": tfidf,
        "embeddings": emb,
        "embedding_separability": sep,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "artifacts_dir": str(ARTIFACTS.relative_to(DATA_PROCESSED.parent.parent)),
        "artifacts_disk_mb": round(dir_size_mb(ARTIFACTS), 2),
    }
    path = save_metrics("phase07_features", payload, device="cpu")

    print("=== A. TF-IDF (fitted on train_clean only) ===")
    for name, r in tfidf.items():
        print(f"  {name:18s} vocab {r['vocabulary_size']:>7,}  "
              f"nnz/row {r['mean_nonzeros_per_row_train']:>6.2f}  "
              f"density {r['train_density']:.2e}  "
              f"fit {r['fit_transform_seconds_train']:.2f}s  "
              f"disk {r['vectorizer_disk_mb']:.2f} MB  "
              f"zero-vector test rows {r['test_rows_with_all_zero_vector']}")

    print("\n=== B. SENTENCE EMBEDDINGS ===")
    print(f"  model {emb['model']}  ({emb['parameters_millions']}M params, CPU)")
    t = emb["tokenization"]["train_subword_tokens"]
    print(f"  sub-word tokens: median {t['median']:.0f}  p95 {t['p95']:.0f}  "
          f"p99 {t['p99']:.0f}  max {t['max']}  "
          f"(rows over {MAX_SEQ}: {emb['tokenization']['rows_exceeding_max_seq']})")
    print(f"  sub-word / word ratio: {emb['tokenization']['subword_to_word_ratio']}")
    for n, r in emb["splits"].items():
        print(f"  {n:12s} {r['n_rows']:>6,} x {r['dim']}  "
              f"{r['encode_seconds']:>6.2f}s  {r['rows_per_second']:>7.1f} rows/s  "
              f"{r['npy_disk_mb']:>5.2f} MB")

    print("\n=== EMBEDDING SEPARABILITY (before any fitting) ===")
    print(f"  mean cosine over all pairs: {sep['mean_cosine_all_pairs']}")
    for lab in LABELS:
        if lab not in sep["per_class"]:
            continue
        p, n = sep["per_class"][lab], sep["nearest_confusable_class"][lab]
        print(f"  {lab:9s} within {p['mean_within_class_cosine']:.3f} "
              f"({p['ratio_to_corpus_mean']:.2f}x corpus)   "
              f"nearest other: {n['nearest_other_class']:<9s} "
              f"gap {n['gap_within_minus_nearest']:+.3f}")

    ln = sep["local_neighbourhood"]
    print(f"\n=== LOCAL NEIGHBOURHOOD (k={ln['k']}, raw embeddings, no fitting) ===")
    print(f"  mean kNN label purity : {ln['mean_knn_purity_overall']:.1%}  "
          f"(chance {ln['purity_expected_by_chance']:.1%})")
    print(f"  kNN majority-vote acc : {ln['knn_vote_accuracy_overall']:.1%}")
    for lab in LABELS:
        if lab in ln["per_class"]:
            r = ln["per_class"][lab]
            print(f"    {lab:9s} purity {r['mean_knn_purity']:.1%}   "
                  f"vote acc {r['knn_vote_accuracy']:.1%}")

    print(f"\n  peak RSS {payload['peak_rss_mb']:.0f} MB   "
          f"artifacts {payload['artifacts_disk_mb']:.1f} MB")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
