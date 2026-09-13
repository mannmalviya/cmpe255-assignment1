"""Phase 8 — Unsupervised analysis: KMeans and DBSCAN on sentence embeddings.

SPLIT USED: train_clean only (15,923 rows), via the embeddings built in Phase 7.
Labels are used ONLY to score the clusters after the fact. No clustering step
sees a label.

This phase tests the prediction registered at the end of Phase 7:

  "KMeans will NOT recover the emotion labels. ARI against the true labels should
   be low. The clusters that emerge should be TOPICAL rather than emotional,
   because all-MiniLM-L6-v2 is trained for semantic textual similarity."

The prediction is reported as it lands, either way.
"""

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (adjusted_rand_score, completeness_score,
                             homogeneity_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.neighbors import NearestNeighbors

import viz
from config import DATA_PROCESSED, FIGURES, LABELS, SEED, ensure_dirs, set_all_seeds
from hardware import save_metrics

ARTIFACTS = DATA_PROCESSED / "features"
K_SWEEP = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
SIL_SAMPLE = 5000        # silhouette is O(n^2); sample it, with a fixed seed


def external_scores(y_true, y_pred) -> dict:
    """How well does a clustering recover the known labels?

    ARI is chance-corrected: 0 means "no better than random agreement", 1 means
    identical partitions. NMI is not chance-corrected but is scale-free.
    Homogeneity asks "is each cluster pure?"; completeness asks "is each class
    kept together?". Reporting both separates the two ways a clustering can fail.
    """
    return {
        "adjusted_rand_index": round(float(adjusted_rand_score(y_true, y_pred)), 4),
        "normalized_mutual_info": round(
            float(normalized_mutual_info_score(y_true, y_pred)), 4),
        "homogeneity": round(float(homogeneity_score(y_true, y_pred)), 4),
        "completeness": round(float(completeness_score(y_true, y_pred)), 4),
    }


def cluster_purity(y_true, y_pred) -> dict:
    """Assign each cluster its majority label; what accuracy does that give?

    This is the most generous possible reading of a clustering as a classifier,
    since the mapping is chosen with the labels in hand. If even this is low, the
    clustering carries no class information.
    """
    total, mapping = 0, {}
    for c in np.unique(y_pred):
        if c == -1:                      # DBSCAN noise
            continue
        counts = Counter(y_true[y_pred == c])
        lab, n = counts.most_common(1)[0]
        mapping[int(c)] = {"majority_label": lab, "size": int((y_pred == c).sum()),
                           "purity": round(n / (y_pred == c).sum(), 4)}
        total += n
    return {"majority_map_accuracy": round(total / len(y_true), 4),
            "distinct_labels_claimed": len({v["majority_label"] for v in mapping.values()}),
            "clusters": mapping}


def top_words(texts, corpus_counts, a0, n_corpus_tokens, k=8) -> list:
    """Distinctive words for a cluster, by the Phase 3 log-odds estimator."""
    c = Counter(t for text in texts for t in text.split())
    n_i = sum(c.values())
    n_j = n_corpus_tokens - n_i
    if n_i == 0 or n_j <= 0:
        return []
    scored = []
    for w, aw in corpus_counts.items():
        yi = c.get(w, 0)
        yj = aw - yi
        num_i, num_j = yi + aw, yj + aw
        den_i, den_j = n_i + a0 - num_i, n_j + a0 - num_j
        if den_i <= 0 or den_j <= 0:
            continue
        delta = math.log(num_i / den_i) - math.log(num_j / den_j)
        z = delta / math.sqrt(1.0 / num_i + 1.0 / num_j)
        scored.append((w, z))
    scored.sort(key=lambda r: -r[1])
    return [w for w, _ in scored[:k]]


def main() -> None:
    set_all_seeds()
    ensure_dirs()
    viz.apply_style()

    train = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
    emb = np.load(ARTIFACTS / "emb_train_clean.npy")
    y = train["label"].to_numpy()
    assert len(emb) == len(train) == 15923

    rng = np.random.default_rng(SEED)
    sil_idx = rng.choice(len(emb), SIL_SAMPLE, replace=False)

    # ---------------------------------------------------------- KMeans sweep
    sweep = []
    labels_at_6 = None
    for k in K_SWEEP:
        t0 = time.perf_counter()
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(emb)
        secs = time.perf_counter() - t0
        pred = km.labels_
        row = {"k": k, "fit_seconds": round(secs, 2),
               "inertia": round(float(km.inertia_), 2),
               "silhouette_sampled": round(float(silhouette_score(
                   emb[sil_idx], pred[sil_idx], metric="cosine")), 4),
               **external_scores(y, pred)}
        sweep.append(row)
        if k == 6:
            labels_at_6 = pred
        print(f"  k={k:>2}  sil {row['silhouette_sampled']:+.4f}  "
              f"ARI {row['adjusted_rand_index']:+.4f}  "
              f"NMI {row['normalized_mutual_info']:.4f}  "
              f"hom {row['homogeneity']:.4f}  comp {row['completeness']:.4f}  "
              f"({secs:.1f}s)")

    # ------------------------------------------------- what IS in the clusters?
    corpus_counts = Counter(t for text in train["text"] for t in text.split())
    a0 = sum(corpus_counts.values())
    purity6 = cluster_purity(y, labels_at_6)
    cluster_detail = []
    for c in range(6):
        m = labels_at_6 == c
        counts = Counter(y[m])
        cluster_detail.append({
            "cluster": c, "size": int(m.sum()),
            "label_mix": {lab: round(counts.get(lab, 0) / m.sum(), 3) for lab in LABELS},
            "majority_label": counts.most_common(1)[0][0],
            "purity": round(counts.most_common(1)[0][1] / m.sum(), 3),
            "top_words": top_words(train.loc[m, "text"], corpus_counts, a0, a0),
            "examples": train.loc[m, "text"].head(3).tolist(),
        })

    # contingency table, rows = cluster, cols = true label
    cont = np.zeros((6, len(LABELS)), dtype=int)
    for ci in range(6):
        for li, lab in enumerate(LABELS):
            cont[ci, li] = int(((labels_at_6 == ci) & (y == lab)).sum())

    # ---------------------------------------------------------------- DBSCAN
    # eps is chosen from the k-distance curve rather than guessed: for
    # min_samples = m, plot each point's distance to its m-th nearest neighbour
    # and look for the knee.
    min_samples = 10
    nn = NearestNeighbors(n_neighbors=min_samples, metric="cosine").fit(emb)
    kdist = np.sort(nn.kneighbors(emb)[0][:, -1])
    knee_pcts = [50, 75, 90, 95, 99]
    kdist_pcts = {str(p): round(float(np.percentile(kdist, p)), 4) for p in knee_pcts}

    db_runs = []
    for eps in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        t0 = time.perf_counter()
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine",
                    n_jobs=-1).fit(emb)
        secs = time.perf_counter() - t0
        pred = db.labels_
        n_clusters = int(len({c for c in pred if c != -1}))
        noise = float((pred == -1).mean())
        row = {"eps": eps, "min_samples": min_samples, "fit_seconds": round(secs, 1),
               "n_clusters": n_clusters, "noise_fraction": round(noise, 4)}
        if n_clusters >= 2:
            row.update(external_scores(y, pred))
            row["largest_cluster_share"] = round(float(
                max(Counter(pred[pred != -1]).values()) / len(pred)), 4)
        db_runs.append(row)
        print(f"  eps={eps:.2f}  clusters {n_clusters:>4}  noise {noise:>6.1%}  "
              f"ARI {row.get('adjusted_rand_index', float('nan')):+.4f}  ({secs:.1f}s)")

    # ---------------------------------------------------------------- figures
    ks = [r["k"] for r in sweep]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # The y-axis is held on the interpretable scale, not auto-fitted. Auto-scaling
    # would magnify variation of 0.02 into a dramatic curve and contradict the
    # finding, which is that every value is near zero.
    ax1.plot(ks, [r["silhouette_sampled"] for r in sweep], "o-",
             color=viz.PRIMARY, zorder=4)
    ax1.set_ylim(-0.05, 0.65)
    ax1.axhspan(0.5, 0.65, color=viz.REFERENCE, alpha=0.5, zorder=1)
    ax1.axhspan(0.25, 0.5, color=viz.REFERENCE, alpha=0.25, zorder=1)
    ax1.text(20, 0.565, "strong structure  ", ha="right", fontsize=7, color=viz.INK_2)
    ax1.text(20, 0.36, "moderate structure  ", ha="right", fontsize=7, color=viz.INK_2)
    ax1.axvline(6, color=viz.INK_2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax1.text(6.4, 0.62, "k = 6 (label count)", fontsize=7.5, color=viz.INK_2, va="top")
    ax1.annotate(f"all k sit at {min(r['silhouette_sampled'] for r in sweep):.02f}"
                 f"-{max(r['silhouette_sampled'] for r in sweep):.02f}",
                 xy=(10, 0.029), xytext=(11, 0.17), fontsize=7.5, color=viz.INK_2,
                 arrowprops=dict(arrowstyle="-", color=viz.MUTED, lw=0.9))
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("silhouette (cosine, 5,000-row sample)")
    viz.value_grid(ax1, "y")
    viz.titles(ax1, "a.  The embedding space has no natural cluster count",
               "silhouette stays near zero at every k: no well-separated groups exist")

    ax2.plot(ks, [r["adjusted_rand_index"] for r in sweep], "o-",
             color=viz.PRIMARY, label="Adjusted Rand Index", zorder=3)
    ax2.plot(ks, [r["normalized_mutual_info"] for r in sweep], "s--",
             color=viz.ACCENT, label="Normalized Mutual Information", zorder=3)
    ax2.axvline(6, color=viz.INK_2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax2.text(6.4, 0.96, "k = 6 (label count)", fontsize=7.5, color=viz.INK_2, va="top")
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("k (number of clusters)")
    ax2.set_ylabel("agreement with the true emotion labels")
    viz.value_grid(ax2, "y")
    ax2.legend(loc="upper right")
    viz.titles(ax2, "b.  Clusters do not recover the emotions",
               "1.0 would be a perfect match; 0.0 is chance for ARI")
    viz.caption(fig, "Split: train_clean (15,923 rows), embeddings from Phase 7. "
                     "Labels used only to score, never to cluster.")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig08_clustering_metrics.png")
    plt.close(fig)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", viz.SEQ_BLUE[0:11])
    row_pct = cont / cont.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    im = ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=60, aspect="auto")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(6), [f"cluster {i}\nn={cont[i].sum():,}" for i in range(6)])
    ax.tick_params(length=0, labelsize=8, colors=viz.INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color=viz.SURFACE, linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    for i in range(6):
        for j in range(len(LABELS)):
            v = row_pct[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > 38 else viz.INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("% of the cluster's rows", fontsize=8, color=viz.INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, color=viz.MUTED, labelcolor=viz.MUTED)
    viz.titles(ax, "Every cluster is a mixture of every emotion",
               "rows sum to 100%. A clustering that found the emotions would show "
               "one dark cell per row.")
    viz.caption(fig, "Split: train_clean. KMeans k=6 on Phase 7 embeddings; "
                     "columns ordered by class size.")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig09_cluster_label_contingency.png")
    plt.close(fig)

    # --------------------------------------------- a defect the clustering found
    # Cluster 5's distinctive words include "http" and "href". Phases 2 and 5
    # certified this corpus clean, but both checked CHARACTERS. Upstream
    # punctuation stripping turned '<a href="http://x">' into the bare tokens
    # 'a href http x', so HTML markup survived as ordinary-looking words.
    markers = ["http", "href", "www", "amp", "rel", "target", "blog"]
    residue = {w: int(train["text"].str.contains(rf"\b{w}\b", regex=True).sum())
               for w in markers}
    any_html = train["text"].str.contains(r"\b(?:http|href|www|amp)\b", regex=True)
    html_issue = {
        "what": "HTML/URL residue surviving as word tokens",
        "found_by": "KMeans cluster 5, whose distinctive words were 'http href "
                    "they the popular are blog book'",
        "missed_by": "Phase 2 (character inventory) and Phase 5 (normalization "
                     "no-op check) - both are correct at the character level and "
                     "blind at the token level",
        "marker_counts": residue,
        "rows_with_any_marker": int(any_html.sum()),
        "share_of_train": round(float(any_html.mean()), 5),
        "examples": train.loc[any_html, "text"].head(3).tolist(),
        "assessment": (
            "1.2% of rows. Left in place, consistent with the Phase 6 flag-don't-"
            "remove policy and the Phase 5 rule that cleaning must not be able to "
            "flatter a result. These tokens are near-uniformly distributed across "
            "classes, so they add noise rather than spurious signal; min_df and IDF "
            "both down-weight them. Recorded as a limitation, and as evidence that "
            "unsupervised analysis earns its place in the pipeline."),
        "class_lift": {lab: round(float((train.loc[any_html, "label"] == lab).mean() /
                                        (train["label"] == lab).mean()), 2)
                       for lab in LABELS},
    }

    # ---------------------------------------------------------------- payload
    at6 = next(r for r in sweep if r["k"] == 6)
    payload = {
        "phase": "08_clustering",
        "splits_used": ["train_clean"],
        "split_policy": "train_clean only, via Phase 7 embeddings. Labels are used "
                        "only to score clusters after the fact; no clustering step "
                        "sees a label.",
        "representation": "sentence-transformers/all-MiniLM-L6-v2, 384-dim, "
                          "L2-normalized (Phase 7)",
        "prediction_under_test": (
            "Phase 7 predicted KMeans would NOT recover the emotion labels, and that "
            "clusters would be topical rather than emotional."),
        "kmeans_sweep": sweep,
        "kmeans_k6": {
            **at6,
            "majority_map": purity6,
            "contingency_counts": {f"cluster_{i}": {lab: int(cont[i, j])
                                                    for j, lab in enumerate(LABELS)}
                                   for i in range(6)},
            "clusters": cluster_detail,
        },
        "dbscan": {
            "min_samples": min_samples,
            "k_distance_percentiles": kdist_pcts,
            "eps_selection": "swept around the k-distance curve rather than guessed",
            "runs": db_runs,
        },
        "discovered_data_issue": html_issue,
        "figures": ["fig08_clustering_metrics.png",
                    "fig09_cluster_label_contingency.png"],
    }
    path = save_metrics("phase08_clustering", payload, device="cpu")

    print("\n=== KMeans k=6 vs the true labels ===")
    print(f"  ARI {at6['adjusted_rand_index']}  NMI {at6['normalized_mutual_info']}  "
          f"homogeneity {at6['homogeneity']}  completeness {at6['completeness']}")
    print(f"  majority-map accuracy {purity6['majority_map_accuracy']:.1%}  "
          f"(clusters claim only {purity6['distinct_labels_claimed']} of 6 labels)")
    print("\n=== WHAT IS IN EACH CLUSTER ===")
    for c in cluster_detail:
        print(f"  cluster {c['cluster']}  n={c['size']:>5}  "
              f"majority {c['majority_label']:<8} purity {c['purity']:.2f}")
        print(f"      words: {' '.join(c['top_words'])}")
        print(f"      e.g.:  {c['examples'][0][:78]}")
    print("\n=== DEFECT FOUND BY THE CLUSTERING ===")
    print(f"  HTML/URL residue in {html_issue['rows_with_any_marker']} rows "
          f"({html_issue['share_of_train']:.2%}): {html_issue['marker_counts']}")
    print(f"  class lift: {html_issue['class_lift']}")
    print(f"  e.g. {html_issue['examples'][0][:80]}")

    print(f"\n  k-distance percentiles (min_samples={min_samples}): {kdist_pcts}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
