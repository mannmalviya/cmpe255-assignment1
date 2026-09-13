"""Phase 4 — Data visualization. Six publication-quality figures.

SPLIT USED: train only for every analysis figure. Figure 1b shows class SHARES
for all three splits, which is permitted because Phase 2 already reported those
counts as part of data understanding; no text from val or test is read here.

One idea per figure. Every figure is readable in print and in grayscale, either
because it uses a single hue or because identity is carried by direct labels and
facet titles rather than by color alone.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

import data as dataio
import viz
from config import FIGURES, LABELS, METRICS, ensure_dirs, set_all_seeds
from hardware import save_metrics

ORDER = ["joy", "sadness", "anger", "fear", "love", "surprise"]  # descending size


def fig01_class_balance(train, all_shares):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # (a) the imbalance itself -------------------------------------------
    counts = [int((train["label"] == lab).sum()) for lab in ORDER]
    y = np.arange(len(ORDER))[::-1]
    bars = ax1.barh(y, counts, height=0.62, color=viz.PRIMARY)
    ax1.set_yticks(y, ORDER)
    ax1.set_xlim(0, max(counts) * 1.22)
    viz.value_grid(ax1, "x")
    viz.rounded_bars(ax1, bars, "h")
    for yi, c in zip(y, counts):
        ax1.text(c + max(counts) * 0.02, yi, f"{c:,}  ({c/len(train):.1%})",
                 va="center", ha="left", fontsize=8, color=viz.INK_2)
    ax1.set_xlabel("training documents")
    viz.titles(ax1, "a.  Training set is imbalanced 9.4 : 1",
               "joy has 5,362 documents; surprise has 572")

    # (b) do the three splits agree? --------------------------------------
    splits = ["train", "val", "test"]
    x = np.arange(len(ORDER))
    w = 0.26
    for i, sp in enumerate(splits):
        vals = [all_shares[sp][lab] * 100 for lab in ORDER]
        bars = ax2.bar(x + (i - 1) * w, vals, width=w * 0.92,
                       color=viz.SERIES[i], label=sp, zorder=3)
        viz.rounded_bars(ax2, bars, "v")
        # Relief rule: aqua is below 3:1 on this surface, so values are labelled.
        for xi, v in zip(x + (i - 1) * w, vals):
            ax2.text(xi, v + 0.7, f"{v:.1f}", ha="center", va="bottom",
                     fontsize=6.2, color=viz.INK_2, rotation=90)
    ax2.set_xticks(x, ORDER)
    ax2.set_ylim(0, 42)
    ax2.set_ylabel("share of split (%)")
    viz.value_grid(ax2, "y")
    ax2.legend([viz.swatch(c) for c in viz.SERIES], splits, loc="upper right",
               ncol=3, columnspacing=1.0, handlelength=1.0, handletextpad=0.5)
    viz.titles(ax2, "b.  The three splits carry the same class mix",
               "total variation distance train vs test = 0.015; largest gap 1.2 pp")

    viz.caption(fig, "Split: train (panel a); class shares for train/val/test "
                     "(panel b, counts from Phase 2). Source: "
                     "reports/metrics/phase02_data_understanding.json")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01_class_balance.png")
    plt.close(fig)


def fig02_length_distribution(train):
    words = train["text"].str.split().str.len().to_numpy()
    fig, ax = plt.subplots(figsize=(8.0, 4.0))

    bins = np.arange(0, 70, 2)
    n, _, patches = ax.hist(words, bins=bins, color=viz.PRIMARY, zorder=3)
    for p in patches:                      # 2px surface gap between adjacent fills
        p.set_edgecolor(viz.SURFACE)
        p.set_linewidth(1.0)

    # Reference lines are labelled in the headroom above the bars, never over a
    # fill, so the muted ink keeps its contrast.
    top = max(n) * 1.16
    ax.set_ylim(0, top)
    for p, lab in [(50, "median 17"), (95, "p95 41"), (99, "p99 52")]:
        v = np.percentile(words, p)
        ax.axvline(v, color=viz.INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
        ax.text(v + 0.7, top * 0.985, lab, fontsize=7.5, color=viz.INK_2, va="top")

    ax.set_xlim(0, 68)
    ax.set_xlabel("words per document")
    ax.set_ylabel("documents")
    viz.value_grid(ax, "y")
    viz.titles(ax, "Documents are short, with a thin long tail",
               "median 17 words, max 66; Pearson skewness +0.59. "
               "A 128-token sequence covers the entire corpus.")
    viz.caption(fig, "Split: train (16,000 documents). Bin width 2 words.")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02_length_distribution.png")
    plt.close(fig)


def fig03_length_by_class(train):
    """Small multiples: each class against the whole corpus, same axes."""
    words = train["text"].str.split().str.len()
    bins = np.arange(0, 68, 3)
    overall, _ = np.histogram(words, bins=bins, density=True)
    centres = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.0), sharex=True, sharey=True)
    means = {}
    for ax, lab in zip(axes.ravel(), ORDER):
        w = words[train["label"] == lab]
        means[lab] = w.mean()
        h, _ = np.histogram(w, bins=bins, density=True)
        ax.fill_between(centres, overall, color=viz.REFERENCE, step="mid",
                        zorder=2, label="all classes")
        ax.step(centres, h, where="mid", color=viz.PRIMARY, linewidth=1.8,
                zorder=3, label=lab)
        ax.axvline(w.mean(), color=viz.ACCENT, linewidth=1.4, zorder=4)
        ax.set_title(f"{lab}   mean {w.mean():.1f} words", loc="left",
                     fontsize=9, color=viz.INK)
        ax.set_xlim(0, 66)
        ax.grid(axis="y", color=viz.GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("words per document")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")

    handles = [viz.swatch(viz.REFERENCE),
               plt.Line2D([], [], color=viz.PRIMARY, lw=1.8),
               plt.Line2D([], [], color=viz.ACCENT, lw=1.4)]
    fig.legend(handles, ["all classes (reference)", "this class", "class mean"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))

    spread = max(means.values()) - min(means.values())
    fig.suptitle("Document length carries no class signal", x=0.008, ha="left",
                 fontsize=11, fontweight="bold", color=viz.INK, y=1.035)
    fig.text(0.008, 1.0, f"Between-class spread of mean length is {spread:.2f} words, "
                         "against a within-corpus standard deviation of 10.99 words.",
             ha="left", va="top", fontsize=8, color=viz.INK_2)
    viz.caption(fig, "Split: train. Shared axes across panels. Bin width 3 words.")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03_length_by_class.png")
    plt.close(fig)


def fig04_vocabulary_structure(train, eda):
    tokens = [t for text in train["text"] for t in text.split()]
    counts = Counter(tokens)
    freqs = np.array(sorted(counts.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)
    n_tokens = len(tokens)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))

    # (a) Zipf ------------------------------------------------------------
    slope = eda["vocabulary"]["zipf_fit"]["slope"]
    r2 = eda["vocabulary"]["zipf_fit"]["r_squared"]
    inter = np.polyfit(np.log(ranks), np.log(freqs), 1)[1]
    ax1.loglog(ranks, freqs, color=viz.PRIMARY, linewidth=1.4, zorder=3)
    ax1.loglog(ranks, np.exp(inter + slope * np.log(ranks)), color=viz.ACCENT,
               linewidth=1.4, linestyle=(0, (4, 3)), zorder=4,
               label=f"fit: slope {slope:.2f}, $R^2$ {r2:.3f}")
    for w in ["i", "feel", "the", "amazed"]:
        r = int(np.where(freqs == counts[w])[0][0]) + 1
        ax1.plot(r, counts[w], "o", color=viz.INK_2, markersize=4, zorder=5)
        ax1.annotate(f" {w}", (r, counts[w]), fontsize=7.5, color=viz.INK_2,
                     va="center")
    ax1.set_xlabel("rank (log)")
    ax1.set_ylabel("frequency (log)")
    ax1.grid(which="major", color=viz.GRID, linewidth=0.6)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right")
    viz.titles(ax1, "a.  The corpus obeys Zipf's law",
               "slope steeper than the canonical -1: the template repeats")

    # (b) coverage --------------------------------------------------------
    cum = np.cumsum(freqs) / n_tokens
    ax2.plot(ranks, cum * 100, color=viz.PRIMARY, linewidth=2.0, zorder=3)
    ax2.set_xscale("log")
    for p, note in [(0.5, "49 types"), (0.8, "620"), (0.9, "1,782"), (0.95, "4,179")]:
        k = int(np.searchsorted(cum, p)) + 1
        ax2.plot(k, p * 100, "o", color=viz.ACCENT, markersize=5, zorder=5)
        ax2.annotate(f"  {note}", (k, p * 100), fontsize=7.5, color=viz.INK_2,
                     va="center")
    hapax = eda["vocabulary"]["hapax_legomena"]
    n_types = eda["vocabulary"]["n_types"]
    ax2.axvspan(n_types - hapax, n_types, color=viz.REFERENCE, alpha=0.55, zorder=1)
    ax2.text(n_types - hapax * 0.55, 22,
             f"{hapax:,} hapax\n({hapax/n_types:.0%} of types,\nlast 0.6% of tokens)",
             fontsize=7.5, color=viz.INK_2, ha="center")
    ax2.set_xlabel("number of word types kept, most frequent first (log)")
    ax2.set_ylabel("% of all tokens covered")
    ax2.set_ylim(0, 104)
    viz.value_grid(ax2, "y")
    viz.titles(ax2, "b.  49 types cover half of all tokens",
               "IDF suppresses the head automatically, so no stopword list is used")

    viz.caption(fig, "Split: train. 306,661 tokens, 15,212 types. Source: "
                     "reports/metrics/phase03_text_eda.json")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04_vocabulary_structure.png")
    plt.close(fig)


def fig05_distinctive_words(eda):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    top = 10
    zmax = max(z for lab in ORDER
               for _w, z, _c in eda["per_class_words"][lab]["top_15_distinctive_log_odds_z"][:top])
    for ax, lab in zip(axes.ravel(), ORDER):
        rows = eda["per_class_words"][lab]["top_15_distinctive_log_odds_z"][:top][::-1]
        words = [r[0] for r in rows]
        zs = [r[1] for r in rows]
        y = np.arange(len(words))
        bars = ax.barh(y, zs, height=0.6, color=viz.PRIMARY, zorder=3)
        ax.set_yticks(y, words, fontsize=8)
        ax.set_xlim(0, zmax * 1.08)
        viz.value_grid(ax, "x")
        viz.rounded_bars(ax, bars, "h")
        n = eda["per_class_words"][lab]["n_tokens"]
        ax.set_title(f"{lab}   ({n:,} tokens)", loc="left", fontsize=9, color=viz.INK)
        ax.tick_params(axis="y", length=0)
    for ax in axes[1]:
        ax.set_xlabel("log-odds z  (over-use vs rest of corpus)")

    fig.suptitle("Every class is separated by emotion adjectives, not by topic",
                 x=0.008, ha="left", fontsize=11, fontweight="bold",
                 color=viz.INK, y=1.045)
    fig.text(0.008, 1.012,
             "Log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008). "
             "Bars are comparable within a panel, not across panels: a smaller class "
             "yields larger z.",
             ha="left", va="top", fontsize=8, color=viz.INK_2)
    viz.caption(fig, "Split: train. Source: reports/metrics/phase03_text_eda.json")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05_distinctive_words.png")
    plt.close(fig)


def fig06_class_overlap(eda):
    """Lower triangle only, with the empty first row and last column trimmed."""
    js = eda["class_overlap"]["jensen_shannon_divergence_bits"]
    rows, cols = ORDER[1:], ORDER[:-1]       # sadness..surprise x joy..love
    m = np.full((len(rows), len(cols)), np.nan)
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            if j <= i:                        # keep the lower triangle
                m[i, j] = js.get(f"{a}|{b}", js.get(f"{b}|{a}"))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", viz.SEQ_BLUE[1:11])
    cmap.set_bad(viz.SURFACE)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    im = ax.imshow(m, cmap=cmap, vmin=0.11, vmax=0.19)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0, labelsize=8.5, colors=viz.INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=viz.SURFACE, linewidth=2.0)
    ax.tick_params(which="minor", length=0)

    # Direct labels inside the cells: no leader lines, so nothing can collide.
    notes = {(0, 0): "closest", (4, 4): "widest gap"}
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isnan(m[i, j]):
                continue
            ink = "#ffffff" if m[i, j] > 0.163 else viz.INK
            note = notes.get((i, j))
            ax.text(j, i - (0.13 if note else 0), f"{m[i, j]:.3f}", ha="center",
                    va="center", fontsize=8.5, color=ink)
            if note:
                ax.text(j, i + 0.19, note, ha="center", va="center", fontsize=7,
                        color=ink, style="italic")

    cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
    cb.set_label("Jensen-Shannon divergence (bits)", fontsize=8, color=viz.INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, color=viz.MUTED, labelcolor=viz.MUTED)

    viz.titles(ax, "All six classes are lexically close to one another",
               "0 = identical word distributions, 1 bit = no shared vocabulary. "
               "The shared\n\"i feel ...\" frame holds every pair inside a narrow "
               "0.118-0.189 band.")
    viz.caption(fig, "Split: train. Lower triangle only; the matrix is symmetric. "
                     "Source: reports/metrics/phase03_text_eda.json")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06_class_overlap.png")
    plt.close(fig)


def main() -> None:
    set_all_seeds()
    ensure_dirs()
    viz.apply_style()

    train = dataio.load_frozen("train")
    eda = json.loads((METRICS / "phase03_text_eda.json").read_text())
    p02 = json.loads((METRICS / "phase02_data_understanding.json").read_text())
    shares = {sp: p02["labels"][sp]["share"] for sp in ("train", "val", "test")}

    fig01_class_balance(train, shares)
    fig02_length_distribution(train)
    fig03_length_by_class(train)
    fig04_vocabulary_structure(train, eda)
    fig05_distinctive_words(eda)
    fig06_class_overlap(eda)

    manifest = {
        "phase": "04_visualization",
        "splits_used": ["train"],
        "split_policy": "train only. Figure 1b reuses the class SHARES already "
                        "reported in Phase 2; no val or test text was read.",
        "palette_validation": {
            "tool": "dataviz skill scripts/validate_palette.js",
            "command": 'validate_palette.js "#2a78d6,#eb6834,#1baf7a" '
                       "--mode light --pairs all",
            "lightness_band": "PASS", "chroma_floor": "PASS",
            "cvd_separation": "PASS worst all-pairs dE 9.2 (deutan), tritan 9.6",
            "normal_vision_floor": "PASS worst all-pairs dE 24.0",
            "contrast_vs_surface": "WARN aqua #1baf7a at 2.74:1 -> relief rule "
                                   "applied: every aqua mark carries a direct value label",
            "categorical_slots_used": 3,
            "note": "Adding a fourth categorical slot requires re-running the validator.",
        },
        "figures": [
            {"file": "fig01_class_balance.png",
             "idea": "the training set is imbalanced 9.4:1, and the three splits "
                     "nonetheless carry the same class mix",
             "form": "horizontal bars (a) + grouped bars (b)", "series": 3},
            {"file": "fig02_length_distribution.png",
             "idea": "documents are short with a thin right tail; 128 tokens covers "
                     "the corpus",
             "form": "histogram with percentile reference lines", "series": 1},
            {"file": "fig03_length_by_class.png",
             "idea": "document length carries no class signal",
             "form": "small multiples against a shared reference", "series": 1},
            {"file": "fig04_vocabulary_structure.png",
             "idea": "Zipfian vocabulary with a very heavy head and a 51% hapax tail",
             "form": "log-log rank-frequency (a) + cumulative coverage (b)", "series": 1},
            {"file": "fig05_distinctive_words.png",
             "idea": "classes are separated by emotion adjectives, not by topic",
             "form": "small-multiple horizontal bars, log-odds z", "series": 1},
            {"file": "fig06_class_overlap.png",
             "idea": "all six classes sit in a narrow lexical band because they "
                     "share the 'i feel ...' frame",
             "form": "lower-triangle heatmap, sequential blue ramp", "series": 1},
        ],
    }
    save_metrics("phase04_visualization", manifest, device="cpu")
    for p in sorted(FIGURES.glob("fig*.png")):
        print(f"  {p.name}  {p.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
