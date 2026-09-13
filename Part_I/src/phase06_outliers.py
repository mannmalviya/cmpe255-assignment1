"""Phase 6 — Outlier analysis.

SPLIT USED: train_clean only (15,923 rows, from Phase 5). Deciding what counts
as off-distribution is a modelling judgement, so it is made on training data.

Four kinds of outlier are examined, each for a different reason:
  1. length outliers  - is a 2-word document even answerable?
  2. off-template rows - the 2.7% that break the "i feel ..." frame
  3. vocabulary-rarity outliers - off-topic content (e.g. the Phase 3 "shinobi" row)
  4. cue-free rows - documents containing no emotion word at all

Nothing is deleted. Outliers here are informative, not corrupt: they are where
models will fail, so Phase 17 needs them flagged, not removed.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as dataio
import viz
from config import DATA_PROCESSED, FIGURES, LABELS, ensure_dirs, set_all_seeds
from hardware import save_metrics

FEEL = r"\b(?:feel|feeling|feels|felt)\b"
CUE_Z = 3.0          # log-odds z above which a word counts as an emotion cue
N_CUES_PER_CLASS = 150


# ------------------------------------------------------ the cue lexicon

def build_cue_lexicon(df: pd.DataFrame) -> tuple:
    """Words a class over-uses strongly enough to act as an emotion cue.

    Reuses the Phase 3 estimator (log-odds with an informative Dirichlet prior)
    rather than a hand-written emotion word list, so the lexicon is derived from
    this corpus instead of imported from outside it.
    """
    per_class = {lab: Counter(t for text in df[df["label"] == lab]["text"]
                              for t in text.split()) for lab in LABELS}
    corpus = Counter()
    for c in per_class.values():
        corpus.update(c)
    a0 = sum(corpus.values())

    cues = {}
    for lab in LABELS:
        rest = Counter(corpus)
        rest.subtract(per_class[lab])
        n_i = sum(per_class[lab].values())
        n_j = sum(v for v in rest.values() if v > 0)
        scored = []
        for w, aw in corpus.items():
            yi, yj = per_class[lab].get(w, 0), max(rest.get(w, 0), 0)
            num_i, num_j = yi + aw, yj + aw
            den_i, den_j = n_i + a0 - num_i, n_j + a0 - num_j
            if den_i <= 0 or den_j <= 0:
                continue
            delta = math.log(num_i / den_i) - math.log(num_j / den_j)
            z = delta / math.sqrt(1.0 / num_i + 1.0 / num_j)
            if z >= CUE_Z:
                scored.append((w, z))
        scored.sort(key=lambda r: -r[1])
        cues[lab] = [w for w, _ in scored[:N_CUES_PER_CLASS]]

    lexicon = set()
    for v in cues.values():
        lexicon.update(v)
    return lexicon, cues


# ------------------------------------------------------ outlier detectors

def tukey_fences(values: np.ndarray) -> dict:
    """Standard 1.5 x IQR fences. Reported for reference, not used as a cutoff."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return {"q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "lower_fence": float(q1 - 1.5 * iqr), "upper_fence": float(q3 + 1.5 * iqr)}


def class_profile(df: pd.DataFrame, mask: pd.Series, base: dict) -> dict:
    """Class mix of a subgroup against the corpus mix. Is the group concentrated?"""
    sub = df[mask]
    if not len(sub):
        return {"n": 0}
    counts = sub["label"].value_counts()
    out = {"n": int(len(sub)), "share_of_train": round(len(sub) / len(df), 5),
           "class_share": {}, "lift_vs_corpus": {}}
    for lab in LABELS:
        s = float(counts.get(lab, 0)) / len(sub)
        out["class_share"][lab] = round(s, 4)
        out["lift_vs_corpus"][lab] = round(s / base[lab], 2) if base[lab] else None
    return out


def main() -> None:
    set_all_seeds()
    ensure_dirs()
    viz.apply_style()

    df = dataio.load_frozen("train_clean") if (DATA_PROCESSED / "train_clean.parquet"
                                               ).exists() else None
    df = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
    words = df["text"].str.split()
    n_words = words.str.len()
    df = df.assign(n_words=n_words)

    base = {lab: float((df["label"] == lab).mean()) for lab in LABELS}

    # --- 1. length ------------------------------------------------------
    fences = tukey_fences(n_words.to_numpy())
    p1, p99 = np.percentile(n_words, [1, 99])
    short_mask = n_words <= 5           # p5 from Phase 3
    long_mask = n_words >= fences["upper_fence"]

    # --- 2. off-template -------------------------------------------------
    has_feel = df["text"].str.contains(FEEL, regex=True)
    off_template = ~has_feel

    # --- 3. vocabulary rarity --------------------------------------------
    corpus_counts = Counter(t for ws in words for t in ws)
    rare = {w for w, c in corpus_counts.items() if c <= 2}
    rare_share = words.map(lambda ws: sum(1 for t in ws if t in rare) / len(ws))
    df = df.assign(rare_share=rare_share)
    rare_cut = float(np.percentile(rare_share, 99))
    rare_mask = rare_share >= rare_cut

    # --- 4. cue-free ------------------------------------------------------
    lexicon, cues_per_class = build_cue_lexicon(df)
    n_cues = words.map(lambda ws: sum(1 for t in ws if t in lexicon))
    df = df.assign(n_cues=n_cues)
    cue_free = n_cues == 0

    # --- cue coverage by length bucket -----------------------------------
    buckets = [(2, 5), (6, 10), (11, 17), (18, 25), (26, 40), (41, 66)]
    coverage = []
    for lo, hi in buckets:
        m = (df["n_words"] >= lo) & (df["n_words"] <= hi)
        coverage.append({
            "bucket": f"{lo}-{hi}",
            "n": int(m.sum()),
            "share_with_a_cue": round(float((df.loc[m, "n_cues"] > 0).mean()), 4),
            "mean_cues": round(float(df.loc[m, "n_cues"].mean()), 2),
        })

    # --- figure ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(coverage))
    vals = [c["share_with_a_cue"] * 100 for c in coverage]
    bars = ax1.bar(x, vals, width=0.6, color=viz.PRIMARY, zorder=3)
    ax1.set_xticks(x, [c["bucket"] for c in coverage])
    ax1.set_ylim(0, 108)
    viz.value_grid(ax1, "y")
    viz.rounded_bars(ax1, bars, "v")
    for xi, v, c in zip(x, vals, coverage):
        ax1.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5,
                 color=viz.INK_2)
        ax1.text(xi, 3, f"n={c['n']:,}", ha="center", va="bottom", fontsize=6.5,
                 color=viz.SURFACE, zorder=5)
    ax1.set_xlabel("document length (words)")
    ax1.set_ylabel("% of documents containing an emotion cue")
    viz.titles(ax1, "a.  Even 2-word documents almost always carry an emotion cue",
               f"cue = one of {len(lexicon):,} words a class over-uses at log-odds "
               "z >= 3. Short is not the same as uninformative.")

    ax2.scatter(df["n_words"], df["rare_share"] * 100, s=3, alpha=0.18,
                color=viz.PRIMARY, edgecolors="none", zorder=3)
    sel = df[rare_mask]
    ax2.scatter(sel["n_words"], sel["rare_share"] * 100, s=6, alpha=0.7,
                color=viz.ACCENT, edgecolors="none", zorder=4)
    ax2.axhline(rare_cut * 100, color=viz.INK_2, lw=1.0, ls=(0, (4, 3)), zorder=5)
    ax2.text(66, rare_cut * 100 + 2, f"p99 = {rare_cut:.0%}", ha="right", fontsize=7.5,
             color=viz.INK_2)
    ax2.set_xlabel("document length (words)")
    ax2.set_ylabel("% of tokens that are rare in train (count <= 2)")
    ax2.set_xlim(0, 68)
    viz.value_grid(ax2, "y")
    ax2.legend([viz.swatch(viz.ACCENT)], [f"flagged off-distribution (n={len(sel)})"],
               loc="upper right", handlelength=1.0, handletextpad=0.5)
    viz.titles(ax2, "b.  Rare-word density falls with length",
               "short documents dominate the off-distribution flag by construction")

    viz.caption(fig, "Split: train_clean (15,923 rows). No row is removed; "
                     "flags are written to data/processed/train_outlier_flags.parquet")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig07_outliers.png")
    plt.close(fig)

    # --- flags file --------------------------------------------------------
    flags = pd.DataFrame({
        "row_id": df["row_id"], "label": df["label"], "n_words": df["n_words"],
        "n_cues": df["n_cues"], "rare_share": df["rare_share"].round(4),
        "is_short": short_mask.values, "is_long": long_mask.values,
        "is_off_template": off_template.values, "is_rare_heavy": rare_mask.values,
        "is_cue_free": cue_free.values,
    })
    flags["n_flags"] = flags[["is_short", "is_long", "is_off_template",
                              "is_rare_heavy", "is_cue_free"]].sum(axis=1)
    flags.to_parquet(DATA_PROCESSED / "train_outlier_flags.parquet", index=False)

    def ex(mask, k=5, cols=("row_id", "label", "text")):
        return df[mask].head(k)[list(cols)].to_dict("records")

    # --- 5. corpus composition: is there more than one text genre? ---------
    # The off-template rows are not merely atypical; inspection shows they are a
    # different KIND of text. Rows beginning "when ..." describe a situation
    # rather than report a feeling, which is the answer format of a
    # "describe a time you felt X" survey, not of a first-person status post.
    first_person = df["text"].str.match(r"^(i|im|ive|id|ill)\b")
    genre_a = has_feel & first_person                    # "i feel <adjective>"
    genre_b = ~genre_a
    when_rows = df["text"].str.match(r"^when\b")
    genre = {
        "genre_a_first_person_feel": {
            **class_profile(df, genre_a, base),
            "median_words": float(df.loc[genre_a, "n_words"].median()),
            "example": df.loc[genre_a, "text"].iloc[0],
        },
        "genre_b_other": {
            **class_profile(df, genre_b, base),
            "median_words": float(df.loc[genre_b, "n_words"].median()),
            "examples": df.loc[genre_b & ~first_person, "text"].head(6).tolist(),
        },
        "rows_starting_with_when": int(when_rows.sum()),
        "of_those_off_template": int((when_rows & off_template).sum()),
        "evidence": [
            "433 rows (2.72%) contain no feel-word at all.",
            "100 of them open with a scene-setting preposition or 'when'.",
            "68 rows in the corpus start with 'when'; 66 are off-template.",
            "Median length of off-template non-first-person rows is 12.5 words "
            "against 17.0 corpus-wide.",
            "Several show non-native English ('suffered of nightmares even since "
            "than', 'kept me in leadingstrings') and terse noun-phrase answers "
            "('earth crake', 'in sweden', 'during lectures').",
        ],
        "interpretation": (
            "The corpus is a MIXTURE of at least two text genres. The dominant one "
            "is first-person present-tense self-report ('i feel <adjective>'). A "
            "small second genre describes the SITUATION that caused an emotion, "
            "names no feeling, and reads like free-text answers to a "
            "'describe a time you felt X' survey translated by non-native speakers. "
            "This is a stylistic inference from the text itself, not documented "
            "provenance; we did not verify it against the dataset's sources."),
        "consequence": (
            "Genre B rows must be labelled from world knowledge, not from a cue "
            "word: nothing in 'when my father passed away' names sadness. This is "
            "the sub-population where a pre-trained language model has a real, "
            "structural advantage over TF-IDF, and it is 2.7% of the corpus. "
            "Phase 17 must report accuracy separately for the two genres; an "
            "aggregate score will hide the only place the LLM can win."),
    }

    payload = {
        "phase": "06_outliers",
        "splits_used": ["train_clean"],
        "split_policy": "train_clean only (15,923 rows). Outlier thresholds are a "
                        "modelling judgement and are set on training data.",
        "policy": "FLAG, DO NOT REMOVE. These rows are where models will fail, so "
                  "Phase 17 needs them identified. Removing them would raise every "
                  "score without any model improving.",
        "length": {
            "tukey_fences_words": fences,
            "p1": float(p1), "p99": float(p99),
            "short_definition": "n_words <= 5 (the p5 of Phase 3)",
            "short": {**class_profile(df, short_mask, base),
                      "examples": ex(short_mask)},
            "long_definition": f"n_words >= upper Tukey fence ({fences['upper_fence']:.0f})",
            "long": {**class_profile(df, long_mask, base),
                     "max_words": int(df["n_words"].max())},
        },
        "off_template": {
            "definition": "text contains no feel/feeling/feels/felt",
            **class_profile(df, off_template, base),
            "examples": ex(off_template),
        },
        "vocabulary_rarity": {
            "definition": f"share of tokens with corpus count <= 2, at or above the "
                          f"p99 of {rare_cut:.4f}",
            **class_profile(df, rare_mask, base),
            "most_extreme": df.nlargest(8, "rare_share")[
                ["row_id", "label", "n_words", "rare_share", "text"]].to_dict("records"),
        },
        "cue_lexicon": {
            "size": len(lexicon),
            "threshold_z": CUE_Z,
            "cap_per_class": N_CUES_PER_CLASS,
            "per_class_size": {lab: len(v) for lab, v in cues_per_class.items()},
        },
        "cue_free": {
            "definition": "document contains no word from the cue lexicon",
            **class_profile(df, cue_free, base),
            "examples": ex(cue_free),
        },
        "cue_coverage_by_length": coverage,
        "corpus_composition": genre,
        "flag_overlap": {
            "rows_with_no_flag": int((flags["n_flags"] == 0).sum()),
            "rows_with_1_flag": int((flags["n_flags"] == 1).sum()),
            "rows_with_2plus_flags": int((flags["n_flags"] >= 2).sum()),
            "rows_with_3plus_flags": int((flags["n_flags"] >= 3).sum()),
        },
        "artefacts": {
            "flags": "data/processed/train_outlier_flags.parquet",
            "figure": "reports/figures/fig07_outliers.png",
        },
    }
    path = save_metrics("phase06_outliers", payload, device="cpu")

    print("=== LENGTH ===")
    print(f"  Tukey fences (words): lower {fences['lower_fence']:.1f}, "
          f"upper {fences['upper_fence']:.1f}   p1 {p1:.0f}  p99 {p99:.0f}")
    print(f"  short (<=5 words): n={int(short_mask.sum())}  "
          f"long (>={fences['upper_fence']:.0f}): n={int(long_mask.sum())}")
    print("\n=== OFF-TEMPLATE (no feel-word) ===")
    print(f"  n={int(off_template.sum())} ({off_template.mean():.2%})  "
          f"class lift: {payload['off_template']['lift_vs_corpus']}")
    print("\n=== CUE COVERAGE BY LENGTH ===")
    for c in coverage:
        print(f"  {c['bucket']:>6} words  n={c['n']:>5}  "
              f"{c['share_with_a_cue']:>7.1%} have a cue   mean cues {c['mean_cues']}")
    print("\n=== CUE-FREE ===")
    print(f"  n={int(cue_free.sum())} ({cue_free.mean():.2%})  "
          f"class lift: {payload['cue_free']['lift_vs_corpus']}")
    print("\n=== MOST OFF-DISTRIBUTION ROWS ===")
    for r in payload["vocabulary_rarity"]["most_extreme"][:5]:
        print(f"  {r['row_id']:>14} [{r['label']:>8}] rare={r['rare_share']:.2f} "
              f"{r['text'][:70]}")
    print("\n=== CORPUS COMPOSITION ===")
    ga, gb = genre["genre_a_first_person_feel"], genre["genre_b_other"]
    print(f"  genre A (i feel ...): n={ga['n']:>6} ({ga['share_of_train']:.2%})  "
          f"median {ga['median_words']:.0f} words")
    print(f"  genre B (other)     : n={gb['n']:>6} ({gb['share_of_train']:.2%})  "
          f"median {gb['median_words']:.0f} words")
    print(f"  genre B class lift  : {gb['lift_vs_corpus']}")
    for e in gb["examples"][:4]:
        print(f"      {e[:78]}")

    print("\n=== FLAG OVERLAP ===")
    print(json.dumps(payload["flag_overlap"], indent=2))
    print(f"\nwrote {path}")
    print("wrote data/processed/train_outlier_flags.parquet")
    print("wrote reports/figures/fig07_outliers.png")


if __name__ == "__main__":
    main()
