"""Phase 10 — Association rule mining (Apriori) over word co-occurrence.

SPLIT USED: train_clean only (15,923 rows). Rules mined from held-out data would
be leakage; these rules inform the Phase 12 feature decision.

Each document is a TRANSACTION and its distinct words are the ITEMS. Two
questions are asked, and they are different:

  A. Within each emotion, which words co-occur more than chance?
     Classic market-basket analysis, per class.
  B. Do word COMBINATIONS predict an emotion better than their best single word?
     Rules of the form {w1, w2} -> emotion, compared against {w1} -> emotion and
     {w2} -> emotion. This is the compositional question the whole study turns
     on, asked in its cheapest possible form: if pairs never beat singletons,
     then bag-of-words is a sufficient statistic and no amount of sequence
     modelling can help.

Apriori is implemented here rather than imported. The corpus is small, and the
support counting uses an inverted index (item -> set of document ids), so the
support of an itemset is the size of an intersection.
"""

import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import viz
from config import DATA_PROCESSED, FIGURES, LABELS, SEED, ensure_dirs, set_all_seeds
from hardware import save_metrics

MIN_SUPPORT = 0.01        # per-class pass: an itemset in >= 1% of that class
CORPUS_MIN_SUPPORT = 0.003   # corpus-wide pass: ~48 docs, low enough to admit the
                             # emotion adjectives (amazed: 67 docs, Phase 3)
MAX_LEN = 3
MIN_RULE_SUPPORT = 30     # absolute document count, so rare rules cannot be noise

# Phase 8 found HTML/URL residue surviving as word tokens in 268 rows. Those
# tokens co-occur almost perfectly with each other, so they dominate any
# lift ranking. They are reported once, as a confirmation of that finding, and
# then excluded so the phase can say something about emotion.
HTML_TOKENS = {"http", "href", "www", "amp", "rel", "target", "blog", "src",
               "img", "bookmark", "a", "s", "com"}


# ------------------------------------------------------------- Apriori

def build_index(docs: list) -> dict:
    """Inverted index: item -> set of transaction ids."""
    index = defaultdict(set)
    for tid, items in enumerate(docs):
        for it in items:
            index[it].add(tid)
    return index


def apriori(docs: list, min_support: float, max_len: int) -> tuple:
    """Frequent itemsets by level-wise search with downward-closure pruning.

    Downward closure: every subset of a frequent itemset is frequent. So a
    k-candidate can be discarded without counting it if any of its (k-1)-subsets
    was infrequent. That pruning is what makes Apriori tractable.
    """
    n = len(docs)
    min_count = max(1, int(np.ceil(min_support * n)))
    index = build_index(docs)

    t0 = time.perf_counter()
    levels, counted, pruned = [], 0, 0

    l1 = {frozenset([it]): tids for it, tids in index.items()
          if len(tids) >= min_count}
    counted += len(index)
    levels.append(l1)

    k = 2
    while k <= max_len and levels[-1]:
        prev = levels[-1]
        prev_sets = set(prev)
        items = sorted({it for s in prev for it in s})
        candidates = {}
        for a, b in itertools.combinations(prev, 2):
            cand = a | b
            if len(cand) != k:
                continue
            if cand in candidates:
                continue
            # downward closure: check every (k-1)-subset before counting
            if any(frozenset(sub) not in prev_sets
                   for sub in itertools.combinations(cand, k - 1)):
                pruned += 1
                continue
            tids = prev[a] & prev[b]
            counted += 1
            if len(tids) >= min_count:
                candidates[cand] = tids
        levels.append(candidates)
        k += 1

    secs = time.perf_counter() - t0
    stats = {"n_transactions": n, "min_support": min_support,
             "min_count": min_count, "seconds": round(secs, 2),
             "itemsets_counted": counted, "candidates_pruned_by_closure": pruned,
             "frequent_by_size": {str(i + 1): len(lv) for i, lv in enumerate(levels)}}
    return levels, stats


def within_class_rules(levels: list, n: int, top_k: int = 8,
                       exclude: set = frozenset()) -> list:
    """Classic rules X -> Y inside one class, ranked by lift.

    lift = P(X and Y) / (P(X) P(Y)). Lift 1 means independence; above 1 means the
    words attract each other. Confidence alone is not enough: a rule predicting a
    very common word gets high confidence for free.
    """
    support = {s: len(t) for lv in levels for s, t in lv.items()}
    rules = []
    for size in (2, 3):
        if len(levels) <= size - 1:
            continue
        for itemset, tids in levels[size - 1].items():
            if itemset & exclude:
                continue
            sup_xy = len(tids) / n
            for r in range(1, size):
                for ante in itertools.combinations(itemset, r):
                    ante = frozenset(ante)
                    cons = itemset - ante
                    if ante not in support or cons not in support:
                        continue
                    conf = sup_xy / (support[ante] / n)
                    lift = conf / (support[cons] / n)
                    rules.append({
                        "antecedent": sorted(ante), "consequent": sorted(cons),
                        "support_docs": len(tids), "support": round(sup_xy, 5),
                        "confidence": round(conf, 4), "lift": round(lift, 2)})
    rules.sort(key=lambda r: -r["lift"])
    return rules[:top_k]


# ------------------------------------------ B. class-association rules

def class_rules(levels: list, labels: np.ndarray, n: int,
                priors: dict) -> pd.DataFrame:
    """Rules {words} -> emotion, over the whole training set.

    confidence = P(emotion | itemset); lift = confidence / P(emotion).

    A rule is emitted for EVERY emotion, not only the itemset's majority label.
    Taking the majority would silence the small classes entirely: with surprise
    at a 3.5% prior, no frequent itemset ever has surprise as its plurality
    label, yet an itemset that raises surprise from 3.5% to 15% is a lift of 4.2
    and is exactly the kind of rule this phase exists to find.
    """
    rows = []
    for size, level in enumerate(levels, start=1):
        for itemset, tids in level.items():
            if len(tids) < MIN_RULE_SUPPORT:
                continue
            idx = np.fromiter(tids, dtype=np.int64, count=len(tids))
            counts = Counter(labels[idx])
            key = " + ".join(sorted(itemset))
            for lab in LABELS:
                c = counts.get(lab, 0)
                if c == 0:
                    continue
                conf = c / len(tids)
                rows.append({"size": size, "itemset": key,
                             "items": sorted(itemset), "support_docs": len(tids),
                             "emotion": lab, "matching_docs": int(c),
                             "confidence": round(conf, 4),
                             "lift": round(conf / priors[lab], 3)})
    return pd.DataFrame(rows)


def pairs_vs_singletons(df: pd.DataFrame) -> dict:
    """Does a two-word rule beat the better of its two one-word parents?

    This is the study's compositional question in miniature. If combining words
    never raises confidence, a linear bag-of-words model already has all the
    available signal, and sequence modelling has nothing to add.

    Each itemset is reduced to its single strongest rule (highest confidence)
    before the comparison, so a pair is judged on its best prediction against
    each parent's best prediction.
    """
    best_rule = (df.sort_values("confidence", ascending=False)
                   .drop_duplicates("itemset").set_index("itemset"))
    single = {r["items"][0]: r for _, r in
              best_rule[best_rule["size"] == 1].reset_index().iterrows()}
    rows = []
    for _, r in best_rule[best_rule["size"] == 2].reset_index().iterrows():
        a, b = r["items"]
        if a not in single or b not in single:
            continue
        best = max(single[a], single[b], key=lambda s: s["confidence"])
        rows.append({
            "pair": r["itemset"], "pair_emotion": r["emotion"],
            "pair_confidence": r["confidence"], "pair_support": r["support_docs"],
            "best_parent": best["itemset"], "parent_emotion": best["emotion"],
            "parent_confidence": best["confidence"],
            "gain": round(r["confidence"] - best["confidence"], 4),
            "changed_prediction": bool(r["emotion"] != best["emotion"]),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return {"n_pairs_compared": 0}
    return {
        "n_pairs_compared": int(len(out)),
        "mean_gain": round(float(out["gain"].mean()), 4),
        "median_gain": round(float(out["gain"].median()), 4),
        "share_pairs_better_than_best_parent": round(float((out["gain"] > 0).mean()), 4),
        "share_pairs_gaining_over_10_points": round(float((out["gain"] > 0.10).mean()), 4),
        "share_changing_the_prediction": round(float(out["changed_prediction"].mean()), 4),
        "top_gains": out.nlargest(10, "gain").to_dict("records"),
        "frame": out,
    }


def main() -> None:
    set_all_seeds()
    ensure_dirs()
    viz.apply_style()

    train = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
    labels = train["label"].to_numpy()
    docs = [sorted(set(t.split())) for t in train["text"]]
    n = len(docs)
    priors = {lab: float((labels == lab).mean()) for lab in LABELS}

    # ------------------------------------------------- A. per-emotion baskets
    print("=== A. FREQUENT ITEMSETS AND RULES WITHIN EACH EMOTION ===")
    per_class = {}
    for lab in LABELS:
        idx = np.flatnonzero(labels == lab)
        sub = [docs[i] for i in idx]
        levels, stats = apriori(sub, MIN_SUPPORT, MAX_LEN)
        raw = within_class_rules(levels, len(sub), top_k=3)
        rules = within_class_rules(levels, len(sub), top_k=6, exclude=HTML_TOKENS)
        per_class[lab] = {"apriori": stats,
                          "top_rules_including_html_residue": raw,
                          "top_rules_by_lift": rules}
        print(f"  {lab:9s} n={len(sub):>5}  frequent by size "
              f"{stats['frequent_by_size']}  ({stats['seconds']}s)")
        for r in rules[:3]:
            print(f"      {'+'.join(r['antecedent']):>28s} -> "
                  f"{'+'.join(r['consequent']):<16s} "
                  f"lift {r['lift']:>6.1f}  conf {r['confidence']:.2f}  "
                  f"n={r['support_docs']}")

    # -------------------------------------------- B. class-association rules
    print("\n=== B. RULES OF THE FORM {words} -> emotion (whole training set) ===")
    levels_all, stats_all = apriori(docs, CORPUS_MIN_SUPPORT, MAX_LEN)
    levels_all = [{k: v for k, v in lv.items() if not (k & HTML_TOKENS)}
                  for lv in levels_all]
    print(f"  corpus-wide Apriori: frequent by size {stats_all['frequent_by_size']}, "
          f"{stats_all['candidates_pruned_by_closure']:,} candidates pruned by "
          f"downward closure, {stats_all['seconds']}s")

    rules_df = class_rules(levels_all, labels, n, priors)
    best_per_emotion = {}
    for lab in LABELS:
        sub = rules_df[rules_df["emotion"] == lab].nlargest(6, "lift")
        best_per_emotion[lab] = sub.drop(columns=["items"]).to_dict("records")
        print(f"  {lab}:")
        for r in best_per_emotion[lab][:3]:
            print(f"      {r['itemset']:<34s} conf {r['confidence']:.2f}  "
                  f"lift {r['lift']:>5.2f}  n={r['support_docs']}")

    comp = pairs_vs_singletons(rules_df)
    frame = comp.pop("frame")
    print("\n=== DO WORD PAIRS BEAT THEIR BEST SINGLE WORD? ===")
    print(f"  pairs compared           : {comp['n_pairs_compared']:,}")
    print(f"  mean confidence gain     : {comp['mean_gain']:+.4f}")
    print(f"  pairs better than parent : {comp['share_pairs_better_than_best_parent']:.1%}")
    print(f"  gaining > 10 points      : {comp['share_pairs_gaining_over_10_points']:.1%}")
    print(f"  changing the prediction  : {comp['share_changing_the_prediction']:.1%}")

    # ------------------------------------------------------------- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    ax1.scatter(frame["parent_confidence"] * 100, frame["pair_confidence"] * 100,
                s=7, alpha=0.28, color=viz.PRIMARY, edgecolors="none", zorder=3)
    lim = [0, 100]
    ax1.plot(lim, lim, color=viz.INK_2, lw=1.1, ls=(0, (4, 3)), zorder=4)
    ax1.text(60, 14, "points on the line\ngained nothing from\ncombining", fontsize=7.5, color=viz.INK_2)
    top = frame.nlargest(2, "gain")
    for _, r in top.iterrows():
        ax1.annotate(f" {r['pair']}  (+{r['gain']*100:.0f} pts)",
                     (r["parent_confidence"] * 100, r["pair_confidence"] * 100),
                     fontsize=6.5, color=viz.INK_2, va="center")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("confidence of the better single word (%)")
    ax1.set_ylabel("confidence of the word pair (%)")
    viz.value_grid(ax1, "y")
    viz.titles(ax1, "a.  Word pairs add little over single words",
               f"{comp['n_pairs_compared']:,} pairs; median gain "
               f"{comp['median_gain']*100:+.1f} points. Points on the line gained "
               "nothing.")

    # Two rules per emotion, not the global top 12. Lift is confidence divided by
    # the class prior, so the rarest class wins a global ranking mechanically:
    # an unfiltered top-12 is entirely `surprise` and says more about its 3.5%
    # prior than about the rules.
    best = (rules_df.sort_values("lift", ascending=False)
                    .drop_duplicates("itemset")
                    .groupby("emotion", sort=False).head(2))
    best = best.set_index("emotion").loc[LABELS].reset_index().iloc[::-1]
    y = np.arange(len(best))
    bars = ax2.barh(y, best["lift"], height=0.6, color=viz.PRIMARY, zorder=3)
    ax2.set_yticks(y, [f"{r.itemset}  → {r.emotion}" for r in best.itertuples()],
                   fontsize=7)
    ax2.axvline(1.0, color=viz.INK_2, lw=1.1, ls=(0, (4, 3)), zorder=4)
    ax2.text(1.15, -0.9, "lift = 1: no better than the class prior", fontsize=7,
             color=viz.INK_2)
    ax2.set_xlim(0, float(best["lift"].max()) * 1.18)
    viz.value_grid(ax2, "x")
    viz.rounded_bars(ax2, bars, "h")
    for yi, r in zip(y, best.itertuples()):
        ax2.text(r.lift + 0.15, yi, f"{r.lift:.1f}x  (n={r.support_docs})",
                 va="center", fontsize=6.5, color=viz.INK_2)
    ax2.set_xlabel("lift = P(emotion | words) / P(emotion)")
    ax2.tick_params(axis="y", length=0)
    viz.titles(ax2, "b.  One adjective is enough to fix the label",
               f"two best rules per emotion, support >= {MIN_RULE_SUPPORT} docs. "
               "Lift favours rare\nclasses by construction, so rules are ranked "
               "within an emotion, not across.")

    viz.caption(fig, "Split: train_clean (15,923 documents). Each document is a "
                     "transaction; its distinct words are the items.")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig11_apriori.png")
    plt.close(fig)

    payload = {
        "phase": "10_apriori",
        "splits_used": ["train_clean"],
        "split_policy": "train_clean only. Rules mined from held-out data would be "
                        "leakage; these rules inform the Phase 12 feature decision.",
        "encoding": "one transaction per document; items are the distinct words. "
                    "Word order and repetition are discarded by construction.",
        "parameters": {"per_class_min_support": MIN_SUPPORT,
                       "corpus_min_support": CORPUS_MIN_SUPPORT,
                       "max_len": MAX_LEN,
                       "min_rule_support_docs": MIN_RULE_SUPPORT,
                       "excluded_tokens": sorted(HTML_TOKENS),
                       "exclusion_reason": "Phase 8 HTML/URL residue; these tokens "
                                           "co-occur near-perfectly and dominate any "
                                           "lift ranking"},
        "implementation": "Apriori written for this study; support counted by "
                          "intersecting an inverted index, with downward-closure "
                          "pruning of candidates.",
        "per_emotion": per_class,
        "corpus_wide_apriori": stats_all,
        "class_association_rules": {
            "n_rules": int(len(rules_df)),
            "by_itemset_size": {str(k): int(v) for k, v in
                                rules_df["size"].value_counts().sort_index().items()},
            "top_by_lift_overall": rules_df.nlargest(15, "lift").drop(
                columns=["items"]).to_dict("records"),
            "lift_caveat": ("lift = confidence / prior, so the rarest class wins any "
                            "global lift ranking mechanically. The unfiltered top 15 "
                            "is entirely `surprise` (prior 3.5%). Compare rules "
                            "within an emotion, not across."),
            "top_per_emotion": best_per_emotion,
        },
        "pairs_vs_singletons": comp,
        "figures": ["fig11_apriori.png"],
    }
    path = save_metrics("phase10_apriori", payload, device="cpu")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
