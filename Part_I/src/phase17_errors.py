"""Phase 17 — Error analysis: where each model family fails, with real examples.

SPLIT USED: test predictions from data/processed/test_predictions_all.parquet
(produced and score-verified by phase17_dump_predictions.py). NOTHING is fitted,
tuned or selected.

Every slice is defined by a rule FIXED IN AN EARLIER PHASE, computed from the text
(or from train_clean), never invented after looking at the errors:
  genre A/B, short, long, off-template, cue-free .... Phase 6 rules
  HTML residue ..................................... Phase 8 marker rule
  near-duplicate of a same-label training row ...... Phase 9 (cosine >= 0.90)
  contradicted by a differently-labelled copy ...... Phase 5
  contains negation ................................ Phase 10 limitation (order/scope)

Predictions under test, registered earlier:
  P1 (Phase 5/12)  errors concentrate on joy<->love and fear<->surprise
  P2 (Phase 6)     the pre-trained model's advantage is largest on genre B
  P3 (Phase 9)     near-duplicate same-label rows are near-free points
  P4 (Phase 10)    any fine-tuned gain must come from what bag-of-words cannot see
                   (order, negation scope, world knowledge)
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import viz
from config import DATA_PROCESSED, FIGURES, LABELS, SEED, ensure_dirs, set_all_seeds
from hardware import save_metrics
from phase06_outliers import FEEL, build_cue_lexicon

MODELS = ["linear_svm", "logreg", "gradient_boosting", "distilbert",
          "qwen_zero_shot", "qwen_few_shot"]
TRAINED = ["linear_svm", "logreg", "gradient_boosting", "distilbert"]
NEG = re.compile(r"\b(not|no|never|nor|nothing|nobody|none|without|dont|didnt|doesnt|"
                 r"cant|couldnt|wont|wouldnt|isnt|wasnt|arent|werent|havent|hasnt|"
                 r"shouldnt|don t|didn t|doesn t|can t|won t|isn t|wasn t)\b")
HTML = re.compile(r"\b(?:http|href|www|amp)\b")
RNG = np.random.default_rng(SEED)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def examples(df, mask, cols, k=6):
    sub = df[mask]
    if len(sub) > k:
        sub = sub.iloc[np.sort(RNG.choice(len(sub), k, replace=False))]
    return sub[cols].to_dict("records")


def build_slices(test, train):
    words = test["text"].str.split()
    n_words = words.str.len()
    first_person = test["text"].str.match(r"^(i|im|ive|id|ill)\b")
    has_feel = test["text"].str.contains(FEEL, regex=True)

    lexicon, _ = build_cue_lexicon(train)
    train_counts = Counter(t for s in train["text"] for t in s.split())

    # Phase 9: near-duplicate of a same-label train_clean row (cosine >= 0.90)
    feats = DATA_PROCESSED / "features"
    Etr, Ete = np.load(feats / "emb_train_clean.npy"), np.load(feats / "emb_test.npy")
    S = Ete @ Etr.T
    trl, tel = train["label"].to_numpy(), test["label"].to_numpy()
    near_same = np.array([bool(((S[i] >= 0.90) & (trl == tel[i])).any())
                          for i in range(len(test))])

    # Phase 5: text also present elsewhere with a DIFFERENT label, split by where
    # the conflicting copy lives - that decides which models ever saw it.
    val = pd.read_parquet(DATA_PROCESSED / "val.parquet")
    raw_train = pd.read_parquet(DATA_PROCESSED / "train.parquet")

    def conflict_in(df):
        m = df.groupby("text")["label"].agg(set)
        return np.array([t in m.index and bool(m[t] - {l})
                         for t, l in zip(test["text"], test["label"])])

    return {
        "genre A (i feel ...)": (first_person & has_feel).to_numpy(),
        "genre B (situation, no feeling named)": (~(first_person & has_feel)).to_numpy(),
        "short (<= 5 words)": (n_words <= 5).to_numpy(),
        "long (>= 46 words)": (n_words >= 46).to_numpy(),
        "cue-free": words.map(lambda ws: not any(t in lexicon for t in ws)).to_numpy(),
        "contains negation": test["text"].str.contains(NEG, regex=True).to_numpy(),
        "HTML residue": test["text"].str.contains(HTML, regex=True).to_numpy(),
        "near-duplicate of same-label train row": near_same,
        "conflicting copy in raw train (removed from train_clean)": conflict_in(raw_train),
        "conflicting copy in val": conflict_in(val),
    }, lexicon


def slice_table(df, slices):
    rows = []
    for name, mask in slices.items():
        n = int(mask.sum())
        r = {"slice": name, "n": n, "share_of_test": round(n / len(df), 4)}
        for m in MODELS:
            k = int((df.loc[mask, m] == df.loc[mask, "label"]).sum())
            r[m] = {"accuracy": round(k / n, 4) if n else None, "ci95": wilson(k, n)}
        rows.append(r)
    return rows


def main():
    set_all_seeds()
    ensure_dirs()
    viz.apply_style()
    df = pd.read_parquet(DATA_PROCESSED / "test_predictions_all.parquet")
    train = pd.read_parquet(DATA_PROCESSED / "train_clean.parquet")
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    assert (df["row_id"].to_numpy() == test["row_id"].to_numpy()).all()

    slices, _ = build_slices(test, train)
    ok = {m: (df[m] == df["label"]).to_numpy() for m in MODELS}
    overall = {m: round(float(ok[m].mean()), 4) for m in MODELS}
    table = slice_table(df, slices)

    print("=== ACCURACY BY SLICE (test) ===")
    print(f"{'slice':<58}{'n':>5}  " + "  ".join(f"{m[:10]:>10}" for m in MODELS))
    print(f"{'ALL':<58}{len(df):>5}  " + "  ".join(f"{overall[m]:>10.3f}" for m in MODELS))
    for r in table:
        print(f"{r['slice']:<58}{r['n']:>5}  " + "  ".join(
            f"{(r[m]['accuracy'] if r[m]['accuracy'] is not None else float('nan')):>10.3f}"
            for m in MODELS))

    # ---------------------------------------------------- P2: genre B advantage
    gA, gB = slices["genre A (i feel ...)"], slices["genre B (situation, no feeling named)"]
    gap = lambda m, mask: float(ok[m][mask].mean() - ok["linear_svm"][mask].mean())
    p2 = {m: {"acc_gain_over_svm_genre_A": round(gap(m, gA), 4),
              "acc_gain_over_svm_genre_B": round(gap(m, gB), 4)}
          for m in ("distilbert", "qwen_zero_shot", "qwen_few_shot")}
    p2["genre_B_n"] = int(gB.sum())

    # ---------------------------------------------------- P4: fixed vs broken
    fixed = ~ok["linear_svm"] & ok["distilbert"]
    broken = ok["linear_svm"] & ~ok["distilbert"]
    enrich = {}
    for name, mask in slices.items():
        base = mask.mean()
        if base == 0:
            continue
        enrich[name] = {
            "base_rate": round(float(base), 4),
            "rate_in_fixed": round(float(mask[fixed].mean()), 4),
            "rate_in_broken": round(float(mask[broken].mean()), 4),
            "enrichment_in_fixed": round(float(mask[fixed].mean() / base), 2),
            "enrichment_in_broken": round(float(mask[broken].mean() / base), 2)}

    # ---------------------------------------------------- the hard core
    hard = np.logical_and.reduce([~ok[m] for m in TRAINED])
    hard_pairs = Counter(tuple(sorted((a, b))) for a, b in
                         zip(df.loc[hard, "label"], df.loc[hard, "distilbert"]))
    ambiguous = {("joy", "love"), ("fear", "surprise")}
    hard_amb = sum(v for k, v in hard_pairs.items() if k in ambiguous)

    # ---------------------------------------------------- P1: error pairs
    def pair_share(m):
        wrong = ~ok[m] & df[m].isin(LABELS).to_numpy()
        c = Counter(tuple(sorted((a, b))) for a, b in
                    zip(df.loc[wrong, "label"], df.loc[wrong, m]))
        tot = int((~ok[m]).sum())
        return {"total_errors": tot,
                "unparseable": int((df[m] == "UNPARSEABLE").sum()),
                "joy<->love": c[("joy", "love")], "fear<->surprise": c[("fear", "surprise")],
                "sadness<->anger": c[("anger", "sadness")],
                "share_joy_love_plus_fear_surprise_of_all_errors": round(
                    (c[("joy", "love")] + c[("fear", "surprise")]) / tot, 4)}
    p1 = {m: pair_share(m) for m in MODELS}

    # ---------------------------------------------------- complementarity
    qwen_only = ok["qwen_zero_shot"] & ~ok["linear_svm"] & ~ok["distilbert"]

    # ---------------------------------------------------- paired tests per slice
    from scipy.stats import binomtest
    slice_mcnemar = {}
    for name, mask in slices.items():
        s_ok, b_ok = ok["linear_svm"][mask], ok["distilbert"][mask]
        f_, br_ = int((~s_ok & b_ok).sum()), int((s_ok & ~b_ok).sum())
        slice_mcnemar[name] = {"n": int(mask.sum()), "distilbert_fixed": f_,
                               "distilbert_broke": br_,
                               "exact_p": (float(f"{binomtest(f_, f_ + br_, 0.5).pvalue:.3g}")
                                           if f_ + br_ else None)}

    # ---------------------------------------------------- EXPLORATORY
    # Chosen AFTER reading the errors, so reported as exploratory, not as tests of
    # a prediction. Question: are some cue words split between labels by the
    # ANNOTATIONS THEMSELVES (measured on train_clean), and do models fail there?
    probes = {"stressed": r"\bstressed\b", "agitated": r"\bagitated\b",
              "passionate": r"\bpassionate\b", "amazing": r"\bamazing\b",
              "hate/hated": r"\bhated?\b",
              "feel/be/get hated": r"\b(?:feel|feeling|felt|be|being|am|get)\s+(?:\w+\s+)?hated\b",
              "amazed (control: consistent label)": r"\bamazed\b"}
    split_words = {}
    for name, pat in probes.items():
        mtr = train["text"].str.contains(pat, regex=True)
        mte = df["text"].str.contains(pat, regex=True).to_numpy()
        c = Counter(train.loc[mtr, "label"])
        ntr = int(mtr.sum())
        top = c.most_common(2)
        split_words[name] = {
            "train_n": ntr,
            "train_label_share": {k: round(v / ntr, 3) for k, v in c.most_common()},
            "top_label_share": round(top[0][1] / ntr, 3),
            "second_label_share": round(top[1][1] / ntr, 3) if len(top) > 1 else 0.0,
            "test_n": int(mte.sum()),
            "test_accuracy": {m: round(float(ok[m][mte].mean()), 3) if mte.sum() else None
                              for m in MODELS}}

    # ---------------------------------------------------- examples
    cols = ["row_id", "text", "label", "linear_svm", "distilbert", "qwen_zero_shot"]
    ex = {
        "hard_core_all_trained_wrong": examples(df, hard, cols, 10),
        "fixed_by_distilbert": examples(df, fixed, cols, 10),
        "broken_by_distilbert": examples(df, broken, cols, 8),
        "fixed_and_contains_negation": examples(
            df, fixed & slices["contains negation"], cols, 8),
        "genre_B_distilbert_right_svm_wrong": examples(df, fixed & gB, cols, 8),
        "genre_B_all_trained_wrong": examples(df, hard & gB, cols, 6),
        "svm_sadness_anger_confusions": examples(
            df, (~ok["linear_svm"]) & df["label"].isin(["sadness", "anger"]).to_numpy()
            & df["linear_svm"].isin(["sadness", "anger"]).to_numpy(), cols, 8),
        "qwen_zero_shot_joy_called_love": examples(
            df, (df["label"] == "joy").to_numpy() & (df["qwen_zero_shot"] == "love").to_numpy(),
            cols + ["qwen_zero_shot_raw"], 6),
        "qwen_zero_shot_unparseable": examples(
            df, (df["qwen_zero_shot"] == "UNPARSEABLE").to_numpy(),
            ["row_id", "text", "label", "qwen_zero_shot_raw", "distilbert"], 8),
        "qwen_right_both_trained_wrong": examples(df, qwen_only, cols, 8),
        "conflicting_copy_rows": examples(
            df, slices["conflicting copy in raw train (removed from train_clean)"]
            | slices["conflicting copy in val"], cols, 14),
    }

    # ---------------------------------------------------- figure
    show = ["genre A (i feel ...)", "genre B (situation, no feeling named)",
            "short (<= 5 words)", "long (>= 46 words)", "contains negation",
            "cue-free", "HTML residue", "near-duplicate of same-label train row"]
    series = [("linear_svm", "linear SVM", viz.SERIES[0], "o"),
              ("distilbert", "DistilBERT fine-tuned", viz.SERIES[1], "s"),
              ("qwen_zero_shot", "Qwen zero-shot", viz.SERIES[2], "D")]
    trow = {r["slice"]: r for r in table}
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ys = np.arange(len(show))[::-1]
    for j, (key, lab, col, mk) in enumerate(series):
        off = (j - 1) * 0.24
        for y, s in zip(ys, show):
            r = trow[s][key]
            lo, hi = r["ci95"]
            ax.plot([lo * 100, hi * 100], [y + off] * 2, color=col, lw=1.6, zorder=3,
                    solid_capstyle="round")
            ax.scatter(r["accuracy"] * 100, y + off, color=col, marker=mk, s=40,
                       zorder=4, edgecolors=viz.SURFACE, linewidths=1.0,
                       label=lab if y == ys[0] else None)
    for j, (key, lab, col, mk) in enumerate(series):
        ax.axvline(overall[key] * 100, color=col, lw=0.9, ls=(0, (3, 3)), zorder=2)
    ax.set_yticks(ys, [f"{s}   n={trow[s]['n']}" for s in show], fontsize=8)
    ax.set_xlim(0, 104)
    ax.set_xlabel("test accuracy within slice (%), 95% Wilson interval")
    ax.tick_params(axis="y", length=0)
    viz.value_grid(ax, "x")
    ax.legend(loc="lower left", fontsize=7.5)
    viz.titles(ax, "Where each family fails: accuracy by slice",
               "dashed verticals = each model's overall accuracy. Slice rules fixed in "
               "Phases 5, 6, 8, 9; none chosen after seeing errors.")
    viz.caption(fig, "Split: test predictions from Phases 12, 13 and 15, regenerated "
                     "and score-verified. Wide intervals on small slices (genre B, "
                     "HTML, short) are real uncertainty, not noise to ignore.")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig14_error_slices.png")
    plt.close(fig)

    payload = {
        "phase": "17_error_analysis",
        "splits_used": ["test predictions (verified reproductions of Phases 12-15)"],
        "policy": "no fitting, tuning or selection; slice rules fixed in earlier phases",
        "overall_accuracy": overall,
        "slices": table,
        "P1_error_pairs": p1,
        "P2_genre_B": p2,
        "P4_fixed_vs_broken": {"n_fixed": int(fixed.sum()), "n_broken": int(broken.sum()),
                               "slice_enrichment": enrich},
        "hard_core": {"n_all_four_trained_models_wrong": int(hard.sum()),
                      "true_vs_distilbert_pairs": {f"{a}<->{b}": v for (a, b), v in
                                                   hard_pairs.most_common()},
                      "share_in_joy_love_or_fear_surprise": round(hard_amb / hard.sum(), 4),
                      "true_label_counts": dict(Counter(df.loc[hard, "label"]))},
        "slice_mcnemar_svm_vs_distilbert": slice_mcnemar,
        "exploratory_split_label_words": {
            "status": "EXPLORATORY - words chosen after reading errors",
            "words": split_words},
        "complementarity": {"qwen_zero_shot_right_while_svm_and_distilbert_wrong":
                            int(qwen_only.sum())},
        "examples": ex,
        "figures": ["fig14_error_slices.png"],
    }
    save_metrics("phase17_errors", payload, device="cpu")

    print("\n=== P1 error pairs ===")
    for m, v in p1.items():
        print(f"  {m:<18} errors {v['total_errors']:>4}  unparseable {v['unparseable']:>3}  "
              f"joy<->love {v['joy<->love']:>3}  fear<->surprise {v['fear<->surprise']:>3}  "
              f"sad<->anger {v['sadness<->anger']:>3}  "
              f"share(jl+fs) {v['share_joy_love_plus_fear_surprise_of_all_errors']:.3f}")
    print("\n=== P2 genre B ===", json.dumps(p2, indent=2))
    print(f"\n=== P4 fixed {int(fixed.sum())} / broken {int(broken.sum())}: enrichment ===")
    for k, v in enrich.items():
        print(f"  {k:<58} base {v['base_rate']:.3f}  fixed x{v['enrichment_in_fixed']:<5}"
              f"  broken x{v['enrichment_in_broken']}")
    print(f"\n=== hard core: {int(hard.sum())} rows all four trained models miss; "
          f"{hard_amb/hard.sum():.1%} in joy<->love or fear<->surprise ===")
    print("  pairs:", hard_pairs.most_common(6))
    print(f"  qwen zero-shot right where SVM and DistilBERT both wrong: {int(qwen_only.sum())}")


if __name__ == "__main__":
    main()
