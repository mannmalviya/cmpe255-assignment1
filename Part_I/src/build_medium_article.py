"""Build the Medium version of the final report.

Source of truth: reports/final_report.md. Changes for Medium only:
  - every table is replaced by its rendered image (render_tables_for_medium.py),
    because Medium's editor has no tables;
  - the study figures from reports/figures are placed after the paragraph each
    one supports;
  - "Figure N." caption prefixes are dropped, since added figures would break the
    numbering.

Outputs:
  reports/medium_article.md     readable Markdown with image links
  reports/medium_segments.json  ordered html / image blocks for pasting into Medium
"""

import json
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "final_report.md"
FIG = "reports/figures"

TABLE_CAPTIONS = [
    "The frozen split and its class counts.",
    "The machine every timing was measured on.",
    "The eight models, what each was trained and tuned on, and where it ran.",
    "Test-split quality for every model. The decision bar is the SVM's 0.858 + 0.030 = 0.888.",
    "Latency, throughput and cost for models served on CPU.",
    "Latency, throughput and cost for models served on GPU. ✗ marks p95 outside the 50 ms budget.",
    "The pre-registered decision rule applied to every language-model configuration.",
    "How certain the fine-tuned model's win is: paired tests, bootstrap and three training seeds.",
    "What moving from the linear SVM to fine-tuned DistilBERT costs and buys at 1 M messages a month.",
    "Measurement defects found during the study, and how each was resolved.",
]

# anchor sentence -> figures placed after that paragraph, in reading order
FIGURES = {
    "accuracy but only 0.086 macro F1.": [
        ("fig01_class_balance.png",
         "The training set is imbalanced 9.4 : 1, yet train, val and test carry the same class mix."),
    ],
    "cleaning transform was applied.": [
        ("fig02_length_distribution.png",
         "Documents are short: median 17 words, maximum 66, so a 128-token sequence covers the corpus."),
        ("fig03_length_by_class.png",
         "Document length carries no class signal: every class has almost the same length distribution."),
    ],
    "which favours bag-of-words models.": [
        ("fig04_vocabulary_structure.png",
         "The vocabulary is Zipfian: 49 word types cover half of all tokens, and half of all types occur once."),
        ("fig05_distinctive_words.png",
         "The words that best separate each class are emotion adjectives, not topics."),
    ],
    "from style, not from documented provenance.": [
        ("fig07_outliers.png",
         "Even 2-word documents almost always contain an emotion cue; rare-word density falls with length."),
    ],
    "collides.": [
        ("fig06_class_overlap.png",
         "All six classes are lexically close, because the shared “i feel …” frame dominates every class."),
    ],
    "and balanced class weights.": [
        ("fig12_classical.png",
         "Per-class F1 tracks class size (ρ = +0.94); cross-validation settles three design questions."),
    ],
    "instruction-following — held.": [
        ("fig08_clustering_metrics.png",
         "Silhouette stays near zero at every k, and agreement with the labels (ARI, NMI) stays near chance."),
        ("fig09_cluster_label_contingency.png",
         "KMeans clusters do not recover the emotions: every cluster mixes every label."),
    ],
    "string matching had missed.": [
        ("fig10_lsh.png",
         "No LSH setting beats exact search: true neighbours sit at cosine 0.54, on the shoulder of the bulk."),
    ],
    "*cost* 2.2 macro F1 points.": [
        ("fig11_apriori.png",
         "Word pairs add nothing over single words; one adjective is enough to fix the label."),
    ],
}


def main():
    md = REPORT.read_text()

    # 1. tables -> image links, in order
    lines, out, t, i = md.splitlines(), [], 0, 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1]):
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            t += 1
            out.append(f"![{TABLE_CAPTIONS[t - 1]}]({FIG}/medium/table{t:02d}.png)")
            i = j
            continue
        out.append(lines[i])
        i += 1
    assert t == len(TABLE_CAPTIONS), f"expected {len(TABLE_CAPTIONS)} tables, found {t}"
    md = "\n".join(out)

    # 2. the report's own figures: image + italic "Figure N." caption -> one image link
    md, n_existing = re.subn(
        r"!\[[^\]]*\]\((figures/[^)]+)\)\s*\n\s*\n\*Figure \d+\.\s*(.*?)\*",
        lambda m: f"![{' '.join(m.group(2).split())}](reports/{m.group(1)})", md, flags=re.S)
    assert n_existing == 2, n_existing

    # 3. added figures, after the paragraph containing each anchor
    for anchor, figs in FIGURES.items():
        assert md.count(anchor) == 1, f"anchor must be unique: {anchor!r} ({md.count(anchor)})"
        end = md.find("\n\n", md.find(anchor))
        block = "".join(f"\n\n![{cap}]({FIG}/{f})" for f, cap in figs)
        md = md[:end] + block + md[end:]

    # 4. the Medium draft already holds the title
    md = md[md.index("## Abstract"):]
    md = ("*Accuracy, latency and cost on the Emotions Dataset for NLP. "
          "CMPE 255 Data Science, Assignment 1.*\n\n") + md
    (ROOT / "reports" / "medium_article.md").write_text(md)

    # 5. ordered blocks for pasting. Medium has two heading sizes: h3 (large), h4 (small).
    html = markdown.markdown(md, extensions=["fenced_code", "sane_lists"])
    html = html.replace("<h3>", "<h4>").replace("</h3>", "</h4>")
    html = html.replace("<h2>", "<h3>").replace("</h2>", "</h3>")
    html = html.replace("<hr />", "")
    parts = re.split(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>', html)
    segs = []
    for k in range(0, len(parts), 3):
        chunk = parts[k].strip()
        if chunk:
            segs.append({"type": "html", "html": chunk})
        if k + 2 < len(parts):
            path = ROOT / parts[k + 2]
            assert path.exists(), path
            segs.append({"type": "image", "path": str(path),
                         "caption": parts[k + 1].replace("&amp;", "&").replace("&quot;", '"')})
    (ROOT / "reports" / "medium_segments.json").write_text(json.dumps(segs, indent=1, ensure_ascii=False))
    n_img = sum(s["type"] == "image" for s in segs)
    print(f"{len(segs)} segments: {len(segs) - n_img} html blocks, {n_img} images")
    for s in segs:
        if s["type"] == "image":
            print("   IMG", Path(s["path"]).name, "|", s["caption"][:72])


if __name__ == "__main__":
    main()
