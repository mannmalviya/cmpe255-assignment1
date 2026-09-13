"""Phase 16 — Head-to-head comparison, assembled ONLY from metrics JSON files.

SPLIT USED: none new. Every number is read from reports/metrics/*.json, which hold
test-split results produced in Phases 11-15. Nothing is re-run or re-scored.

Before any table is built, two audits run, and either one failing stops the phase:

  1. HARDWARE AUDIT. Every metrics file carries a hardware block. Its stable
     fingerprint (CPU model, logical cores, RAM, GPU model, GPU memory, driver)
     must match reports/metrics/hardware_reference.json. If ANY file differs, the
     timings are not comparable and the phase raises instead of printing a table.
  2. PROTOCOL AUDIT. Every timing row must report >= 200 timed single-row runs
     after >= 20 discarded warm-up runs, and name the device it ran on.

Timings are reported in separate CPU-served and GPU-served tables and are never
mixed or averaged within one column. Quality metrics are device-independent (the
Phase 15 CPU and GPU runs agree on every test row) and share one table.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

import viz
from config import (DOCS, FIGURES, LABELS, LATENCY_BUDGET_P95_MS,
                    LATENCY_MIN_RUNS, LATENCY_WARMUP_RUNS, METRICS,
                    TARGET_MACRO_F1, ensure_dirs)
from hardware import save_metrics, stamp_fingerprint

MONTHLY_VOLUME = 1_000_000


def load(name):
    return json.loads((METRICS / f"{name}.json").read_text())


# ---------------------------------------------------------------- audits

def hardware_audit() -> dict:
    ref = json.loads((METRICS / "hardware_reference.json").read_text())
    rows, mismatches, torch_versions = [], [], {}
    for path in sorted(METRICS.glob("phase*.json")):
        hw = json.loads(path.read_text()).get("hardware")
        if hw is None:
            mismatches.append({"file": path.name, "problem": "no hardware block"})
            continue
        fp = stamp_fingerprint(hw)
        ok = fp == ref["fingerprint"]
        rows.append({"file": path.name, "device_used": hw.get("device_used"),
                     "torch_version": hw.get("torch_version"), "fingerprint_match": ok})
        torch_versions.setdefault(str(hw.get("torch_version")), []).append(path.name)
        if not ok:
            mismatches.append({"file": path.name, "fingerprint": fp})
    if mismatches:
        raise RuntimeError(
            "HARDWARE AUDIT FAILED - timings are not comparable. Stop and re-run "
            f"every timing on one machine.\nreference: {ref['fingerprint']}\n"
            f"mismatches: {json.dumps(mismatches, indent=2)}")
    return {"reference_fingerprint": ref["fingerprint"], "files_checked": len(rows),
            "all_match": True, "files": rows,
            "torch_versions_seen": torch_versions}


def protocol_audit(models: list) -> list:
    problems = []
    for m in models:
        s = m["latency_single_row"]
        if s["n_runs"] < LATENCY_MIN_RUNS:
            problems.append(f"{m['model']}: {s['n_runs']} timed runs < {LATENCY_MIN_RUNS}")
        if s["warmup_runs_discarded"] < LATENCY_WARMUP_RUNS:
            problems.append(f"{m['model']}: warm-up {s['warmup_runs_discarded']}")
        if not m.get("device_used"):
            problems.append(f"{m['model']}: no device recorded")
    if problems:
        raise RuntimeError("PROTOCOL AUDIT FAILED:\n" + "\n".join(problems))
    return [{"model": m["model"], "timed_runs": m["latency_single_row"]["n_runs"],
             "warmup_discarded": m["latency_single_row"]["warmup_runs_discarded"],
             "device": m["device_used"]} for m in models]


# ---------------------------------------------------------------- assemble

def collect() -> list:
    p11, p12, p13 = load("phase11_baseline"), load("phase12_classical"), load("phase13_zeroshot")
    p14, p15g, p15c = load("phase14_fewshot"), load("phase15_finetune_gpu"), load("phase15_finetune_cpu")
    seeds, sig = load("phase15_seed_check"), load("phase15_significance")

    def tag(m, family, label, extra=None):
        return {**m, "family": family, "label": label, "extra": extra or {}}

    rows = [
        tag(p11["models"]["baseline_majority"], "baseline", "Majority class"),
        tag(p11["models"]["baseline_stratified"], "baseline", "Stratified random"),
        tag(p12["models"]["logreg"], "classical", "TF-IDF + logistic regression",
            {"cv_macro_f1": p12["models"]["logreg"]["notes"]["best_cv_macro_f1"]}),
        tag(p12["models"]["linear_svm"], "classical", "TF-IDF + linear SVM",
            {"cv_macro_f1": p12["models"]["linear_svm"]["notes"]["best_cv_macro_f1"]}),
        tag(p12["models"]["gradient_boosting"], "classical", "TF-IDF + LightGBM",
            {"cv_macro_f1": p12["models"]["gradient_boosting"]["notes"]["best_cv_macro_f1"]}),
        tag(p13["model"], "prompted LLM", "Qwen2.5-1.5B zero-shot",
            {"unparseable": p13["model"]["parsing"]["n_unparseable_strict"]}),
        tag(p14["model"], "prompted LLM", "Qwen2.5-1.5B few-shot (12)",
            {"unparseable": p14["model"]["parsing"]["n_unparseable_strict"]}),
    ]
    seed_f1 = [p15g["model"]["quality"]["macro_f1"]] + [r["test_macro_f1"] for r in seeds["runs"]]
    ft_extra = {"seed_mean_macro_f1": round(float(np.mean(seed_f1)), 4),
                "seed_sd_macro_f1": round(float(np.std(seed_f1, ddof=1)), 4),
                "n_seeds": len(seed_f1),
                "bootstrap_margin_ci95": [sig["paired_bootstrap_macro_f1"]["ci95_low"],
                                          sig["paired_bootstrap_macro_f1"]["ci95_high"]],
                "p_margin_ge_3pts": sig["paired_bootstrap_macro_f1"]["p_margin_ge_threshold"],
                "mcnemar_p": sig["mcnemar_exact"]["exact_p_value"]}
    rows.append(tag(p15g["model"], "fine-tuned encoder", "DistilBERT fine-tuned (GPU)", ft_extra))
    rows.append(tag(p15c["model"], "fine-tuned encoder", "DistilBERT fine-tuned (CPU)", ft_extra))
    return rows


def fmt_usd(x):
    return f"${x:.2e}" if x < 0.01 else f"${x:.4f}"


def build_tables(rows, best_classical_f1) -> str:
    bar = best_classical_f1 + 0.03
    lines = []

    # 1. quality (device-independent). DistilBERT GPU and CPU rows have identical
    # predictions, so it appears once here.
    lines += ["### Table A — Quality on the test split (device-independent)", "",
              "| Model | Family | Macro F1 | Accuracy | Acc − macro F1 | "
              + " | ".join(f"F1 {l}" for l in LABELS) + " | Clears 0.85 | Clears LLM bar ≥ "
              f"{bar:.4f} |",
              "|" + "---|" * 2 + "---:|" * (3 + len(LABELS)) + "---|---|"]
    for r in rows:
        if r["label"].endswith("(CPU)"):
            continue
        q = r["quality"]
        name = r["label"].replace(" (GPU)", "")
        lines.append(
            f"| {name} | {r['family']} | **{q['macro_f1']:.4f}** | {q['accuracy']:.4f} | "
            f"{q['accuracy_minus_macro_f1']:+.4f} | "
            + " | ".join(f"{q['per_class'][l]['f1']:.3f}" for l in LABELS)
            + f" | {'yes' if q['macro_f1'] >= TARGET_MACRO_F1 else 'no'} | "
            f"{'**yes**' if q['macro_f1'] >= bar else 'no'} |")

    # 2 + 3. cost and latency, split by the device the model actually ran on.
    for dev, title in (("cpu", "B — Served on CPU"), ("cuda", "C — Served on GPU")):
        sub = [r for r in rows if str(r["device_used"]).startswith(dev)]
        lines += ["", f"### Table {title} (`{sub[0]['device_used']}`)", "",
                  "| Model | Macro F1 | p50 ms | p95 ms | p99 ms | Within 50 ms p95 | "
                  "Best rows/s (batch) | $ / 1,000 | $ / month at 1 M | Disk MB | "
                  "Peak memory MB |",
                  "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
        for r in sub:
            s, b, c = r["latency_single_row"], r["throughput_batched"], r["cost"]
            lines.append(
                f"| {r['label']} | {r['quality']['macro_f1']:.4f} | {s['p50_ms']:.3f} | "
                f"{s['p95_ms']:.3f} | {s['p99_ms']:.3f} | "
                f"{'yes' if s['within_p95_budget'] else '**no**'} | "
                f"{b['best_rows_per_second']:,.1f} ({b['best_batch_size']}) | "
                f"{fmt_usd(c['usd_per_1000_predictions'])} | "
                f"{fmt_usd(c['usd_per_1000_predictions'] * MONTHLY_VOLUME / 1000)} | "
                f"{r['model_size_on_disk_mb']:,.3f} | "
                f"{r['peak_memory_mb'] if r['peak_memory_mb'] else '—'} |")
    return "\n".join(lines)


def decision(rows, best) -> list:
    bar = best["quality"]["macro_f1"] + 0.03
    out = []
    for r in rows:
        if r["family"] not in ("prompted LLM", "fine-tuned encoder"):
            continue
        q, s = r["quality"], r["latency_single_row"]
        out.append({
            "model": r["label"], "device": r["device_used"],
            "macro_f1": q["macro_f1"],
            "margin_vs_best_classical": round(q["macro_f1"] - best["quality"]["macro_f1"], 4),
            "passes_accuracy_bar": bool(q["macro_f1"] >= bar),
            "p95_ms": s["p95_ms"], "passes_latency_budget": bool(s["within_p95_budget"]),
            "worth_it_under_phase1_rule": bool(q["macro_f1"] >= bar and s["within_p95_budget"]),
            "latency_x_best_classical": round(s["p95_ms"] / best["latency_single_row"]["p95_ms"], 1),
            "cost_x_best_classical": round(r["cost"]["usd_per_1000_predictions"]
                                          / best["cost"]["usd_per_1000_predictions"], 1),
            "extra_monthly_usd_at_1M": round(
                (r["cost"]["usd_per_1000_predictions"]
                 - best["cost"]["usd_per_1000_predictions"]) * MONTHLY_VOLUME / 1000, 4),
        })
    return out


def figure(rows, best):
    """2x2: full range on top (context), zoom on the contenders below (the decision).

    A single 0-1 panel put DistilBERT (0.898) on top of the 0.888 bar, hiding the
    one comparison the figure exists to show. The zoom row carries the seed range
    as an error bar, so the reader sees how narrowly the worst seed clears the bar.
    """
    viz.apply_style()
    bar = best["quality"]["macro_f1"] + 0.03
    seeds = json.loads((METRICS / "phase15_seed_check.json").read_text())["runs"]
    ft_f1 = [r["test_macro_f1"] for r in seeds]
    short = {"Majority class": "majority", "Stratified random": "stratified",
             "TF-IDF + logistic regression": "logreg", "TF-IDF + linear SVM": "linear SVM",
             "TF-IDF + LightGBM": "LightGBM", "Qwen2.5-1.5B zero-shot": "Qwen zero-shot",
             "Qwen2.5-1.5B few-shot (12)": "Qwen few-shot",
             "DistilBERT fine-tuned (GPU)": "DistilBERT GPU",
             "DistilBERT fine-tuned (CPU)": "DistilBERT CPU"}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4),
                             gridspec_kw={"height_ratios": [1, 1.05]})

    def xval(r, key):
        return (r["cost"]["usd_per_1000_predictions"] if key == "cost"
                else r["latency_single_row"]["p95_ms"])

    def draw(ax, key, zoom, offsets):
        for r in rows:
            y = r["quality"]["macro_f1"]
            if zoom and y < 0.8:
                continue
            x = xval(r, key)
            gpu = str(r["device_used"]).startswith("cuda")
            n = short[r["label"]]
            if zoom and n.startswith("DistilBERT"):
                lo = min(ft_f1 + [y]); hi = max(ft_f1 + [y])
                ax.plot([x, x], [lo, hi], color=viz.INK_2, lw=1.2, zorder=3)
                for v in (lo, hi):
                    ax.plot([x / 1.12, x * 1.12], [v, v], color=viz.INK_2, lw=1.2, zorder=3)
            ax.scatter(x, y, s=52, color=viz.ACCENT if gpu else viz.PRIMARY,
                       marker="s" if gpu else "o", zorder=5,
                       edgecolors=viz.SURFACE, linewidths=1.2)
            ax.annotate(n, (x, y), xytext=offsets.get(n, (7, 0)),
                        textcoords="offset points", fontsize=7.5, color=viz.INK_2,
                        va="center")
        ax.axhline(bar, color=viz.INK_2, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.set_xscale("log")
        viz.value_grid(ax, "y")

    full_off = {"logreg": (7, -9), "linear SVM": (-64, 0), "LightGBM": (7, -9),
                "DistilBERT GPU": (-40, 12), "DistilBERT CPU": (7, -11),
                "Qwen zero-shot": (7, 5), "Qwen few-shot": (7, -9),
                "majority": (7, -5), "stratified": (7, 6)}
    zoom_off = {"logreg": (8, -2), "linear SVM": (8, 2), "LightGBM": (8, 0),
                "DistilBERT GPU": (-86, 0), "DistilBERT CPU": (10, 0)}

    (a, b), (c, d) = axes
    for ax, key in ((a, "cost"), (b, "p95")):
        draw(ax, key, False, full_off)
        ax.set_ylim(0, 1.0)
    for ax, key in ((c, "cost"), (d, "p95")):
        draw(ax, key, True, zoom_off)
        ax.set_ylim(0.83, 0.91)

    a.set_xlim(1e-8, 0.08); c.set_xlim(1e-7, 3e-3)
    b.set_xlim(0.06, 400);  d.set_xlim(0.1, 120)
    for ax in (b, d):
        ax.axvline(LATENCY_BUDGET_P95_MS, color=viz.INK_2, lw=1.0, ls=(0, (1, 2)), zorder=2)
        ax.axvspan(LATENCY_BUDGET_P95_MS, 1e4, color=viz.REFERENCE, alpha=0.4, zorder=1)
    b.text(56, 0.07, "outside\n50 ms budget", fontsize=7, color=viz.INK_2)
    d.text(54, 0.834, "outside\nbudget", fontsize=7, color=viz.INK_2)
    for ax, x0 in ((c, 1.3e-7), (d, 0.12)):
        ax.text(x0, bar + 0.0016,
                f"LLM bar: macro F1 {bar:.4f} (SVM + 3 points)", fontsize=7,
                color=viz.INK_2)
    for ax in (a, c):
        ax.set_xlabel("cost per 1,000 predictions (USD, log)")
    for ax in (b, d):
        ax.set_xlabel("p95 single-row latency (ms, log)")
    for ax in (a, b, c, d):
        ax.set_ylabel("test macro F1")
    a.legend([plt.Line2D([], [], marker="o", ls="", color=viz.PRIMARY),
              plt.Line2D([], [], marker="s", ls="", color=viz.ACCENT)],
             ["served on CPU", "served on GPU"], loc="lower right")

    viz.titles(a, "a.  Accuracy against cost, all models",
               "dashed = LLM bar (SVM + 3 points). Prompted LLMs: costliest, least accurate")
    viz.titles(b, "b.  Accuracy against latency, all models",
               "dashed = LLM bar; dotted = 50 ms budget. Both prompted LLMs outside it")
    viz.titles(c, "c.  Zoom: the contenders, against cost",
               "vertical bar on DistilBERT = range over 3 training seeds")
    viz.titles(d, "d.  Zoom: the contenders, against latency",
               "only DistilBERT clears the bar; its worst seed clears it by 0.16 points")
    viz.caption(fig, "All points read from reports/metrics/*.json (test split). One "
                     "machine; hardware fingerprint verified for all files. Device "
                     "marked per point. CPU and GPU timings are never mixed in a column.")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_tradeoff.png")
    plt.close(fig)


def main():
    ensure_dirs()
    hw = hardware_audit()
    print(f"HARDWARE AUDIT: {hw['files_checked']} files, all fingerprints match: "
          f"{hw['all_match']}")
    print(f"  torch versions seen: { {k: len(v) for k, v in hw['torch_versions_seen'].items()} }")

    rows = collect()
    proto = protocol_audit(rows)
    print(f"PROTOCOL AUDIT: {len(proto)} timing rows, all >= {LATENCY_MIN_RUNS} runs "
          f"with >= {LATENCY_WARMUP_RUNS} warm-up discarded, device recorded")

    best = next(r for r in rows if r["model"] == "linear_svm")
    tables = build_tables(rows, best["quality"]["macro_f1"])
    dec = decision(rows, best)
    figure(rows, best)

    (DOCS / "phase16_tables_generated.md").write_text(
        "<!-- GENERATED by src/phase16_comparison.py from reports/metrics/*.json. "
        "Do not edit by hand. -->\n\n" + tables + "\n")

    with open(METRICS / "phase16_comparison_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "family", "device", "macro_f1", "accuracy",
                    *[f"f1_{l}" for l in LABELS], "p50_ms", "p95_ms", "p99_ms",
                    "best_rows_per_s", "usd_per_1000", "disk_mb", "peak_memory_mb"])
        for r in rows:
            q, s = r["quality"], r["latency_single_row"]
            w.writerow([r["label"], r["family"], r["device_used"], q["macro_f1"],
                        q["accuracy"], *[q["per_class"][l]["f1"] for l in LABELS],
                        s["p50_ms"], s["p95_ms"], s["p99_ms"],
                        r["throughput_batched"]["best_rows_per_second"],
                        r["cost"]["usd_per_1000_predictions"],
                        r["model_size_on_disk_mb"], r["peak_memory_mb"]])

    save_metrics("phase16_comparison", {
        "phase": "16_comparison",
        "splits_used": ["none new - reads test results from Phases 11-15"],
        "source_files": sorted(p.name for p in METRICS.glob("phase1[1-5]*.json")),
        "hardware_audit": hw, "protocol_audit": proto,
        "decision_rule": "LLM worth it only if macro F1 >= best classical + 0.03 AND "
                         "p95 <= 50 ms (Phase 1, fixed before any result)",
        "best_classical": {"model": "linear_svm",
                           "macro_f1": best["quality"]["macro_f1"]},
        "decisions": dec,
        "fine_tuned_uncertainty": rows[-1]["extra"],
    }, device="n/a (aggregation only)")

    print("\n" + tables)
    print("\n=== DECISION RULE, PER LLM CONFIGURATION ===")
    for d in dec:
        print(f"  {d['model']:<30} margin {d['margin_vs_best_classical']:+.4f}  "
              f"p95 {d['p95_ms']:>8.2f} ms  worth it: {d['worth_it_under_phase1_rule']}  "
              f"({d['cost_x_best_classical']:,.0f}x cost, "
              f"+${d['extra_monthly_usd_at_1M']:.4f}/month at 1M)")


if __name__ == "__main__":
    main()
