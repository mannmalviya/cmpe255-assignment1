"""Render every markdown table in reports/final_report.md as a PNG, for Medium.

Medium's editor has no table support, so the report's tables are published as
images. They are drawn from the report file itself, so no number is retyped, in
the same palette and type as the study's figures (src/viz.py).

Output: reports/figures/medium/tableNN.png, plus manifest.json mapping each table
to the section heading it sits under.
"""

import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import viz
from config import PROJECT_ROOT

REPORT = PROJECT_ROOT / "reports" / "final_report.md"
OUT = PROJECT_ROOT / "reports" / "figures" / "medium"
WRAP = 34            # characters per line inside a cell before wrapping
DPI = 220


def clean(cell: str):
    bold = "**" in cell
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", cell)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"(?<!\w)\*(.*?)\*(?!\w)", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    return t.strip(), bold


def parse_tables(md: str):
    lines, tables, heading = md.splitlines(), [], ""
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            heading = ln.lstrip("#").strip()
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1]):
            header = [c for c in ln.strip("|").split("|")]
            aligns = ["right" if c.strip().endswith(":") else "left"
                      for c in lines[i + 1].strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(lines[j].strip("|").split("|"))
                j += 1
            tables.append({"heading": heading, "header": header, "aligns": aligns,
                           "rows": rows, "line": i + 1})
            i = j
            continue
        i += 1
    return tables


def render(tab, path: Path):
    header = [clean(c) for c in tab["header"]]
    rows = [[clean(c) for c in r] for r in tab["rows"]]
    ncol = len(header)
    wrapped = lambda s: textwrap.wrap(s, WRAP) or [""]

    # column widths in characters, capped by the wrap width
    widths = []
    for c in range(ncol):
        cells = [header[c][0]] + [r[c][0] for r in rows if c < len(r)]
        widths.append(max(max(len(x) for x in wrapped(s)) for s in cells) + 2)
    total_chars = sum(widths)

    char_w, line_h = 0.083, 0.225            # inches at 10.5 pt
    row_heights = []
    for r in [header] + rows:
        n = max(len(wrapped(r[c][0])) if c < len(r) else 1 for c in range(ncol))
        row_heights.append(n * line_h + 0.16)
    fig_w = max(4.0, total_chars * char_w + 0.3)
    fig_h = sum(row_heights) + 0.25

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(fig_h, 0)
    ax.axis("off")
    fig.patch.set_facecolor(viz.SURFACE)

    x_edges = [0.15]
    for w in widths:
        x_edges.append(x_edges[-1] + w * char_w)
    y = 0.12
    for ri, (r, h) in enumerate(zip([header] + rows, row_heights)):
        if ri > 0 and ri % 2 == 0:
            ax.add_patch(Rectangle((0.15, y), x_edges[-1] - 0.15, h, color="#f3f3f0", lw=0))
        for c in range(ncol):
            text, bold = r[c] if c < len(r) else ("", False)
            lines = wrapped(text)
            align = tab["aligns"][c] if c < len(tab["aligns"]) else "left"
            x = x_edges[c + 1] - 0.08 if align == "right" else x_edges[c] + 0.08
            for li, t in enumerate(lines):
                ax.text(x, y + 0.13 + li * line_h + line_h / 2, t, ha=align, va="center",
                        fontsize=10.5, family="DejaVu Sans",
                        color=viz.INK if (ri == 0 or bold) else "#2b2c30",
                        fontweight="bold" if (ri == 0 or bold) else "normal")
        y += h
        if ri == 0:
            ax.plot([0.15, x_edges[-1]], [y, y], color=viz.INK_2, lw=0.9)
    ax.plot([0.15, x_edges[-1]], [0.12, 0.12], color=viz.INK, lw=1.4)
    ax.plot([0.15, x_edges[-1]], [y, y], color=viz.INK, lw=1.4)
    fig.savefig(path, dpi=DPI, facecolor=viz.SURFACE, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = parse_tables(REPORT.read_text())
    manifest = []
    for n, tab in enumerate(tables, 1):
        path = OUT / f"table{n:02d}.png"
        render(tab, path)
        manifest.append({"file": path.name, "heading": tab["heading"], "line": tab["line"],
                         "first_header_cell": clean(tab["header"][0])[0],
                         "n_rows": len(tab["rows"])})
        print(f"  {path.name}  under '{tab['heading']}'  ({len(tab['rows'])} rows)")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(tables)} tables rendered to {OUT}")


if __name__ == "__main__":
    main()
