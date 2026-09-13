"""Shared figure style for the study.

One design system for every figure: validated categorical palette, recessive
chrome, thin marks, rounded data-ends, direct labels. Print- and grayscale-safe.

Palette provenance: the categorical slots below were validated with the
dataviz skill's checker at `--mode light --pairs all`:
  lightness band PASS, chroma floor PASS,
  worst all-pairs CVD dE 9.2 (deutan), worst normal-vision dE 24.0,
  contrast WARN on aqua (2.74:1) -> relief rule applied, every figure using
  aqua carries visible direct labels.
Do not add a fourth categorical slot without re-running that validator.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# --- palette ------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua (validated)
PRIMARY = SERIES[0]
ACCENT = SERIES[1]
REFERENCE = "#d8d7d0"                         # neutral "all data" comparison fill

# Sequential blue ramp, light -> dark (magnitude encoding only).
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]

FONT = ["DejaVu Sans", "system-ui", "sans-serif"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


# --- rounded data-ends ---------------------------------------------------

def _pt_to_data(ax, radius_pt: float) -> tuple:
    """Convert a radius in typographic points to (dx, dy) in data units."""
    bbox = ax.get_window_extent()
    dpi = ax.figure.dpi
    px = radius_pt * dpi / 72.0
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return abs(x1 - x0) * px / bbox.width, abs(y1 - y0) * px / bbox.height


def rounded_bars(ax, patches, orient: str = "h", radius_pt: float = 3.5) -> None:
    """Replace plain bars with bars rounded on the data end only.

    The baseline end stays square so the bar visibly sits on the axis; only the
    end that encodes the value is rounded. Call after the axis limits are final.
    """
    ax.figure.canvas.draw()
    rx, ry = _pt_to_data(ax, radius_pt)
    for p in list(patches):
        x0, y0 = p.get_x(), p.get_y()
        w, h = p.get_width(), p.get_height()
        if w <= 0 or h <= 0:
            continue
        color, alpha = p.get_facecolor(), p.get_alpha()
        # x and y radii are computed in their own data units, so the corner
        # renders circular on the page regardless of the axis scales.
        if orient == "h":
            a, b = min(rx, w), min(ry, h / 2.0)
            verts = [(x0, y0), (x0 + w - a, y0), (x0 + w, y0), (x0 + w, y0 + b),
                     (x0 + w, y0 + h - b), (x0 + w, y0 + h), (x0 + w - a, y0 + h),
                     (x0, y0 + h), (x0, y0)]
        else:
            a, b = min(rx, w / 2.0), min(ry, h)
            verts = [(x0, y0), (x0, y0 + h - b), (x0, y0 + h), (x0 + a, y0 + h),
                     (x0 + w - a, y0 + h), (x0 + w, y0 + h), (x0 + w, y0 + h - b),
                     (x0 + w, y0), (x0, y0)]
        codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
                 Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
        p.set_visible(False)
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=color,
                               edgecolor="none", alpha=alpha, zorder=p.zorder))


# --- chrome helpers ------------------------------------------------------

def swatch(color: str):
    """A legend proxy. Needed because rounded_bars() hides the original patches,
    which would otherwise leave the legend with invisible handles."""
    return plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")


def titles(ax, title: str, subtitle: str = "") -> None:
    """Bold title, muted subtitle beneath it. Text never wears a series color."""
    if subtitle:
        ax.set_title(f"{title}\n", loc="left", pad=14)
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, color=INK_2)
    else:
        ax.set_title(title, loc="left", pad=8)


def value_grid(ax, axis: str = "x") -> None:
    """A hairline grid on the value axis only, drawn behind the marks."""
    ax.grid(axis=axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def caption(fig, text: str) -> None:
    fig.text(0.0, -0.015, text, ha="left", va="top", fontsize=7, color=MUTED,
             wrap=True)
