"""Matplotlib setup: a colorblind-safe palette and the recessive chart chrome.

Curves use the Okabe-Ito palette, validated with the dataviz checks: six
fixed hues for the swept parameter (five Okabe-Ito hues and Paul Tol's
wine), vermillion for the paper's recommended
value, black for the exact baseline, and the muted text token for the
near-exact context series.  Every series also has its own marker shape, so
identity never rests on hue alone.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rand-nla-fom-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

SURFACE = "#ffffff"
# All text (titles, axis labels, tick labels, legends, annotations) and the
# axes themselves (spines and tick marks) are solid black.  The secondary ink
# is reserved for non-text chrome: whiskers, outlines, and the like.  The grid
# and the contour lines of the trajectory figure stay faint so that they sit
# behind the data.
INK = "#000000"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#000000"
CONTOUR = "#c3c2b7"
# The Okabe-Ito colorblind-safe palette (Okabe & Ito, 2008).  The six chromatic
# colors below pass the dataviz validator's all-pairs check on white: the
# worst pair under deuteranopia (purple vs green, dE 7.6) sits in the band
# that is admissible with a secondary encoding, which the marker shapes
# supply; sky blue and orange are below 3:1 contrast on white, so every
# series also carries a legend entry and a marker.
OKABE_ITO = dict(
    blue="#0072B2",
    sky="#56B4E9",
    green="#009E73",
    orange="#E69F00",
    purple="#CC79A7",
    vermillion="#D55E00",
    black="#000000",
    yellow="#F0E442",
)
# Paul Tol's "wine" (muted scheme) extends the sweep to a sixth value, the
# boundary case sigma = 1.  With the five Okabe-Ito hues and vermillion it
# passes the dataviz validator's all-pairs check on white: lightness band,
# chroma floor, normal-vision floor (worst pair dE 15.6), and the worst CVD
# pair is still purple vs green.
TOL_WINE = "#882255"
# Fixed order for the swept parameter (never cycled past six values).
SWEEP_COLORS = (
    OKABE_ITO["blue"],
    OKABE_ITO["sky"],
    OKABE_ITO["green"],
    OKABE_ITO["orange"],
    OKABE_ITO["purple"],
    TOL_WINE,
)
SWEEP_MARKERS = ("o", "s", "^", "D", "v", "p")
ACCENT = OKABE_ITO["vermillion"]
ACCENT_MARKER = "P"
MEASURED = OKABE_ITO["blue"]
BOUND = OKABE_ITO["vermillion"]
BOUND_EXACT = OKABE_ITO["green"]
REFERENCE = OKABE_ITO["black"]
REFERENCE_MARKER = "X"
CONTEXT = MUTED
CONTEXT_MARKER = "d"
SEGMENTS = (OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["green"])
BAND_ALPHA = 0.12

# Figure geometry, in inches.  The paper is set at 11pt with 1in margins, so
# the text width is 6.5in; figures are drawn at that width and included with
# ``width=\textwidth``, so their fonts are not rescaled.  Heights stay well
# under half of the 9in text height so that a figure and its caption share a
# page with text instead of being pushed to a float page.
TEXT_WIDTH = 6.5
GRID_SIZE = (TEXT_WIDTH, 4.0)  # 2 x 2 panels (convergence, work)
ZOOM_SIZE = (TEXT_WIDTH, 3.8)  # 2 x 2 panels of medians only; room for the row labels
MODEL_SIZE = (TEXT_WIDTH, 2.7)  # 1 x 2 panels (cost model)
TRAJECTORY_SIZE = (TEXT_WIDTH, 3.4)  # 1 x 2 equal-aspect panels
RATIO_SIZE = (4.4, 3.0)  # single panel, included at 0.7 of the text width
BAR_ROW = 0.22  # height per bar of the wall-clock charts ...
BAR_BASE = 1.5  # ... plus room for the legend and the axis


def sweep_colors(count: int) -> list[str]:
    """The first ``count`` sweep colors, in their fixed order."""
    if count > len(SWEEP_COLORS):
        raise ValueError(
            f"the sweep palette has {len(SWEEP_COLORS)} validated colors, not {count}"
        )
    return list(SWEEP_COLORS[:count])


def sweep_markers(count: int) -> list[str]:
    """Marker shapes matching :func:`sweep_colors`."""
    return list(SWEEP_MARKERS[:count])


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "medium",
            "axes.titlecolor": INK,
            "axes.labelsize": 8,
            "axes.labelcolor": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.facecolor": SURFACE,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=list(SWEEP_COLORS)),
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "legend.labelcolor": INK,
            "lines.linewidth": 1.5,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "figure.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.facecolor": SURFACE,
            "mathtext.fontset": "dejavusans",
        }
    )


def save(fig, stem: Path, formats=("png", "pdf")) -> list[Path]:
    """Save a figure in every format and close it."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths
