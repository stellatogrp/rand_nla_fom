"""Matplotlib setup: a colorblind-safe palette and the recessive chart chrome.

Curves use the Okabe-Ito palette, validated with the dataviz checks: five
fixed hues for the swept parameter, vermillion for the paper's recommended
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
# Titles, axis labels, and tick labels are solid black; the secondary ink is
# for legends and in-plot annotations.
INK = "#000000"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
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
# Fixed order for the swept parameter (never cycled past five values).
SWEEP_COLORS = (
    OKABE_ITO["blue"],
    OKABE_ITO["sky"],
    OKABE_ITO["green"],
    OKABE_ITO["orange"],
    OKABE_ITO["purple"],
)
SWEEP_MARKERS = ("o", "s", "^", "D", "v")
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
            "legend.labelcolor": INK_SECONDARY,
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
