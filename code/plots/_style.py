"""Matplotlib setup and the figure idioms the studies share."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rand-nla-fom-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed categorical slots, assigned by inner solver and never cycled.
SOLVER_COLORS = {
    "schur_cg": "#2a78d6",
    "schur_block_kaczmarz": "#eb6834",
    "schur_randomized_kaczmarz": "#1baf7a",
    "coupled_gmres": "#eda100",
}
SOLVER_LABELS = {
    "schur_cg": "Schur CG",
    "schur_block_kaczmarz": "block Kaczmarz",
    "schur_randomized_kaczmarz": "randomized Kaczmarz",
    "coupled_gmres": "coupled GMRES",
}
CONDITION_LABELS = {
    "moderately-conditioned": r"$\kappa(A)=10$",
    "mildly-ill-conditioned": r"$\kappa(A)=20$",
    "ill-conditioned": r"$\kappa(A)=100$",
}


def save_figure(fig, output: Path) -> Path:
    """Lay out, save, and close a figure; return where it landed."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def condition_label(condition: str, matrix_condition: float | None = None) -> str:
    """Name a conditioning regime, preferring the measured condition number."""
    if matrix_condition is not None and np.isfinite(matrix_condition):
        return rf"$\kappa(A)={matrix_condition:g}$"
    return CONDITION_LABELS.get(condition, condition)


def matrix_label(rows: Sequence[dict]) -> str:
    """Describe the constraint-matrix dimensions common to ``rows``."""
    dimensions = {
        (int(row["m"]), int(row["n"]), int(row.get("solver_n", row["n"])))
        for row in rows
    }
    if len(dimensions) != 1:
        return "mixed matrix dimensions"
    constraints, variables, solver_variables = dimensions.pop()
    if solver_variables != variables:
        return (
            rf"$A$: ${constraints}\times{variables}$; "
            rf"$[A\ I]$: ${constraints}\times{solver_variables}$"
        )
    return rf"$A$: ${constraints}\times{variables}$"


def plot_convergence(
    traces: Iterable[tuple[str, Sequence[dict]]],
    title: str,
    ylabel: str,
    output: Path,
) -> Path:
    """Plot accuracy against outer and against cumulative inner iterations.

    ``traces`` pairs a legend label with the ``histories.csv`` rows of one
    run, ordered by ``iteration``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for label, rows in traces:
        accuracy = np.array([row["accuracy"] for row in rows], dtype=float)
        outer = np.array([row["iteration"] for row in rows], dtype=float)
        inner = np.array(
            [row["cumulative_inner_iterations"] for row in rows], dtype=float
        )
        axes[0].semilogy(outer, accuracy, label=label)
        axes[1].semilogy(inner, accuracy, label=label)
    axes[0].set_xlabel("outer iteration")
    axes[1].set_xlabel("cumulative inner iterations")
    axes[0].set_ylabel(ylabel)
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    fig.suptitle(title)
    return save_figure(fig, output)
