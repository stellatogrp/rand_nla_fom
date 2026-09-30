"""Figures for the split-form LASSO study."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments._paths import study_dir

from ._data import column, load, load_histories, select
from ._style import SOLVER_COLORS, SOLVER_LABELS, plt, save_figure

STUDY = "lasso"
DESIGNS = ("independent", "correlated")
DESIGN_LABELS = {
    "independent": "independent columns",
    "correlated": r"correlated columns ($\rho=0.9$)",
}
SOLVERS = (
    "schur_cg",
    "schur_block_kaczmarz",
    "schur_randomized_kaczmarz",
    "coupled_gmres",
)
YLABEL = "relative objective error + feasibility"
# Stages that keep convergence traces, with the figure each one draws.
TRACE_FIGURES = {
    "solver": "solver_convergence.pdf",
    "size": "size_convergence.pdf",
    "hybrid": "hybrid_convergence.pdf",
}


def _variants(rows: list[dict]) -> list[str]:
    """Return the variants of ``rows`` in first-seen order."""
    seen: dict[str, None] = {}
    for row in rows:
        seen.setdefault(row["variant"], None)
    return list(seen)


def _variant_title(rows: list[dict], stage: str, variant: str) -> str:
    """Name one row of a trace figure by what distinguishes it."""
    row = select(rows, stage=stage, variant=variant)[0]
    if stage == "solver":
        label = DESIGN_LABELS.get(variant, variant)
        return rf"{label}, $\kappa(D)={row['design_condition']:.3g}$"
    if stage == "size":
        return rf"$m={int(row['m'])}$, $p={int(row['p'])}$"
    coefficients = ", ".join(f"{row[name]:.3g}" for name in ("c1", "c2", "c3", "c4"))
    return rf"{variant}: $(c_1, c_2, c_3, c_4) = ({coefficients})$"


def plot_traces(
    results: list[dict], histories: list[dict], stage: str, output: Path
) -> Path:
    """Accuracy against outer and cumulative inner iterations, per solver.

    One row per variant of the stage: the design for ``solver``, the instance
    size for ``size``, and the penalty for ``hybrid``.
    """
    variants = _variants(select(results, stage=stage))
    fig, axes = plt.subplots(
        len(variants), 2, figsize=(9, 3.6 * len(variants)), squeeze=False
    )
    for row_index, variant in enumerate(variants):
        for solver in SOLVERS:
            trace = sorted(
                select(histories, stage=stage, variant=variant, label=solver),
                key=lambda row: row["iteration"],
            )
            if not trace:
                continue
            accuracy = column(trace, "accuracy")
            style = dict(
                color=SOLVER_COLORS.get(solver, "#808080"),
                label=SOLVER_LABELS.get(solver, solver),
            )
            axes[row_index, 0].semilogy(column(trace, "iteration"), accuracy, **style)
            axes[row_index, 1].semilogy(
                column(trace, "cumulative_inner_iterations"), accuracy, **style
            )
        axes[row_index, 0].set_ylabel(YLABEL)
        axes[row_index, 0].set_title(
            _variant_title(results, stage, variant), fontsize=9
        )
        axes[row_index, 1].set_xscale("log")
        for axis in axes[row_index]:
            axis.grid(True, which="both", alpha=0.3)
            axis.legend(fontsize=7)
    axes[-1, 0].set_xlabel("outer iteration")
    axes[-1, 1].set_xlabel("cumulative inner iterations")
    return save_figure(fig, output)


def plot_inner_per_outer(
    results: list[dict], histories: list[dict], output: Path
) -> Path:
    """Inner iterations spent at each outer iteration, with the running mean.

    Section 4 bounds the expected inner work of one outer iteration by a
    constant that does not grow with the iteration count; a flat running mean
    is the observable consequence.
    """
    variants = _variants(select(results, stage="solver"))
    fig, axes = plt.subplots(
        1, len(variants), figsize=(4.6 * len(variants), 3.8), squeeze=False
    )
    for axis, variant in zip(axes[0], variants, strict=True):
        for solver in SOLVERS:
            trace = sorted(
                select(histories, stage="solver", variant=variant, label=solver),
                key=lambda row: row["iteration"],
            )
            if not trace:
                continue
            iterations = column(trace, "iteration")
            counts = column(trace, "inner_iterations")
            color = SOLVER_COLORS.get(solver, "#808080")
            axis.plot(iterations, counts, color=color, alpha=0.35, linewidth=0.7)
            axis.plot(
                iterations,
                np.cumsum(counts) / iterations,
                color=color,
                linewidth=1.6,
                label=SOLVER_LABELS.get(solver, solver),
            )
        axis.set_yscale("log")
        axis.set_xlabel("outer iteration $k$")
        axis.set_title(_variant_title(results, "solver", variant), fontsize=9)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7)
    axes[0, 0].set_ylabel("inner iterations $N_k$ (thin) and running mean (thick)")
    return save_figure(fig, output)


def _sweep_panel(axis, rows: list[dict], field: str, xfield: str) -> None:
    for design in DESIGNS:
        selected = sorted(select(rows, design=design), key=lambda row: row[xfield])
        if not selected:
            continue
        x = column(selected, xfield)
        values = column(selected, field)
        (line,) = axis.plot(x, values, "o-", markersize=4, label=DESIGN_LABELS[design])
        failed = ~np.array([row["reached_accuracy"] for row in selected], dtype=bool)
        if failed.any():
            axis.scatter(
                x[failed], values[failed], marker="x", s=55, color=line.get_color()
            )
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7)


def plot_sweeps(results: list[dict], output: Path) -> Path:
    """Outer and inner work against the metric and the inner tolerance."""
    gamma_rows = select(results, stage="gamma")
    sigma_rows = select(results, stage="sigma")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), squeeze=False)
    for row_index, field in enumerate(("outer_iterations", "inner_iterations")):
        _sweep_panel(axes[row_index, 0], gamma_rows, field, "gamma_x")
        _sweep_panel(axes[row_index, 1], sigma_rows, field, "sigma")
        axes[row_index, 0].set_xscale("log")
    axes[0, 0].set_ylabel("outer iterations to target")
    axes[1, 0].set_ylabel("block projections to target")
    axes[1, 0].set_xlabel(r"shared step size $\gamma$")
    axes[1, 1].set_xlabel(r"inner tolerance $\sigma$")
    return save_figure(fig, output)


def make(results_dir: Path | None = None) -> list[Path]:
    results = load(STUDY, results_dir=results_dir)
    output = study_dir(STUDY, results_dir)
    written = []
    if select(results, stage="gamma") or select(results, stage="sigma"):
        written.append(plot_sweeps(results, output / "parameter_sweeps.pdf"))
    traced = [stage for stage in TRACE_FIGURES if select(results, stage=stage)]
    if traced:
        histories = load_histories(STUDY, results_dir)
        for stage in traced:
            written.append(
                plot_traces(results, histories, stage, output / TRACE_FIGURES[stage])
            )
        if "solver" in traced:
            written.append(
                plot_inner_per_outer(results, histories, output / "inner_per_outer.pdf")
            )
    return written
