"""Figures for the inner-tolerance sweep."""

from __future__ import annotations

from pathlib import Path

from experiments._paths import study_dir

from ._data import column, load, select
from ._style import condition_label, plt, save_figure

STUDY = "sigma_sweep"
KINDS = ("equality QP", "LP", "inequality QP")
CONDITIONS = ("moderately-conditioned", "mildly-ill-conditioned")
QUANTITIES = (
    ("outer_iterations", "outer iterations to target"),
    ("inner_iterations", "block projections to target"),
    ("runtime_seconds", "solver time (seconds)"),
)


def plot_work(rows: list[dict], output: Path) -> Path:
    """Plot outer work, inner projections, and runtime against ``sigma``."""
    kinds = [kind for kind in KINDS if select(rows, problem=kind)]
    fig, axes = plt.subplots(
        len(QUANTITIES),
        len(kinds),
        figsize=(4 * len(kinds), 9.5),
        sharex=True,
        squeeze=False,
    )
    for column_index, kind in enumerate(kinds):
        for condition in CONDITIONS:
            selected = sorted(
                select(rows, problem=kind, matrix=condition),
                key=lambda row: row["sigma"],
            )
            if not selected:
                continue
            sigmas = column(selected, "sigma")
            label = condition_label(condition, selected[0].get("matrix_condition"))
            for axis_index, (field, _) in enumerate(QUANTITIES):
                axes[axis_index, column_index].semilogy(
                    sigmas, column(selected, field), "o-", label=label
                )

        axes[0, column_index].set_title(kind)
        axes[-1, column_index].set_xlabel(r"inner tolerance $\sigma$")
        for axis_index in range(len(QUANTITIES)):
            axis = axes[axis_index, column_index]
            axis.grid(True, which="both", alpha=0.3)
            axis.legend(fontsize=8)

    for axis_index, (_, ylabel) in enumerate(QUANTITIES):
        axes[axis_index, 0].set_ylabel(ylabel)
    return save_figure(fig, output)


def make(results_dir: Path | None = None) -> list[Path]:
    rows = load(STUDY, results_dir=results_dir)
    output = study_dir(STUDY, results_dir)
    return [plot_work(rows, output / "iteration_counts.pdf")]
