"""Figures for the shared-step-size sweep."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments._paths import RESULTS, study_dir

from ._data import (
    MissingResults,
    column,
    load,
    load_histories,
    missing,
    select,
)
from ._style import condition_label, plt, save_figure

STUDY = "gamma_sweep"
INEQUALITY_RESULTS = "inequality_qp_results.csv"
KINDS = ("equality QP", "LP", "inequality QP")
CONDITIONS = ("moderately-conditioned", "ill-conditioned")
XLABEL = r"shared step size $\gamma$"


def _series(rows: list[dict], kind: str, condition: str) -> list[dict]:
    selected = select(rows, problem=kind, matrix=condition)
    return sorted(selected, key=lambda row: row["gamma"])


def _mark_failures(axis, line, gamma: np.ndarray, values: np.ndarray, failed) -> None:
    """Cross out the settings that never converged within the iteration cap."""
    axis.scatter(
        gamma[failed],
        values[failed],
        marker="x",
        s=55,
        color=line.get_color(),
        zorder=3,
    )


def plot_iteration_counts(rows: list[dict], output: Path) -> Path:
    """Outer and inner work against gamma, one column per problem family."""
    kinds = [kind for kind in KINDS if select(rows, problem=kind)]
    fig, axes = plt.subplots(
        2, len(kinds), figsize=(4 * len(kinds), 7), sharex=True, squeeze=False
    )
    for column_index, kind in enumerate(kinds):
        for condition in CONDITIONS:
            selected = _series(rows, kind, condition)
            if not selected:
                continue
            gamma = column(selected, "gamma")
            failed = ~np.array([row["converged"] for row in selected], dtype=bool)
            for axis_index, field in enumerate(
                ("outer_iterations", "inner_iterations")
            ):
                axis = axes[axis_index, column_index]
                values = column(selected, field)
                (line,) = axis.loglog(gamma, values, "o-", label=condition)
                _mark_failures(axis, line, gamma, values, failed)
        axes[0, column_index].set_title(kind)
        axes[1, column_index].set_xlabel(XLABEL)
        for axis_index in range(2):
            axis = axes[axis_index, column_index]
            axis.grid(True, which="both", alpha=0.3)
            axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("outer iterations")
    axes[1, 0].set_ylabel("total inner CG iterations")
    fig.suptitle(r"Section 5 with $\gamma_x=\gamma_\lambda=\gamma$")
    return save_figure(fig, output)


def plot_inequality(
    rows: list[dict], output: Path, histories: list[dict] | None = None
) -> Path:
    """Focused iteration-count and accuracy comparison for the inequality QP."""
    nrows = 2 if histories else 1
    fig, axes = plt.subplots(
        nrows, 2, figsize=(10, 7 if histories else 4.1), squeeze=False
    )
    for condition in CONDITIONS:
        selected = _series(rows, "inequality QP", condition)
        if not selected:
            continue
        label = condition_label(condition, selected[0].get("matrix_condition"))
        gamma = column(selected, "gamma")
        failed = ~np.array([row["converged"] for row in selected], dtype=bool)
        for axis, field in zip(
            axes[0], ("outer_iterations", "inner_iterations"), strict=True
        ):
            values = column(selected, field)
            (line,) = axis.loglog(
                gamma, values, "o-", markersize=4, linewidth=1.6, label=label
            )
            _mark_failures(axis, line, gamma, values, failed)
    axes[0, 0].set_ylabel("outer iterations")
    axes[0, 1].set_ylabel("total inner CG iterations")
    for axis in axes[0]:
        axis.set_xlabel(XLABEL)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(title="constraint matrix", fontsize=8)

    if histories:
        traced = set()
        for condition in CONDITIONS:
            trace = sorted(
                select(histories, matrix=condition),
                key=lambda row: row["iteration"],
            )
            if not trace:
                continue
            traced.add(trace[0]["gamma"])
            accuracy = column(trace, "accuracy")
            axes[1, 0].semilogy(
                column(trace, "iteration"), accuracy, label=condition_label(condition)
            )
            axes[1, 1].semilogy(
                column(trace, "cumulative_inner_iterations"),
                accuracy,
                label=condition_label(condition),
            )
        axes[1, 0].set_xlabel("outer iteration")
        axes[1, 1].set_xlabel("cumulative inner CG iterations")
        title = (
            rf"$\gamma={traced.pop():g}$" if len(traced) == 1 else r"traced $\gamma$"
        )
        for axis in axes[1]:
            axis.set_ylabel("accuracy")
            axis.grid(True, which="both", alpha=0.3)
            axis.legend(title=title, fontsize=8)

    fig.suptitle(r"Inequality QP ($n=500$, $m=100$): iteration counts vs. $\gamma$")
    return save_figure(fig, output)


def make(results_dir: Path | None = None) -> list[Path]:
    output = study_dir(STUDY, results_dir)
    written = []
    try:
        rows = load(STUDY, results_dir=results_dir)
    except MissingResults:
        rows = []
    else:
        written.append(plot_iteration_counts(rows, output / "iteration_counts.pdf"))

    # The focused inequality run has its own table and traces; fall back to
    # the full sweep when only that has been run.
    try:
        focused = load(STUDY, INEQUALITY_RESULTS, results_dir)
    except MissingResults:
        focused = rows
    try:
        histories = load_histories(STUDY, results_dir)
    except MissingResults:
        histories = None
    if select(focused, problem="inequality QP"):
        written.append(
            plot_inequality(
                focused, output / "inequality_qp_inner_iterations.pdf", histories
            )
        )
    if not written:
        raise MissingResults(missing(output / RESULTS))
    return written
