"""Figures for the inner-solver comparison over random problem instances."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments._paths import study_dir

from ._data import column, load, select
from ._style import (
    CONDITION_LABELS,
    SOLVER_COLORS,
    SOLVER_LABELS,
    matrix_label,
    plt,
    save_figure,
)

STUDY = "solver_sweep"
RANDOMIZED_STUDY = "solver_sweep_randomized_spectrum"
KINDS = ("equality QP", "LP", "inequality QP")
CONDITIONS = ("moderately-conditioned", "ill-conditioned")
SOLVERS = (
    "schur_cg",
    "schur_block_kaczmarz",
    "schur_randomized_kaczmarz",
    "coupled_gmres",
)
FIELDS = (
    ("outer_iterations", "outer iterations to target", False),
    ("inner_iterations", "inner iterations to target", True),
    ("runtime_seconds", "solver time (seconds)", True),
)


def _reached(rows: list[dict], kind: str, condition: str, solver: str) -> list[dict]:
    selected = select(rows, problem=kind, matrix=condition, inner_solver=solver)
    return [row for row in selected if row["reached_accuracy"]]


def _spectrum_label(rows: list[dict]) -> str:
    """Describe how the singular values were drawn across instances."""
    flags = {bool(row.get("randomize_spectrum")) for row in rows}
    if flags == {True}:
        return "randomized interior spectrum"
    if flags == {False}:
        return "fixed geometric spectrum"
    return "mixed spectrum sampling"


def _caption(rows: list[dict], kinds: list[str]) -> str:
    instances = max(
        len(select(rows, problem=kind, matrix=condition, inner_solver=solver))
        for kind in kinds
        for condition in CONDITIONS
        for solver in SOLVERS
    )
    return (
        f"Inner solver comparison over {instances} random instances per family "
        f"({_spectrum_label(rows)}); "
        rf"$\sigma={rows[0]['sigma']:g}$, $\theta={rows[0]['theta']:g}$, "
        rf"$\gamma_x=\gamma_\lambda={rows[0]['gamma_x']:g}$; "
        "target: normalized objective error + feasibility "
        rf"$\leq {rows[0]['target_accuracy']:g}$"
    )


def _bar_panel(axis, rows: list[dict], kind: str, field: str, solvers) -> None:
    """Draw one grouped bar panel with mean +/- one standard deviation."""
    width = 0.8 / len(solvers)
    centers = np.arange(len(CONDITIONS))
    drawn_means: list[float] = []
    failures: list[tuple[float, str]] = []
    for offset, solver in enumerate(solvers):
        means, errors, notes = [], [], []
        for condition in CONDITIONS:
            values = column(_reached(rows, kind, condition, solver), field)
            values = values[np.isfinite(values)]
            means.append(values.mean() if values.size else np.nan)
            errors.append(float(values.std(ddof=1)) if values.size > 1 else 0.0)
            total = len(
                select(rows, problem=kind, matrix=condition, inner_solver=solver)
            )
            notes.append("" if values.size == total else f"{values.size}/{total}")
        positions = centers + (offset - (len(solvers) - 1) / 2) * width
        means = np.array(means, dtype=float)
        errors = np.array(errors, dtype=float)
        # Clip the lower whisker so a wide spread stays drawable on a log axis.
        lower = np.minimum(errors, np.nan_to_num(means) * 0.95)
        axis.bar(
            positions,
            np.nan_to_num(means),
            width * 0.9,
            yerr=np.vstack((lower, errors)),
            color=SOLVER_COLORS.get(solver, "#808080"),
            label=SOLVER_LABELS.get(solver, solver),
            error_kw=dict(elinewidth=1.0, capsize=2.5, ecolor="#3d3d3a"),
        )
        for position, mean, spread, note in zip(
            positions, means, errors, notes, strict=True
        ):
            if np.isnan(mean):
                failures.append((position, SOLVER_COLORS.get(solver, "#808080")))
                continue
            drawn_means.append(mean)
            if note:
                # Say how many instances the bar averages when some of them
                # never reached the target.
                axis.annotate(
                    note,
                    (position, mean + spread),
                    textcoords="offset points",
                    xytext=(0, 4),
                    fontsize=5.5,
                    ha="center",
                    va="bottom",
                    color=SOLVER_COLORS.get(solver, "#808080"),
                    fontweight="bold",
                )
    axis.set_xticks(centers)
    axis.set_xticklabels([CONDITION_LABELS.get(name, name) for name in CONDITIONS])
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.set_axisbelow(True)

    positive = [value for value in drawn_means if value > 0]
    if axis.get_yscale() == "log" and positive:
        # Failed solvers plot as a zero-height bar; keep the axis on the
        # decades that carry data instead of running down to the bar base.
        axis.set_ylim(bottom=min(positive) / 5)
    low, high = axis.get_ylim()
    middle = np.sqrt(low * high) if axis.get_yscale() == "log" else (low + high) / 2
    for position, color in failures:
        axis.text(
            position,
            middle,
            "no instance\nreached target",
            rotation=90,
            fontsize=5.5,
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
        )


def plot_comparison(rows: list[dict], output: Path) -> Path:
    """Plot outer iterations, inner iterations, and time for every family."""
    kinds = [kind for kind in KINDS if select(rows, problem=kind)]
    solvers = [solver for solver in SOLVERS if select(rows, inner_solver=solver)]
    fig, axes = plt.subplots(
        len(FIELDS), len(kinds), figsize=(4.2 * len(kinds), 10), squeeze=False
    )
    for column_index, kind in enumerate(kinds):
        for row_index, (field, ylabel, logarithmic) in enumerate(FIELDS):
            axis = axes[row_index, column_index]
            if logarithmic:
                axis.set_yscale("log")
            _bar_panel(axis, rows, kind, field, solvers)
            if column_index == 0:
                axis.set_ylabel(ylabel)
        axes[0, column_index].set_title(
            f"{kind}\n{matrix_label(select(rows, problem=kind))}"
        )
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        _caption(rows, kinds)
        + "\nbars are means, error bars one standard deviation over instances"
    )
    return save_figure(fig, output)


def plot_distributions(rows: list[dict], output: Path) -> Path:
    """Show the per-instance spread behind the error bars."""
    kinds = [kind for kind in KINDS if select(rows, problem=kind)]
    solvers = [solver for solver in SOLVERS if select(rows, inner_solver=solver)]
    fig, axes = plt.subplots(
        2, len(kinds), figsize=(4.2 * len(kinds), 7), sharex="col", squeeze=False
    )
    for column_index, kind in enumerate(kinds):
        for row_index, field in enumerate(("outer_iterations", "inner_iterations")):
            axis = axes[row_index, column_index]
            if field == "inner_iterations":
                axis.set_yscale("log")
            labels, samples, colors, positions = [], [], [], []
            slots = [
                (condition, solver) for condition in CONDITIONS for solver in solvers
            ]
            for slot, (condition, solver) in enumerate(slots):
                values = column(_reached(rows, kind, condition, solver), field)
                values = values[np.isfinite(values)]
                labels.append(
                    f"{SOLVER_LABELS.get(solver, solver)}\n"
                    f"{CONDITION_LABELS.get(condition, condition)}"
                )
                if not values.size:
                    continue
                samples.append(values)
                colors.append(SOLVER_COLORS.get(solver, "#808080"))
                positions.append(slot)
            if not samples:
                continue
            drawn = axis.boxplot(
                samples,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker=".", markersize=2),
            )
            for patch, color in zip(drawn["boxes"], colors, strict=True):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
                patch.set_edgecolor("#3d3d3a")
            for element in ("medians", "whiskers", "caps"):
                for line in drawn[element]:
                    line.set_color("#3d3d3a")
            axis.set_xticks(np.arange(len(labels)))
            axis.set_xlim(-0.7, len(labels) - 0.3)
            axis.grid(True, axis="y", which="both", alpha=0.3)
            axis.set_axisbelow(True)
            if row_index == 1:
                axis.set_xticklabels(labels, rotation=90, fontsize=6)
        axes[0, column_index].set_title(kind)
    axes[0, 0].set_ylabel("outer iterations to target")
    axes[1, 0].set_ylabel("inner iterations to target")
    fig.suptitle(_caption(rows, kinds))
    return save_figure(fig, output)


def make(
    results_dir: Path | None = None, randomize_spectrum: bool = False
) -> list[Path]:
    study = RANDOMIZED_STUDY if randomize_spectrum else STUDY
    rows = load(study, results_dir=results_dir)
    output = study_dir(study, results_dir)
    return [
        plot_comparison(rows, output / "solver_comparison.pdf"),
        plot_distributions(rows, output / "solver_distributions.pdf"),
    ]


def make_randomized(results_dir: Path | None = None) -> list[Path]:
    return make(results_dir, randomize_spectrum=True)
