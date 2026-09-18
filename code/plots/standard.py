"""Figures for the standard QP and LP comparisons."""

from __future__ import annotations

from pathlib import Path

from experiments._paths import study_dir

from ._data import group, load_histories, select
from ._style import plot_convergence, plt, save_figure

STUDY = "standard"
YLABEL = "relative objective error + feasibility"
PARAMETER_GROUPS = ("M", "sigma", "theta")


def plot_parameters(histories: list[dict], problem: str, output: Path) -> Path:
    """One panel per parameter group, accuracy against outer iteration."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, name in zip(axes, PARAMETER_GROUPS, strict=True):
        rows = select(histories, problem=problem, group=name)
        for (label,), trace in group(rows, "label").items():
            trace = sorted(trace, key=lambda row: row["iteration"])
            axis.semilogy(
                [row["iteration"] for row in trace],
                [row["accuracy"] for row in trace],
                label=label,
            )
        axis.set_title(name)
        axis.set_xlabel("outer iteration")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_ylabel(YLABEL)
    fig.suptitle(f"Section 5 {problem}: M, tolerance, and relaxation")
    return save_figure(fig, output)


def plot_solvers(histories: list[dict], problem: str, output: Path) -> Path:
    """Accuracy against outer and cumulative inner iterations, per solver."""
    rows = select(histories, problem=problem, group="solver")
    traces = [
        (label, sorted(trace, key=lambda row: row["iteration"]))
        for (label,), trace in group(rows, "label").items()
    ]
    return plot_convergence(
        traces,
        f"Section 5 {problem}: linear-system methods",
        YLABEL,
        output,
    )


def make(results_dir: Path | None = None) -> list[Path]:
    histories = load_histories(STUDY, results_dir)
    output = study_dir(STUDY, results_dir)
    written = []
    for problem in sorted({row["problem"] for row in histories}):
        stem = problem.lower()
        written.append(
            plot_parameters(histories, problem, output / f"{stem}_parameters.pdf")
        )
        written.append(plot_solvers(histories, problem, output / f"{stem}_solvers.pdf"))
    return written
