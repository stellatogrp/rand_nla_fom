"""Figures for the relaxation sweep.

Each ``theta`` is repeated over several trials, so the curves show the median
and whiskers spanning the observed range.  Only trials that reached the target
contribute: a run that never got there has no cost to report, and its wall
time measures a failure rather than a solve.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments._paths import study_dir

from ._data import load, select
from ._style import condition_label, matrix_label, plt, save_figure

STUDY = "theta_sweep"
KINDS = ("equality QP", "LP", "inequality QP")
CONDITIONS = ("moderately-conditioned", "mildly-ill-conditioned")
POLICIES = ("fixed", "frontier")
STYLES = {"fixed": "-", "frontier": "--"}
QUANTITIES = (
    ("outer_iterations", "outer iterations to target"),
    ("inner_iterations", "row projections to target"),
    ("runtime_seconds", "solver time (seconds)"),
)


def aggregate(rows: list[dict]) -> dict:
    """Reduce per-trial rows to a median and an observed range per theta."""
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[float(row["theta"])].append(row)

    thetas = np.array(sorted(grouped), dtype=float)
    summary = {name: {"median": [], "low": [], "high": []} for name, _ in QUANTITIES}
    incomplete = []
    for theta in thetas:
        trials = grouped[theta]
        reached = [row for row in trials if row["reached_accuracy"]]
        incomplete.append(len(reached) < len(trials))
        for name, _ in QUANTITIES:
            values = np.array([row[name] for row in reached], dtype=float)
            values = values[np.isfinite(values)]
            for statistic, function in (
                ("median", np.median),
                ("low", np.min),
                ("high", np.max),
            ):
                summary[name][statistic].append(
                    float(function(values)) if values.size else np.nan
                )
    return {
        "theta": thetas,
        "incomplete": np.array(incomplete, dtype=bool),
        "trials": max((len(group) for group in grouped.values()), default=0),
        **{
            name: {
                key: np.array(value, dtype=float) for key, value in statistics.items()
            }
            for name, statistics in summary.items()
        },
    }


def plot_work(rows: list[dict], output: Path) -> Path:
    """Plot outer work, inner projections, and runtime against ``theta``."""
    kinds = [kind for kind in KINDS if select(rows, problem=kind)]
    fig, axes = plt.subplots(
        len(QUANTITIES),
        len(kinds),
        figsize=(4.2 * len(kinds), 9.5),
        sharex=True,
        squeeze=False,
    )
    colors = {condition: f"C{index}" for index, condition in enumerate(CONDITIONS)}
    trials = 0
    for column_index, kind in enumerate(kinds):
        # NaN entries for thetas that never reached the target confuse the log
        # autoscale, so track the finite extent and set the limits explicitly.
        extents: dict[int, list[float]] = defaultdict(list)
        for condition in CONDITIONS:
            for policy in POLICIES:
                selected = select(rows, problem=kind, matrix=condition, policy=policy)
                if not selected:
                    continue
                summary = aggregate(selected)
                trials = max(trials, summary["trials"])
                label = (
                    f"{condition_label(condition, selected[0].get('matrix_condition'))}"
                    rf", {policy} $\sigma$"
                )
                for index, (name, _) in enumerate(QUANTITIES):
                    axis = axes[index, column_index]
                    median = summary[name]["median"]
                    lower = np.clip(median - summary[name]["low"], 0, None)
                    upper = np.clip(summary[name]["high"] - median, 0, None)
                    axis.errorbar(
                        summary["theta"],
                        median,
                        yerr=np.vstack((lower, upper)),
                        fmt=STYLES[policy],
                        marker="o",
                        markersize=3.5,
                        linewidth=1.4,
                        elinewidth=1.0,
                        capsize=2.5,
                        color=colors[condition],
                        label=label,
                    )
                    axis.set_yscale("log")
                    finite = np.concatenate(
                        (
                            summary[name]["low"][np.isfinite(summary[name]["low"])],
                            summary[name]["high"][np.isfinite(summary[name]["high"])],
                        )
                    )
                    if finite.size:
                        extents[index].extend((finite.min(), finite.max()))
                    # A theta where some but not all trials reached the target
                    # still plots a median; flag it so the thinner sample is
                    # visible.
                    partial = summary["incomplete"] & np.isfinite(median)
                    if partial.any():
                        axis.scatter(
                            summary["theta"][partial],
                            median[partial],
                            marker="x",
                            s=45,
                            color=colors[condition],
                            zorder=4,
                        )

        family = select(rows, problem=kind)
        axes[0, column_index].set_title(f"{kind}\n{matrix_label(family)}")
        axes[-1, column_index].set_xlabel(r"relaxation $\theta$")
        for index in range(len(QUANTITIES)):
            axis = axes[index, column_index]
            axis.grid(True, which="both", alpha=0.3)
            if extents[index]:
                axis.set_ylim(min(extents[index]) / 1.6, max(extents[index]) * 1.6)

    for index, (_, ylabel) in enumerate(QUANTITIES):
        axes[index, 0].set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=True):
        seen.setdefault(label, handle)
    fig.legend(
        seen.values(),
        seen.keys(),
        loc="lower center",
        ncol=len(seen),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    cap = int(max(row["max_outer_iterations"] for row in rows))
    fig.suptitle(
        "Target: normalized objective error + feasibility "
        rf"$\leq {rows[0]['target_accuracy']:g}$; "
        rf"$\gamma_x=\gamma_\lambda={rows[0]['gamma_x']:g}$; "
        "randomized Kaczmarz inner solver; "
        rf"median of {trials} trials, whiskers span the observed range; "
        rf"$\times$ marks $\theta$ where some trial missed the target "
        rf"within {cap} outer iterations"
    )
    return save_figure(fig, output)


def make(results_dir: Path | None = None) -> list[Path]:
    rows = load(STUDY, results_dir=results_dir)
    output = study_dir(STUDY, results_dir)
    return [plot_work(rows, output / "iteration_counts.pdf")]
