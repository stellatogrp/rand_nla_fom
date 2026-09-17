"""Shared plotting and result helpers for the experiment scripts."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Callable, Iterable, Protocol

os.environ.setdefault("MPLCONFIGDIR", "/tmp/randomized-sketch-descent-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from randomized_sketch_descent import DRSResult


class Labeled(Protocol):
    label: str


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write homogeneous result rows to CSV."""
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_objective_error(values: np.ndarray, optimum: float) -> np.ndarray:
    """Return absolute objective error scaled by ``max(1, |optimum|)``."""
    return np.abs(values - optimum) / max(1.0, abs(optimum))


def accuracy_history(optimum: float, result: DRSResult) -> np.ndarray:
    """Combine normalized objective error and equality feasibility."""
    objective_error = normalized_objective_error(
        result.objective_values, optimum
    )
    return objective_error + result.feasibility_norms


def save_figure(fig, output: Path) -> None:
    """Lay out, save, and close a Matplotlib figure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_runs(
    runs: Iterable[tuple[Labeled, DRSResult]],
    merit: Callable[[DRSResult], np.ndarray],
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for setting, result in runs:
        values = merit(result)
        axes[0].semilogy(np.arange(1, result.iterations + 1), values,
                         label=setting.label)
        axes[1].semilogy(np.cumsum(result.inner_iterations), values,
                         label=setting.label)
    axes[0].set_xlabel("outer iteration")
    axes[1].set_xlabel("cumulative inner iterations")
    axes[0].set_ylabel(ylabel)
    for axis in axes:
        axis.grid(True, alpha=.3)
        axis.legend(fontsize=7)
    fig.suptitle(title)
    save_figure(fig, output)
