"""Loading the CSV tables the experiment scripts leave behind."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from pathlib import Path

import numpy as np

from experiments._paths import HISTORIES, RESULTS, study_dir
from experiments._results import read_rows, select


class MissingResults(FileNotFoundError):
    """Raised when a figure is requested before its study has been run."""


def missing(path: Path) -> str:
    """Explain that a table has to be produced before it can be plotted."""
    return (
        f"{path} is missing or empty; run the study that writes it first "
        "(`uv run python -m experiments.run_experiments --help` lists them)"
    )


def load(
    study: str, name: str = RESULTS, results_dir: Path | None = None
) -> list[dict]:
    """Read one table of a study, with a message that says how to make it."""
    path = study_dir(study, results_dir) / name
    rows = read_rows(path)
    if not rows:
        raise MissingResults(missing(path))
    return rows


def load_histories(study: str, results_dir: Path | None = None) -> list[dict]:
    """Read a study's per-iteration convergence traces."""
    return load(study, HISTORIES, results_dir)


def group(rows: Iterable[dict], *fields: str) -> dict[tuple, list[dict]]:
    """Bucket rows by the values of ``fields``, preserving first-seen order."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return dict(grouped)


def traces(
    rows: Iterable[dict], *fields: str, label: str = "label"
) -> list[tuple[str, list[dict]]]:
    """Return ``(label, trace)`` pairs ready for ``plot_convergence``.

    Rows are bucketed by ``fields`` and each bucket is sorted by iteration.
    """
    grouped = group(rows, *fields)
    return [
        (str(trace[0][label]), sorted(trace, key=lambda row: row["iteration"]))
        for trace in grouped.values()
    ]


def column(rows: Sequence[dict], name: str) -> np.ndarray:
    """Extract one numeric column as a float array."""
    return np.array([row.get(name, np.nan) for row in rows], dtype=float)


def ordered(rows: Iterable[dict], field: str, order: Sequence[Hashable]) -> list[dict]:
    """Sort rows by a categorical field's declared order, then by the rest."""
    rank = {value: index for index, value in enumerate(order)}
    return sorted(rows, key=lambda row: rank.get(row[field], len(rank)))


__all__ = [
    "MissingResults",
    "missing",
    "column",
    "group",
    "load",
    "load_histories",
    "ordered",
    "select",
    "traces",
]
