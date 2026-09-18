"""Reading and writing the CSV tables every study produces.

Each study directory under ``results/`` holds at most three tables:

``results.csv``
    One row per run: the settings that produced it and the scalars measured
    from it.  Long studies append to this file as they go, so an interrupted
    run can be resumed.
``histories.csv``
    One row per run and outer iteration, for the studies whose figures show
    convergence curves.  Keeping the traces on disk is what lets plotting
    live in :mod:`plots` and read only CSVs.
``summary.csv``
    Aggregates over ``results.csv``, when a study has something to aggregate.

The reader coerces every field back to ``bool``, ``int``, ``float``, or
``str`` so that a table written by one process reads back with the same types
in another.  Empty fields become ``float('nan')``: a missing measurement is
numeric, not a category.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Hashable, Iterable, Sequence
from pathlib import Path


def _coerce(value: str) -> object:
    """Recover the type a CSV field was written with."""
    if value == "":
        return float("nan")
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    """Write homogeneous result rows to CSV, creating the directory."""
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    """Read a result table, or return ``[]`` when it does not exist yet."""
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return [
            {name: _coerce(value) for name, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def merge_rows(
    rows: Iterable[dict],
    replacements: Sequence[dict],
    key: Callable[[dict], Hashable],
    order: Callable[[dict], Hashable] | None = None,
) -> list[dict]:
    """Replace the rows that ``replacements`` supersedes, then reorder."""
    replaced = {key(row) for row in replacements}
    merged = [row for row in rows if key(row) not in replaced]
    merged.extend(replacements)
    return sorted(merged, key=order) if order is not None else merged


def select(rows: Iterable[dict], **fields: object) -> list[dict]:
    """Return the rows matching every ``field=value`` pair."""
    return [
        row
        for row in rows
        if all(row.get(name) == value for name, value in fields.items())
    ]
