"""CSV reading and writing shared by the drivers, the aggregator, and the plots.

Every table is a plain CSV so that it can be inspected without this package.
``results.csv`` holds one row per run; ``histories.csv`` holds one row per run
and outer iteration and is read column-wise, because it is large.
"""

from __future__ import annotations

import csv
import itertools
from array import array
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def _coerce(value: str) -> object:
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


def append_rows(path: Path, rows: Sequence[dict]) -> None:
    """Append homogeneous rows to a CSV, writing the header when it is new."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as stream:
            header = next(csv.reader(stream))
        if header != fieldnames:
            raise ValueError(f"{path} has columns {header}, rows have {fieldnames}")
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    """Write rows to a fresh CSV, replacing whatever was there."""
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    """Read a small table into typed dictionaries, or ``[]`` if absent."""
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return [
            {name: _coerce(value) for name, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def read_columns(path: Path) -> dict[str, np.ndarray]:
    """Read a large table column-wise: float arrays, or object arrays of strings.

    The type of each column is sniffed from the first data row, and numeric
    columns are accumulated in compact ``array('d')`` buffers, so a table
    with a million rows reads in a few seconds and a few hundred megabytes.
    """
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        first = next(reader, None)
        if first is None:
            return {name: np.array([], dtype=float) for name in header}
        numeric = [_numeric(value) for value in first]
        buffers: list = [array("d") if flag else [] for flag in numeric]
        for row in itertools.chain([first], reader):
            for buffer, flag, value in zip(buffers, numeric, row, strict=True):
                if flag:
                    buffer.append(float(value) if value != "" else float("nan"))
                else:
                    buffer.append(value)
    table: dict[str, np.ndarray] = {}
    for name, flag, buffer in zip(header, numeric, buffers, strict=True):
        if flag:
            table[name] = np.frombuffer(buffer, dtype=float).copy()
        else:
            table[name] = np.array(buffer, dtype=object)
    return table


def select(rows: list[dict], **fields: object) -> list[dict]:
    """Return the rows whose fields equal every given value."""
    return [row for row in rows if all(row.get(k) == v for k, v in fields.items())]
