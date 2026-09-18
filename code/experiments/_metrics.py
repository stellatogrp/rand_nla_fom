"""Accuracy measures shared by every study, and the traces they produce.

The merit function used throughout is the normalized objective error plus a
feasibility residual.  Studies that report work at a fixed accuracy locate the
first outer iterate meeting the target with :func:`target_index`; studies that
show convergence curves write the whole trace with :func:`history_rows` so
that plotting can read it back from CSV.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rand_nla_fom import DRSResult

ACCURACY_FLOOR = 1e-16


def normalized_objective_error(
    values: NDArray[np.float64], optimum: float
) -> NDArray[np.float64]:
    """Return absolute objective error scaled by ``max(1, |optimum|)``."""
    return np.abs(values - optimum) / max(1.0, abs(optimum))


def accuracy_history(
    optimum: float,
    result: DRSResult,
    feasibility: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Combine normalized objective error with a feasibility residual."""
    if feasibility is None:
        feasibility = result.feasibility_norms
    error = normalized_objective_error(result.objective_values, optimum)
    return error + feasibility


def target_index(accuracy: NDArray[np.float64], target: float) -> int | None:
    """Return the first iterate index attaining ``target``, if any."""
    reached = np.flatnonzero(accuracy <= target)
    return int(reached[0]) if reached.size else None


def history_rows(
    keys: dict,
    result: DRSResult,
    optimum: float,
    feasibility: NDArray[np.float64] | None = None,
) -> list[dict]:
    """Expand one run into per-outer-iteration rows for ``histories.csv``.

    ``keys`` identifies the run and is repeated on every row; it must match
    the identifying columns of the study's ``results.csv`` so the two tables
    can be joined when plotting.
    """
    if feasibility is None:
        feasibility = result.feasibility_norms
    error = normalized_objective_error(result.objective_values, optimum)
    accuracy = np.maximum(error + feasibility, ACCURACY_FLOOR)
    cumulative = np.cumsum(result.inner_iterations)
    return [
        dict(
            **keys,
            iteration=index + 1,
            inner_iterations=int(result.inner_iterations[index]),
            cumulative_inner_iterations=int(cumulative[index]),
            objective_error=float(error[index]),
            feasibility=float(feasibility[index]),
            accuracy=float(accuracy[index]),
            theta_k=float(result.theta_history[index]),
            sigma_k=float(result.sigma_history[index]),
        )
        for index in range(result.iterations)
    ]
