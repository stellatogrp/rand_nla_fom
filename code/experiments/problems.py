"""Shared problem primitives for reproducible experiments.

Every reference optimum in the experiments is computed by CVXPY with the
interior-point solver Clarabel through :func:`solve_reference`; no reference
algorithm is implemented here.  The proximal maps the method needs are the
only problem-specific code.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Prox = Callable[[Vector, float], Vector]
# Coefficients (c1, c2, c3, c4) of the hybrid penalty
#     c1 ||D x - y||^2 + c2 ||D x - y||_1 + c3 ||x||_1 + c4 ||x||_2.
Coefficients = tuple[float, float, float, float]

REFERENCE_SOLVER = cp.CLARABEL
# Clarabel is asked for the tightest tolerance first; when it reports it as
# unattainable (the second-order cone of the hybrid penalty ends with an
# "almost solved" status below 1e-8) the solve is repeated at the next one.
# The loosest is still ten times below the accuracy the experiments target.
REFERENCE_TOLERANCES = (1e-12, 1e-10, 1e-8)


def solve_reference(problem: cp.Problem) -> float:
    """Solve a CVXPY problem to high accuracy and return its optimal value."""
    for tolerance in REFERENCE_TOLERANCES:
        with warnings.catch_warnings():
            # An inaccurate status at the tight tolerances is handled below.
            warnings.simplefilter("ignore", UserWarning)
            problem.solve(
                solver=REFERENCE_SOLVER,
                tol_gap_abs=tolerance,
                tol_gap_rel=tolerance,
                tol_feas=tolerance,
                max_iter=500,
            )
        if problem.status == cp.OPTIMAL:
            return float(problem.value)
    raise RuntimeError(f"reference solve ended with status {problem.status!r}")


def _quadratic(x: cp.Variable, hessian: Vector) -> cp.Expression:
    """Return ``x^T Q x`` for a full or, when one-dimensional, diagonal ``Q``."""
    q = np.asarray(hessian, dtype=float)
    if q.ndim == 1:
        return cp.sum(cp.multiply(q, cp.square(x)))
    return cp.quad_form(x, cp.psd_wrap(q))


def equality_qp_optimum(
    hessian: Vector, linear_term: Vector, matrix: Vector, rhs: Vector
) -> float:
    """Return the optimum of ``x'Qx/2 + c'x`` subject to ``A x = b``."""
    x = cp.Variable(matrix.shape[1])
    objective = 0.5 * _quadratic(x, hessian) + np.asarray(linear_term) @ x
    return solve_reference(cp.Problem(cp.Minimize(objective), [matrix @ x == rhs]))


def lp_optimum(cost: Vector, matrix: Vector, rhs: Vector) -> float:
    """Return the optimum of ``c'x`` subject to ``A x = b, x >= 0``."""
    x = cp.Variable(matrix.shape[1])
    return solve_reference(
        cp.Problem(cp.Minimize(np.asarray(cost) @ x), [matrix @ x == rhs, x >= 0])
    )


def inequality_qp_optimum(hessian: Vector, matrix: Vector, rhs: Vector) -> float:
    """Return the optimum of ``x'Qx`` subject to ``A x <= b``."""
    x = cp.Variable(matrix.shape[1])
    return solve_reference(
        cp.Problem(cp.Minimize(_quadratic(x, hessian)), [matrix @ x <= rhs])
    )


def hybrid_optimum(
    design: Vector, response: Vector, coefficients: Coefficients
) -> tuple[float, Vector]:
    """Return the optimum and a minimizer of the hybrid penalty.

    The plain LASSO is the case ``(1/2, 0, tau, 0)``.
    """
    c1, c2, c3, c4 = coefficients
    x = cp.Variable(design.shape[1])
    residual = design @ x - response
    objective = (
        c1 * cp.sum_squares(residual)
        + c2 * cp.norm1(residual)
        + c3 * cp.norm1(x)
        + c4 * cp.norm2(x)
    )
    value = solve_reference(cp.Problem(cp.Minimize(objective)))
    return value, np.asarray(x.value, dtype=float)


def nonnegative_linear_prox(cost: Vector) -> Prox:
    """Return the proximal map of a linear objective over the nonnegative orthant."""

    def prox(vector: Vector, step: float) -> Vector:
        return np.maximum(vector - step * cost, 0.0)

    return prox


def slack_quadratic_prox(hessian: Vector) -> Prox:
    """Return the proximal map of ``x'Qx + indicator(s >= 0)``."""
    variables = hessian.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)

    def prox(vector: Vector, step: float) -> Vector:
        primal = eigenvectors @ (
            (eigenvectors.T @ vector[:variables]) / (1 + 2 * step * eigenvalues)
        )
        return np.r_[primal, np.maximum(vector[variables:], 0.0)]

    return prox


def _soft(vector: Vector, level: float) -> Vector:
    return np.sign(vector) * np.maximum(np.abs(vector) - level, 0.0)


def hybrid_objective(
    design: Vector, response: Vector, coefficients: Coefficients
) -> Callable[[Vector], float]:
    """Return the hybrid objective as a function of ``x`` alone."""
    c1, c2, c3, c4 = coefficients

    def objective(primal: Vector) -> float:
        residual = design @ primal - response
        return float(
            c1 * residual @ residual
            + c2 * np.abs(residual).sum()
            + c3 * np.abs(primal).sum()
            + c4 * np.linalg.norm(primal)
        )

    return objective


def hybrid_split_objective(
    features: int, coefficients: Coefficients
) -> Callable[[Vector], float]:
    """Return ``f(x, z)`` of the split form, with ``z`` standing for ``D x - y``."""
    c1, c2, c3, c4 = coefficients

    def objective(vector: Vector) -> float:
        primal, residual = vector[:features], vector[features:]
        return float(
            c3 * np.abs(primal).sum()
            + c4 * np.linalg.norm(primal)
            + c1 * residual @ residual
            + c2 * np.abs(residual).sum()
        )

    return objective


def hybrid_prox(features: int, coefficients: Coefficients) -> Prox:
    """Return the proximal map of the split hybrid objective on ``(x, z)``.

    Both blocks have closed forms: on ``x`` the map of ``c3||.||_1 + c4||.||_2``
    is a soft threshold followed by a block shrinkage, and on ``z`` the map of
    ``c1||.||^2 + c2||.||_1`` is a soft threshold followed by a scaling.
    """
    c1, c2, c3, c4 = coefficients

    def prox(vector: Vector, step: float) -> Vector:
        primal = _soft(vector[:features], step * c3)
        norm = float(np.linalg.norm(primal))
        if norm > 0:
            primal *= max(1.0 - step * c4 / norm, 0.0)
        residual = _soft(vector[features:], step * c2) / (1.0 + 2.0 * step * c1)
        return np.r_[primal, residual]

    return prox
