"""Shared problem primitives for reproducible experiments."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, linprog, minimize

Vector = NDArray[np.float64]
Prox = Callable[[Vector, float], Vector]


def equality_qp_solution(
    hessian: Vector, linear_term: Vector, matrix: Vector, rhs: Vector
) -> Vector:
    """Solve an equality-constrained convex QP through its KKT system."""
    constraints, variables = matrix.shape
    kkt = np.block(
        [
            [hessian, matrix.T],
            [matrix, np.zeros((constraints, constraints))],
        ]
    )
    return np.linalg.solve(kkt, np.r_[-linear_term, rhs])[:variables]


def standard_form_lp_optimum(cost: Vector, matrix: Vector, rhs: Vector) -> float:
    """Return the optimum of ``min cost @ x`` with ``matrix @ x = rhs, x >= 0``."""
    result = linprog(
        cost,
        A_eq=matrix,
        b_eq=rhs,
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun)


def inequality_qp_optimum(hessian: Vector, matrix: Vector, rhs: Vector) -> float:
    """Independently solve ``min x'Qx`` subject to ``matrix @ x <= rhs``."""
    variables = hessian.shape[0]
    result = minimize(
        lambda x: float(x @ hessian @ x),
        np.zeros(variables),
        jac=lambda x: 2 * hessian @ x,
        hess=lambda _: 2 * hessian,
        constraints=LinearConstraint(matrix, -np.inf, rhs),
        method="trust-constr",
        options={"gtol": 1e-11, "xtol": 1e-12, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun)


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
