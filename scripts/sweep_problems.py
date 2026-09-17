"""Problem families shared by scalar-parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from randomized_sketch_descent import quadratic_prox

from .problems import (
    Prox,
    Vector,
    equality_qp_solution,
    inequality_qp_optimum,
    nonnegative_linear_prox,
    slack_quadratic_prox,
    standard_form_lp_optimum,
)


Condition = Literal["moderately-conditioned", "ill-conditioned"]
CONDITIONS: tuple[Condition, ...] = (
    "moderately-conditioned",
    "ill-conditioned",
)
SINGULAR_LIMITS = {
    "moderately-conditioned": (1.0, 0.1),
    "ill-conditioned": (10.0, 0.1),
}


@dataclass(frozen=True)
class SweepProblem:
    kind: str
    condition: Condition
    dimension: int
    matrix: np.ndarray
    rhs: np.ndarray
    prox: Prox
    objective: Callable[[Vector], float]
    optimum: float
    max_iterations: int
    solver_tolerance: float
    matrix_condition: float


def matrix_with_spectrum(
    rng: np.random.Generator,
    constraints: int,
    variables: int,
    condition: Condition,
    rowspace: np.ndarray | None = None,
) -> np.ndarray:
    """Return a full-row-rank matrix with controlled singular values."""
    source = (
        rng.standard_normal((constraints, variables))
        if rowspace is None
        else rowspace
    )
    _, _, right_vectors = np.linalg.svd(source, full_matrices=False)
    high, low = SINGULAR_LIMITS[condition]
    singular_values = np.geomspace(high, low, constraints)
    return singular_values[:, None] * right_vectors


def make_equality_qp(condition: Condition, seed: int = 41) -> SweepProblem:
    rng = np.random.default_rng(seed)
    constraints, variables = 16, 80
    matrix = matrix_with_spectrum(rng, constraints, variables, condition)
    factor = rng.standard_normal((variables, variables)) / np.sqrt(variables)
    hessian = factor.T @ factor + 0.3 * np.eye(variables)
    linear_term = rng.standard_normal(variables) / np.sqrt(variables)
    rhs = matrix @ rng.standard_normal(variables)
    solution = equality_qp_solution(hessian, linear_term, matrix, rhs)

    def objective(vector: Vector) -> float:
        return float(0.5 * vector @ hessian @ vector + linear_term @ vector)

    return SweepProblem(
        "equality QP",
        condition,
        variables,
        matrix,
        rhs,
        quadratic_prox(hessian, linear_term),
        objective,
        objective(solution),
        8_000,
        2e-7,
        np.linalg.cond(matrix),
    )


def make_lp(condition: Condition, seed: int = 42) -> SweepProblem:
    rng = np.random.default_rng(seed)
    constraints, variables = 8, 40
    rowspace = np.vstack(
        (
            rng.standard_normal((constraints - 1, variables)),
            np.ones(variables),
        )
    )
    matrix = matrix_with_spectrum(
        rng, constraints, variables, condition, rowspace
    )
    feasible = rng.uniform(0.2, 1.0, variables)
    feasible /= feasible.sum()
    rhs = matrix @ feasible
    cost = rng.standard_normal(variables)

    def objective(vector: Vector) -> float:
        return float(cost @ vector)

    return SweepProblem(
        "LP",
        condition,
        variables,
        matrix,
        rhs,
        nonnegative_linear_prox(cost),
        objective,
        standard_form_lp_optimum(cost, matrix, rhs),
        20_000,
        1e-6,
        np.linalg.cond(matrix),
    )


def make_inequality_qp(condition: Condition, seed: int = 43) -> SweepProblem:
    rng = np.random.default_rng(seed)
    constraints, variables = 100, 500
    matrix = matrix_with_spectrum(rng, constraints, variables, condition)
    factor = rng.standard_normal((variables, variables)) / np.sqrt(variables)
    hessian = factor.T @ factor + 0.3 * np.eye(variables)
    feasible = rng.standard_normal(variables)
    rhs = matrix @ feasible + rng.uniform(0.05, 0.2, constraints)
    slack_matrix = np.hstack((matrix, np.eye(constraints)))

    def objective(vector: Vector) -> float:
        primal = vector[:variables]
        return float(primal @ hessian @ primal)

    return SweepProblem(
        "inequality QP",
        condition,
        variables,
        slack_matrix,
        rhs,
        slack_quadratic_prox(hessian),
        objective,
        inequality_qp_optimum(hessian, matrix, rhs),
        8_000,
        2e-7,
        np.linalg.cond(matrix),
    )


def _large_matrix(
    rng: np.random.Generator,
    constraints: int,
    variables: int,
    condition: Condition,
) -> np.ndarray:
    """Construct a large controlled-spectrum matrix without a full SVD."""
    basis, _ = np.linalg.qr(
        rng.standard_normal((variables, constraints)), mode="reduced"
    )
    high, low = SINGULAR_LIMITS[condition]
    return np.geomspace(high, low, constraints)[:, None] * basis.T


def _large_lp_matrix(
    rng: np.random.Generator,
    constraints: int,
    variables: int,
    condition: Condition,
) -> np.ndarray:
    """Construct orthonormal rows with a well-conditioned leading basis."""
    trailing = rng.standard_normal((constraints, variables - constraints))
    trailing *= 0.05 / np.sqrt(variables - constraints)
    source = np.hstack((np.eye(constraints), trailing))
    eigenvalues, eigenvectors = np.linalg.eigh(source @ source.T)
    inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    high, low = SINGULAR_LIMITS[condition]
    singular_values = np.geomspace(high, low, constraints)
    return singular_values[:, None] * (inverse_root @ source)


def _diagonal_quadratic_prox(
    diagonal: Vector,
    linear_term: Vector,
) -> Prox:
    def prox(vector: Vector, step: float) -> Vector:
        return (vector - step * linear_term) / (1 + step * diagonal)

    return prox


def make_large_equality_qp(
    condition: Condition,
    constraints: int,
    variables: int,
    seed: int = 41,
) -> SweepProblem:
    """Build a scalable equality QP with an exact KKT solution."""
    rng = np.random.default_rng(seed)
    matrix = _large_matrix(rng, constraints, variables, condition)
    diagonal = np.geomspace(0.3, 3.0, variables)
    solution = rng.standard_normal(variables) / np.sqrt(variables)
    multiplier = rng.standard_normal(constraints) / np.sqrt(constraints)
    linear_term = -diagonal * solution - matrix.T @ multiplier
    rhs = matrix @ solution

    def objective(vector: Vector) -> float:
        return float(0.5 * (diagonal * vector) @ vector + linear_term @ vector)

    high, low = SINGULAR_LIMITS[condition]
    return SweepProblem(
        "equality QP",
        condition,
        variables,
        matrix,
        rhs,
        _diagonal_quadratic_prox(diagonal, linear_term),
        objective,
        objective(solution),
        200_000,
        2e-7,
        high / low,
    )


def make_large_lp(
    condition: Condition,
    constraints: int,
    variables: int,
    seed: int = 42,
) -> SweepProblem:
    """Build a scalable LP with an exact primal-dual certificate."""
    rng = np.random.default_rng(seed)
    matrix = _large_lp_matrix(rng, constraints, variables, condition)
    solution = np.zeros(variables)
    solution[:constraints] = rng.uniform(0.2, 1.0, constraints)
    solution /= solution.sum()
    multiplier = rng.standard_normal(constraints) / np.sqrt(constraints)
    reduced_cost = np.zeros(variables)
    reduced_cost[constraints:] = rng.uniform(0.2, 1.0, variables - constraints)
    cost = -matrix.T @ multiplier + reduced_cost
    rhs = matrix @ solution

    def objective(vector: Vector) -> float:
        return float(cost @ vector)

    high, low = SINGULAR_LIMITS[condition]
    return SweepProblem(
        "LP",
        condition,
        variables,
        matrix,
        rhs,
        nonnegative_linear_prox(cost),
        objective,
        objective(solution),
        200_000,
        1e-6,
        high / low,
    )


def make_large_inequality_qp(
    condition: Condition,
    constraints: int,
    variables: int,
    seed: int = 43,
) -> SweepProblem:
    """Build a scalable inequality QP with an exact KKT solution."""
    rng = np.random.default_rng(seed)
    matrix = _large_matrix(rng, constraints, variables, condition)
    diagonal = np.geomspace(0.3, 3.0, variables)
    multiplier = rng.uniform(0.2, 1.0, constraints) / np.sqrt(constraints)
    solution = -(matrix.T @ multiplier) / (2 * diagonal)
    rhs = matrix @ solution
    slack_matrix = np.hstack((matrix, np.eye(constraints)))

    def prox(vector: Vector, step: float) -> Vector:
        primal = vector[:variables] / (1 + 2 * step * diagonal)
        return np.r_[primal, np.maximum(vector[variables:], 0.0)]

    def objective(vector: Vector) -> float:
        primal = vector[:variables]
        return float((diagonal * primal) @ primal)

    high, low = SINGULAR_LIMITS[condition]
    return SweepProblem(
        "inequality QP",
        condition,
        variables,
        slack_matrix,
        rhs,
        prox,
        objective,
        objective(solution),
        200_000,
        2e-7,
        high / low,
    )
