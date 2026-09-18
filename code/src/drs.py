"""Algorithm in Section 5 of ``Rand_LA_FOM-4``.

The code solves ``min f(x)`` subject to ``A x = b`` using the block diagonal
metric ``M = diag(I/gamma_x, I/gamma_lambda)``.  Equation (58) can be solved
in four equivalent ways:

``coupled_gmres``
    GMRES on the full nonsymmetric primal--dual system.
``schur_cg``
    CG on its SPD multiplier Schur complement
    ``(I + gamma_x*gamma_lambda*A*A.T) nu = rhs``.
``schur_block_kaczmarz``
    Block Kaczmarz on a cached random partition of the rows of the same
    Schur-complement system.
``schur_randomized_kaczmarz``
    Randomized Kaczmarz on single rows of that system, drawn with
    probability proportional to their squared norms.

All four solvers warm start from the preceding inner solution and stop at the
relaxed relative-error test in Theorem 5,
``||epsilon||_M <= sigma ||s||_M``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._linear_solvers import (
    BlockKaczmarz,
    RandomizedKaczmarz,
    Vector,
    coupled_gmres,
    schur_cg,
)

Prox = Callable[[Vector, float], ArrayLike]


@dataclass(frozen=True)
class DRSResult:
    primal: Vector
    multiplier: Vector
    fixed_point: Vector
    converged: bool
    iterations: int
    objective_values: Vector
    feasibility_norms: Vector
    step_norms_m: Vector
    linear_residual_norms_m: Vector
    relative_error_ratios: Vector
    inner_iterations: NDArray[np.int_]
    primal_iterates: NDArray[np.float64]
    sigma_history: Vector
    theta_history: Vector


class _StepState(NamedTuple):
    primal: Vector
    multiplier: Vector
    candidate: Vector
    primal_step: Vector
    dual_step: Vector
    error_norm: float
    step_norm: float


def _vector(value: ArrayLike, size: int, name: str) -> Vector:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return result.copy()


def quadratic_prox(hessian: ArrayLike, linear_term: ArrayLike) -> Prox:
    """Return the proximal map of ``x.T Q x / 2 + c.T x``."""
    q = np.asarray(hessian, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not np.allclose(q, q.T):
        raise ValueError("hessian must be square and symmetric")
    c = _vector(linear_term, q.shape[0], "linear_term")
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    if eigenvalues[0] < -1e-11:
        raise ValueError("hessian must be positive semidefinite")

    def prox(z: Vector, gamma: float) -> Vector:
        transformed = eigenvectors.T @ (z - gamma * c)
        return eigenvectors @ (transformed / (1.0 + gamma * eigenvalues))

    return prox


def primal_dual_drs(
    prox_f: Prox,
    A: ArrayLike,
    b: ArrayLike,
    *,
    gamma_x: float = 1.0,
    gamma_lambda: float = 1.0,
    sigma: float = 0.2,
    theta: float = 1.0,
    linear_solver: str = "schur_cg",
    objective: Callable[[Vector], float] | None = None,
    x0: ArrayLike | None = None,
    multiplier0: ArrayLike | None = None,
    max_iterations: int = 5_000,
    max_inner_iterations: int | None = None,
    tolerance: float = 1e-8,
    adaptive: bool = False,
    adaptation_interval: int = 20,
    block_size: int | None = None,
    random_seed: int = 0,
    store_iterates: bool = True,
) -> DRSResult:
    """Run relaxed, relative-error preconditioned DRS (equations 58--60).

    This implementation requires ``0 <= sigma < (2 - theta) / 2`` and
    ``0 < theta < 2``. Setting ``sigma=0`` asks for a solve to
    machine precision.  Inner iteration count zero means that the warm start
    already passed the relative-error test. With ``adaptive=True``, every
    ``adaptation_interval`` outer steps the method adjusts ``theta`` from the
    observed contraction of ``max(step, feasibility)`` and adjusts ``sigma``
    to target roughly two inner iterations. No reference solution is used.
    """
    matrix = np.asarray(A, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("A must be a finite two-dimensional array")
    m, n = matrix.shape
    rhs_b = _vector(b, m, "b")
    if gamma_x <= 0 or gamma_lambda <= 0:
        raise ValueError("gamma_x and gamma_lambda must be positive")
    if not (0 < theta < 2):
        raise ValueError("theta must satisfy 0 < theta < 2")
    if not (0 <= sigma < (2 - theta) / 2):
        raise ValueError("sigma must satisfy 0 <= sigma < (2 - theta) / 2")
    solvers = {
        "schur_cg",
        "schur_block_kaczmarz",
        "schur_randomized_kaczmarz",
        "coupled_gmres",
    }
    if linear_solver not in solvers:
        raise ValueError(f"linear_solver must be one of {sorted(solvers)}")
    if max_iterations < 1 or tolerance < 0:
        raise ValueError("invalid iteration limit or tolerance")
    if adaptation_interval < 2:
        raise ValueError("adaptation_interval must be at least 2")
    if block_size is None:
        block_size = max(1, round(0.05 * m))
    if not 1 <= block_size <= m:
        raise ValueError("block_size must lie between 1 and the number of constraints")
    default_inner = {
        "schur_cg": m,
        "schur_block_kaczmarz": 100 * m,
        "schur_randomized_kaczmarz": 100 * m,
        "coupled_gmres": m + n,
    }[linear_solver]
    inner_limit = (
        default_inner if max_inner_iterations is None else max_inner_iterations
    )
    if inner_limit < 1:
        raise ValueError("max_inner_iterations must be positive")

    x = np.zeros(n) if x0 is None else _vector(x0, n, "x0")
    lam = np.zeros(m) if multiplier0 is None else _vector(multiplier0, m, "multiplier0")
    inner_guess = np.zeros(n + m)
    tiny = np.finfo(float).tiny
    kaczmarz_solvers = {"schur_block_kaczmarz", "schur_randomized_kaczmarz"}
    schur_matrix = (
        np.eye(m) + gamma_x * gamma_lambda * matrix @ matrix.T
        if linear_solver in kaczmarz_solvers
        else None
    )
    if schur_matrix is None:
        kaczmarz = None
    elif linear_solver == "schur_block_kaczmarz":
        kaczmarz = BlockKaczmarz(
            schur_matrix, block_size, np.random.default_rng(random_seed)
        )
    else:
        kaczmarz = RandomizedKaczmarz(schur_matrix, np.random.default_rng(random_seed))
    objectives: list[float] = []
    feasibilities: list[float] = []
    steps: list[float] = []
    residuals: list[float] = []
    ratios: list[float] = []
    inner_counts: list[int] = []
    primals: list[Vector] = []
    sigma_values: list[float] = []
    theta_values: list[float] = []
    progress: list[float] = []
    converged = False
    v = _vector(prox_f(x, gamma_x), n, "prox_f result")

    def mnorm(primal: Vector, dual: Vector) -> float:
        squared = (primal @ primal) / gamma_x + (dual @ dual) / gamma_lambda
        return float(np.sqrt(squared))

    def apply_coupled(u: Vector) -> Vector:
        xi, nu = u[:n], u[n:]
        return np.concatenate(
            (xi + gamma_x * matrix.T @ nu, nu - gamma_lambda * matrix @ xi)
        )

    def assess(u: Vector, system_rhs: Vector) -> tuple[bool, _StepState]:
        xi, nu = u[:n], u[n:]
        reflected_x = xi - gamma_x * matrix.T @ nu
        candidate = _vector(prox_f(reflected_x, gamma_x), n, "prox_f result")
        primal_step = candidate - xi
        dual_step = gamma_lambda * (matrix @ xi - rhs_b)
        error = apply_coupled(u) - system_rhs
        error_norm = mnorm(error[:n], error[n:])
        step_norm = mnorm(primal_step, dual_step)
        floor = (
            100 * np.finfo(float).eps * max(1.0, mnorm(system_rhs[:n], system_rhs[n:]))
        )
        accepted = error_norm <= max(sigma * step_norm, floor)
        state = _StepState(
            xi,
            nu,
            candidate,
            primal_step,
            dual_step,
            error_norm,
            step_norm,
        )
        return accepted, state

    for _ in range(max_iterations):
        system_rhs = np.concatenate((x, lam - gamma_lambda * rhs_b))

        def assess_current(u: Vector) -> tuple[bool, _StepState]:
            return assess(u, system_rhs)

        if linear_solver in {"schur_cg", *kaczmarz_solvers}:
            schur_rhs = lam - gamma_lambda * rhs_b + gamma_lambda * matrix @ x

            def schur(z: Vector) -> Vector:
                return z + gamma_x * gamma_lambda * matrix @ (matrix.T @ z)

            def assemble(nu: Vector) -> Vector:
                xi = x - gamma_x * matrix.T @ nu
                return np.concatenate((xi, nu))

            def assess_multiplier(nu: Vector) -> tuple[bool, _StepState]:
                transposed_product = matrix.T @ nu
                xi = x - gamma_x * transposed_product
                candidate = _vector(
                    prox_f(xi - gamma_x * transposed_product, gamma_x),
                    n,
                    "prox_f result",
                )
                error = schur(nu) - schur_rhs
                primal_step = candidate - xi
                dual_step = nu - lam - error
                error_norm = float(np.linalg.norm(error) / np.sqrt(gamma_lambda))
                step_norm = mnorm(primal_step, dual_step)
                floor = (
                    100
                    * np.finfo(float).eps
                    * max(1.0, mnorm(system_rhs[:n], system_rhs[n:]))
                )
                state = _StepState(
                    xi,
                    nu,
                    candidate,
                    primal_step,
                    dual_step,
                    error_norm,
                    step_norm,
                )
                return error_norm <= max(sigma * step_norm, floor), state

            if linear_solver == "schur_cg":
                inner = schur_cg(
                    schur,
                    schur_rhs,
                    inner_guess[n:],
                    assemble,
                    assess_current,
                    inner_limit,
                )
            else:
                assert kaczmarz is not None
                inner = kaczmarz.solve(
                    schur_rhs,
                    inner_guess[n:],
                    assess_multiplier,
                    inner_limit,
                )
        else:
            inner = coupled_gmres(
                apply_coupled,
                system_rhs,
                inner_guess,
                assess_current,
                inner_limit,
            )

        accepted, state = inner.accepted, inner.state
        inner_count = inner.iterations

        if not accepted:
            raise RuntimeError(
                f"{linear_solver} failed relative-error test after "
                f"{inner_limit} steps; "
                f"residual={state.error_norm:.3e}, "
                f"bound={sigma * state.step_norm:.3e}"
            )

        v = state.candidate
        inner_guess = np.concatenate((state.primal, state.multiplier))
        x += theta * state.primal_step
        lam += theta * state.dual_step

        feasibility = float(np.linalg.norm(matrix @ v - rhs_b))
        objectives.append(np.nan if objective is None else float(objective(v)))
        feasibilities.append(feasibility)
        steps.append(state.step_norm)
        residuals.append(state.error_norm)
        ratios.append(state.error_norm / max(state.step_norm, tiny))
        inner_counts.append(inner_count)
        if store_iterates:
            primals.append(v.copy())
        sigma_values.append(sigma)
        theta_values.append(theta)
        progress.append(max(state.step_norm, feasibility))
        if theta * state.step_norm <= tolerance and feasibility <= tolerance:
            converged = True
            break

        if adaptive and len(progress) % adaptation_interval == 0:
            old = max(progress[-adaptation_interval], tiny)
            contraction = (progress[-1] / old) ** (1 / (adaptation_interval - 1))
            if contraction < 0.98:
                theta = min(1.8, theta + 0.1)
            elif contraction > 1.001:
                theta = max(0.6, theta - 0.1)

            mean_inner = float(np.mean(inner_counts[-adaptation_interval:]))
            if mean_inner > 2.5:
                sigma = min(sigma * 1.3, 0.45)
            elif mean_inner < 1.25:
                sigma = max(sigma / 1.3, 0.03)
            sigma = min(sigma, (2 - theta) / 2 - 1e-3)

    return DRSResult(
        primal=v.copy(),
        multiplier=lam.copy(),
        fixed_point=np.concatenate((x, lam)),
        converged=converged,
        iterations=len(steps),
        objective_values=np.asarray(objectives),
        feasibility_norms=np.asarray(feasibilities),
        step_norms_m=np.asarray(steps),
        linear_residual_norms_m=np.asarray(residuals),
        relative_error_ratios=np.asarray(ratios),
        inner_iterations=np.asarray(inner_counts, dtype=int),
        primal_iterates=(np.asarray(primals) if store_iterates else np.empty((0, n))),
        sigma_history=np.asarray(sigma_values),
        theta_history=np.asarray(theta_values),
    )
