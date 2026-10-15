"""Problem instances and their CVXPY ground truth.

Two families are used:

``EqualityQP``
    ``min 1/2 x'Qx + q'x  s.t.  Ax = b`` with ``Q = V diag(lambda) V'`` and
    ``lambda`` log-spaced in ``[1/kappa, 1]``.  A KKT point is planted:
    ``x*`` and ``nu*`` are drawn first and ``q = -Qx* - A'nu*``, ``b = Ax*``.
    Because ``Q`` is positive definite the planted point is the unique
    minimizer, which gives an independent check on the CVXPY solution.
``HuberFit``
    ``min sum_i huber_M((Ax - b)_i)`` with the standard Huber function
    (``r^2/2`` inside ``[-M, M]``, ``M(|r| - M/2)`` outside): robust
    regression with a fraction of gross outliers.  Convex and ``L``-smooth
    with ``L = ||A||_2^2`` but not strongly convex.

Every reference optimum is computed by CVXPY with Clarabel through
:func:`solve_reference`; the tightest tolerance is tried first and relaxed
when Clarabel reports it unattainable.  The achieved tolerance, the KKT
residual of the returned point, and the wall-clock time are all recorded.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

Array = np.ndarray

REFERENCE_SOLVER = cp.CLARABEL
REFERENCE_TOLERANCES = (1e-12, 1e-10, 1e-8)


@dataclass
class GroundTruth:
    """A CVXPY solution together with how accurate and how expensive it was."""

    x: Array
    value: float
    status: str
    tolerance: float
    seconds: float
    stationarity: float
    feasibility: float
    planted_error: float


def solve_reference(problem: cp.Problem) -> tuple[str, float, float]:
    """Solve to the tightest attainable tolerance; return status, tol, seconds."""
    started = time.perf_counter()
    for tolerance in REFERENCE_TOLERANCES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            problem.solve(
                solver=REFERENCE_SOLVER,
                tol_gap_abs=tolerance,
                tol_gap_rel=tolerance,
                tol_feas=tolerance,
                max_iter=500,
            )
        if problem.status == cp.OPTIMAL:
            return problem.status, tolerance, time.perf_counter() - started
    raise RuntimeError(f"reference solve ended with status {problem.status!r}")


@dataclass
class EqualityQP:
    """``min 1/2 x'Qx + q'x`` subject to ``Ax = b`` with a planted solution."""

    Q: Array
    q: Array
    A: Array
    b: Array
    x_planted: Array
    kappa: float
    seed: int

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    @property
    def m(self) -> int:
        return self.A.shape[0]

    def value_and_gradient(self, x: Array) -> tuple[float, Array]:
        Qx = self.Q @ x
        return float(0.5 * x @ Qx + self.q @ x), Qx + self.q

    def gradient(self, x: Array) -> Array:
        return self.Q @ x + self.q

    def objective(self, x: Array) -> float:
        return self.value_and_gradient(x)[0]

    def feasibility(self, x: Array) -> float:
        return float(np.linalg.norm(self.A @ x - self.b))

    def smoothness(self) -> float:
        """Largest eigenvalue of ``Q`` by Lanczos: the gradient's Lipschitz constant."""
        value = scipy.sparse.linalg.eigsh(
            self.Q, k=1, which="LA", return_eigenvectors=False
        )
        return float(value[0])

    def ground_truth(self) -> GroundTruth:
        x = cp.Variable(self.n)
        objective = 0.5 * cp.quad_form(x, cp.psd_wrap(self.Q)) + self.q @ x
        constraint = self.A @ x == self.b
        problem = cp.Problem(cp.Minimize(objective), [constraint])
        status, tolerance, seconds = solve_reference(problem)
        solution = np.asarray(x.value, dtype=float)
        gradient = self.gradient(solution)
        # The multiplier that best explains the gradient, so the stationarity
        # measure does not depend on CVXPY's sign convention for duals.
        nu = np.linalg.lstsq(self.A.T, -gradient, rcond=None)[0]
        return GroundTruth(
            x=solution,
            value=self.objective(solution),
            status=status,
            tolerance=tolerance,
            seconds=seconds,
            stationarity=float(np.linalg.norm(gradient + self.A.T @ nu)),
            feasibility=self.feasibility(solution),
            planted_error=float(np.linalg.norm(solution - self.x_planted)),
        )


def make_qp(seed: int, n: int = 1000, m: int = 300, kappa: float = 100.0) -> EqualityQP:
    """Draw an equality QP with ``kappa(Q) = kappa`` and a planted KKT point."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((n, n)))
    spectrum = np.geomspace(1.0 / kappa, 1.0, n)
    Q = (basis * spectrum) @ basis.T
    Q = 0.5 * (Q + Q.T)
    A = rng.standard_normal((m, n)) / np.sqrt(n)
    x_star = rng.standard_normal(n) / np.sqrt(n)
    nu_star = rng.standard_normal(m) / np.sqrt(m)
    q = -(Q @ x_star) - A.T @ nu_star
    b = A @ x_star
    return EqualityQP(Q=Q, q=q, A=A, b=b, x_planted=x_star, kappa=kappa, seed=seed)


@dataclass
class HuberFit:
    """``min sum_i huber_M((Ax - b)_i)`` with the standard Huber function."""

    A: Array
    b: Array
    threshold: float
    x_true: Array
    seed: int

    @property
    def n(self) -> int:
        return self.A.shape[1]

    @property
    def m(self) -> int:
        return self.A.shape[0]

    def value_and_gradient(self, x: Array) -> tuple[float, Array]:
        r = self.A @ x - self.b
        M = self.threshold
        magnitude = np.abs(r)
        quadratic = magnitude <= M
        values = np.where(quadratic, 0.5 * r * r, M * (magnitude - 0.5 * M))
        slope = np.clip(r, -M, M)
        return float(values.sum()), self.A.T @ slope

    def gradient(self, x: Array) -> Array:
        return self.value_and_gradient(x)[1]

    def objectives(self, points: Array) -> Array:
        """The objective at every row of ``points`` (shape ``(count, n)``)."""
        r = points @ self.A.T - self.b
        M = self.threshold
        magnitude = np.abs(r)
        values = np.where(magnitude <= M, 0.5 * r * r, M * (magnitude - 0.5 * M))
        return values.sum(axis=1)

    def objective(self, x: Array) -> float:
        return self.value_and_gradient(x)[0]

    def smoothness(self) -> float:
        """``||A||_2^2`` by Lanczos on ``A'A``; the Huber curvature is at most one."""
        singular = scipy.sparse.linalg.svds(self.A, k=1, return_singular_vectors=False)
        return float(singular[0] ** 2)

    def ground_truth(self) -> GroundTruth:
        x = cp.Variable(self.n)
        # CVXPY's huber is twice the standard one: x^2 inside, 2M|x| - M^2 outside.
        objective = 0.5 * cp.sum(cp.huber(self.A @ x - self.b, self.threshold))
        problem = cp.Problem(cp.Minimize(objective))
        status, tolerance, seconds = solve_reference(problem)
        solution = np.asarray(x.value, dtype=float)
        value, gradient = self.value_and_gradient(solution)
        return GroundTruth(
            x=solution,
            value=value,
            status=status,
            tolerance=tolerance,
            seconds=seconds,
            stationarity=float(np.linalg.norm(gradient)),
            feasibility=0.0,
            planted_error=float("nan"),
        )


def make_huber(
    seed: int,
    n: int = 1000,
    m: int = 1500,
    threshold: float = 0.5,
    noise: float = 0.1,
    outlier_fraction: float = 0.1,
    outlier_range: tuple[float, float] = (3.0, 8.0),
) -> HuberFit:
    """Draw a robust-regression instance: Gaussian design, noise, and gross outliers."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    x_true = rng.standard_normal(n)
    b = A @ x_true + noise * rng.standard_normal(m)
    outliers = rng.choice(m, size=int(round(outlier_fraction * m)), replace=False)
    magnitude = rng.uniform(*outlier_range, size=outliers.size)
    b[outliers] += rng.choice([-1.0, 1.0], size=outliers.size) * magnitude
    return HuberFit(A=A, b=b, threshold=threshold, x_true=x_true, seed=seed)


class AffineProjector:
    """Projection onto ``{z : Az = b}`` through a Cholesky factor of ``AA'``."""

    def __init__(self, A: Array, b: Array):
        self.A = A
        self.b = b
        self.factor = scipy.linalg.cho_factor(A @ A.T, lower=True)

    def __call__(self, v: Array) -> Array:
        residual = self.A @ v - self.b
        return v - self.A.T @ scipy.linalg.cho_solve(self.factor, residual)
