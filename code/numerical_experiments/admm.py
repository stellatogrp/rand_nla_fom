"""Inexact and exact ADMM for ``min f(x) + g(z)`` subject to ``x = z``.

Algorithm 3 of the paper with ``f`` the smooth quadratic of an equality QP
and ``g`` the indicator of ``{z : Az = b}``.  The ``x``-subproblem
``phi_k(x) = f(x) + <p_k, x> + c/2 ||x - z_k||^2`` is the linear system
``(Q + cI) x = c w_k - q``.  It is solved either exactly through a Cholesky
factor of ``Q + cI`` (``exact_admm``) or by GMRES warm-started at the previous
``x`` (``inexact_admm``), using the graph pair ``(x, grad f(x))``.  The
``z``-subproblem is the projection onto the affine set and is always exact.

With ``w_k = z_k - p_k / c`` the graph residual is ``e = x + grad f(x)/c - w_k``,
which for a quadratic is ``1/c`` times the residual ``(Q + cI) x - (c w_k - q)``
of the linear system, so GMRES minimizes ``||e||`` over its Krylov space.
The relative test reads ``||e|| <= sigma ||z_{k+1} + grad f(x)/c - w_k||``;
the terminal test is ``||x - z_{k+1}|| <= delta``.  The hard stop of
:mod:`numerical_experiments.ppm` applies to the inexact method: ``max_total_inner``
GMRES iterations in total or ``time_limit`` seconds, with the reason recorded
in ``Trace.stopped_by``.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg

from .ppm import MAX_TOTAL_INNER, TIME_LIMIT, Trace, stop_reason

Array = np.ndarray

# Krylov dimension at which GMRES restarts.  The contraction constants of
# ``theory.gmres_constants`` hold within one cycle, and no outer iteration of
# the study comes near this many inner iterations (``max_inner_per_outer``).
RESTART = 200


class Gmres:
    """GMRES on ``(Q + cI) x = rhs``, one Arnoldi step at a time.

    The basis is orthogonalized by classical Gram-Schmidt with one
    reorthogonalization, and the Hessenberg least-squares problem is kept in
    triangular form by Givens rotations, so ``solution`` costs one triangular
    solve and one product with the basis.  ``start`` begins a cycle at ``x0``
    given its residual ``rhs - (Q + cI) x0``, which the caller already has.
    """

    def __init__(self, Q: Array, c: float, restart: int = RESTART):
        self.Q = Q
        self.c = c
        self.restart = restart
        n = Q.shape[0]
        self.V = np.empty((restart + 1, n))
        self.R = np.zeros((restart + 1, restart))
        self.cs = np.empty(restart)
        self.sn = np.empty(restart)
        self.g = np.zeros(restart + 1)

    def start(self, x0: Array, residual: Array) -> None:
        self.x0 = x0
        self.j = 0
        beta = float(np.linalg.norm(residual))
        self.g[:] = 0.0
        self.g[0] = beta
        # A zero residual leaves nothing to do: the cycle is already exhausted.
        self.exhausted = beta == 0.0
        if not self.exhausted:
            self.V[0] = residual / beta

    @property
    def full(self) -> bool:
        """No further step is possible in this cycle (restart or breakdown)."""
        return self.exhausted or self.j == self.restart

    def step(self) -> None:
        """One Arnoldi step: one product with ``Q`` and the orthogonalization."""
        j = self.j
        V = self.V[: j + 1]
        u = self.Q @ V[j] + self.c * V[j]
        h = V @ u
        u -= h @ V
        correction = V @ u
        u -= correction @ V
        h += correction
        norm = float(np.linalg.norm(u))
        column = np.append(h, norm)
        for i in range(j):
            first = self.cs[i] * column[i] + self.sn[i] * column[i + 1]
            column[i + 1] = -self.sn[i] * column[i] + self.cs[i] * column[i + 1]
            column[i] = first
        radius = float(np.hypot(column[j], column[j + 1]))
        self.cs[j] = column[j] / radius
        self.sn[j] = column[j + 1] / radius
        column[j] = radius
        self.R[: j + 1, j] = column[: j + 1]
        self.g[j + 1] = -self.sn[j] * self.g[j]
        self.g[j] = self.cs[j] * self.g[j]
        self.j = j + 1
        # Happy breakdown: the Krylov space is invariant and holds the solution.
        if norm <= 1e-14 * radius:
            self.exhausted = True
        else:
            self.V[j + 1] = u / norm

    def solution(self) -> Array:
        """The iterate minimizing the residual over ``x0`` plus the Krylov space."""
        if self.j == 0:
            return self.x0
        y = scipy.linalg.solve_triangular(self.R[: self.j, : self.j], self.g[: self.j])
        return self.x0 + y @ self.V[: self.j]


def inexact_admm(
    qp,
    c: float,
    projector,
    sigma: float,
    b: int,
    delta: float,
    *,
    exact_tolerance: float | None = None,
    max_outer: int = 50_000,
    max_inner: int = 2_000_000,
    max_total_inner: int | None = MAX_TOTAL_INNER,
    time_limit: float | None = TIME_LIMIT,
    restart: int = RESTART,
) -> Trace:
    """Run Algorithm 3 with GMRES on the ``x``-subproblem.

    Each outer iteration starts a GMRES cycle at the previously accepted
    ``x``, whose residual ``-c e`` is already known, and checks the pair
    ``(x, grad f(x))`` every ``b`` GMRES iterations; a check forms the
    iterate and evaluates the gradient there, which is one product with ``Q``
    besides the ``b`` of the Arnoldi steps.
    """
    n = qp.n
    z = np.zeros(n)
    p = np.zeros(n)
    x = z.copy()
    solver = Gmres(qp.Q, c, restart)
    trace = Trace(gamma=1.0)
    started = time.perf_counter()
    a = qp.gradient(x)
    evaluations = 1
    total_inner = 0
    stopped = ""
    k = 0
    while True:
        w = z - p / c
        e = x + a / c - w
        e0_norm = float(np.linalg.norm(e))
        solver.start(x, -c * e)
        inner = 0
        checks = 0
        while True:
            checks += 1
            z_new = projector(x - a / c)
            zeta_norm = float(np.linalg.norm(x - z_new))
            y_minus_w = float(np.linalg.norm(z_new + a / c - w))
            if zeta_norm <= delta:
                accepted_by = "terminal"
                break
            e_norm = float(np.linalg.norm(e))
            if exact_tolerance is None:
                passed = e_norm <= sigma * y_minus_w
            else:
                passed = e_norm <= exact_tolerance
            if passed:
                accepted_by = "relative"
                break
            stopped = stop_reason(total_inner, max_total_inner, started, time_limit)
            if stopped:
                accepted_by = stopped
                break
            if inner >= max_inner:
                raise RuntimeError(f"inner loop exceeded {max_inner} iterations")
            for _ in range(b):
                if solver.full:
                    # Restart (or recover from a breakdown) at the current
                    # iterate, with its true residual.
                    x = solver.solution()
                    a = qp.gradient(x)
                    evaluations += 1
                    solver.start(x, -c * (x + a / c - w))
                    if solver.full:
                        break
                solver.step()
            inner += b
            total_inner += b
            evaluations += b
            x = solver.solution()
            a = qp.gradient(x)
            evaluations += 1
            e = x + a / c - w
        trace.elapsed.append(time.perf_counter() - started)
        trace.iterates.append(x.copy())
        trace.feasible_iterates.append(z_new)
        trace.w.append(w)
        trace.z_norm.append(zeta_norm)
        trace.inner.append(inner)
        trace.checks.append(checks)
        trace.e0_norm.append(e0_norm)
        trace.y_minus_w.append(y_minus_w)
        trace.accepted_by.append(accepted_by)
        if accepted_by == "terminal":
            trace.converged = True
            trace.stopped_by = "converged"
            break
        if stopped:
            trace.stopped_by = stopped
            break
        p = p + c * (x - z)
        z = z_new
        k += 1
        if k >= max_outer:
            trace.stopped_by = "max_outer"
            break
    trace.seconds = time.perf_counter() - started
    trace.gradient_evaluations = evaluations
    return trace


def factor_subproblem(qp, c: float):
    """Cholesky factor of ``Q + cI`` for the exact ``x``-update."""
    # numpy's Cholesky rather than scipy.linalg.cho_factor: with lower=True SciPy
    # calls Accelerate's dpotrf with uplo='U', which writes out of bounds from
    # order 1449 (heap corruption, crashes later in unrelated code); see
    # "Known issue" in README.md.  The lower factor with the flag
    # ``True`` is what scipy.linalg.cho_solve expects; Fortran order spares
    # LAPACK a copy of the factor on every solve.
    return np.asfortranarray(np.linalg.cholesky(qp.Q + c * np.eye(qp.n))), True


def exact_admm(
    qp, c: float, factor, projector, delta: float, *, max_outer: int = 50_000
) -> Trace:
    """Standard ADMM with the ``x``-subproblem solved by the Cholesky factor."""
    n = qp.n
    z = np.zeros(n)
    p = np.zeros(n)
    trace = Trace(gamma=1.0)
    started = time.perf_counter()
    k = 0
    while True:
        w = z - p / c
        x = scipy.linalg.cho_solve(factor, c * z - qp.q - p)
        p_new = p + c * (x - z)
        z_new = projector(x + p_new / c)
        zeta_norm = float(np.linalg.norm(x - z_new))
        trace.elapsed.append(time.perf_counter() - started)
        trace.iterates.append(x)
        trace.feasible_iterates.append(z_new)
        trace.w.append(w)
        trace.z_norm.append(zeta_norm)
        trace.inner.append(0)
        trace.checks.append(1)
        trace.e0_norm.append(float("nan"))
        trace.y_minus_w.append(float(np.linalg.norm(z_new + (-p_new) / c - w)))
        if zeta_norm <= delta:
            trace.accepted_by.append("terminal")
            trace.converged = True
            trace.stopped_by = "converged"
            break
        trace.accepted_by.append("exact")
        p = p_new
        z = z_new
        k += 1
        if k >= max_outer:
            trace.stopped_by = "max_outer"
            break
    trace.seconds = time.perf_counter() - started
    trace.gradient_evaluations = 0
    return trace


def exact_residual(qp, c: float, factor, projector, w: Array) -> float:
    """``||R(w)||`` for the ADMM operator: the exact step's ``||x - z||`` at ``w``."""
    x = scipy.linalg.cho_solve(factor, c * w - qp.q)
    a = qp.gradient(x)
    z = projector(x - a / c)
    return float(np.linalg.norm(x - z))


def exact_residuals(
    qp, c: float, factor, projector, ws: Array, chunk: int = 512
) -> Array:
    """:func:`exact_residual` at every row of ``ws``, in blocks of ``chunk`` rows.

    Solving for a block of right-hand sides at once runs the triangular solves
    and the products as matrix-matrix operations.
    """
    ws = np.atleast_2d(ws)
    out = np.empty(len(ws))
    for start in range(0, len(ws), chunk):
        block = ws[start : start + chunk].T
        x = scipy.linalg.cho_solve(factor, c * block - qp.q[:, None])
        a = qp.Q @ x + qp.q[:, None]
        v = x - a / c
        correction = scipy.linalg.cho_solve(
            projector.factor, projector.A @ v - projector.b[:, None]
        )
        z = v - projector.A.T @ correction
        out[start : start + chunk] = np.linalg.norm(x - z, axis=0)
    return out
