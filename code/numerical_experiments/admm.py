"""Inexact and exact ADMM for ``min f(x) + g(z)`` subject to ``x = z``.

Algorithm 3 of the paper with ``f`` the smooth quadratic of an equality QP
and ``g`` the indicator of ``{z : Az = b}``.  The ``x``-subproblem
``phi_k(x) = f(x) + <p_k, x> + c/2 ||x - z_k||^2`` is solved either exactly
through a Cholesky factor of ``Q + cI`` (``exact_admm``) or by gradient
descent with stepsize ``2 / (L + 2c)`` warm-started at the previous ``x``
(``inexact_admm``), using the graph pair ``(x, grad f(x))``.  The
``z``-subproblem is the projection onto the affine set and is always exact.

With ``w_k = z_k - p_k / c`` the graph residual is ``e = x + grad f(x)/c - w_k``
and the relative test reads ``||e|| <= sigma ||z_{k+1} + grad f(x)/c - w_k||``;
the terminal test is ``||x - z_{k+1}|| <= delta``.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg

from .ppm import Trace

Array = np.ndarray


def inexact_admm(
    qp,
    c: float,
    smoothness: float,
    projector,
    sigma: float,
    b: int,
    delta: float,
    *,
    exact_tolerance: float | None = None,
    max_outer: int = 50_000,
    max_inner: int = 2_000_000,
) -> Trace:
    """Run Algorithm 3 with gradient descent on the ``x``-subproblem."""
    n = qp.n
    z = np.zeros(n)
    p = np.zeros(n)
    x = z.copy()
    coefficient = 2.0 * c / (smoothness + 2.0 * c)
    trace = Trace(gamma=1.0)
    started = time.perf_counter()
    a = qp.gradient(x)
    evaluations = 1
    k = 0
    while True:
        w = z - p / c
        e = x + a / c - w
        e0_norm = float(np.linalg.norm(e))
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
            if inner >= max_inner:
                raise RuntimeError(f"inner loop exceeded {max_inner} iterations")
            for _ in range(b):
                x = x - coefficient * e
                a = qp.gradient(x)
                e = x + a / c - w
            inner += b
            evaluations += b
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
            break
        p = p + c * (x - z)
        z = z_new
        k += 1
        if k >= max_outer:
            break
    trace.seconds = time.perf_counter() - started
    trace.gradient_evaluations = evaluations
    return trace


def factor_subproblem(qp, c: float):
    """Cholesky factor of ``Q + cI`` for the exact ``x``-update."""
    return scipy.linalg.cho_factor(qp.Q + c * np.eye(qp.n), lower=True)


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
            break
        trace.accepted_by.append("exact")
        p = p_new
        z = z_new
        k += 1
        if k >= max_outer:
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
