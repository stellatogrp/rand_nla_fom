"""Inexact proximal point with a gradient-descent inner solver (Proposition 5.1).

Algorithm 1 of the paper with ``C = I`` and ``A = grad f``: at outer
iteration ``k`` the inner loop runs gradient descent with stepsize
``2 gamma / (gamma L + 2)`` on ``phi_k(x) = f(x) + ||x - w_k||^2 / (2 gamma)``,
warm-started at the previously accepted point, and uses the graph pair
``(x, grad f(x))`` whose residual is ``e = gamma grad f(x) + x - w_k``.
The pair is checked every ``b`` gradient steps: stop if ``||grad f(x)|| <= delta``,
accept if ``||e|| <= sigma ||x - w_k||``, otherwise take ``b`` more steps.

``exact_tolerance`` replaces the relative test by the absolute test
``||e|| <= exact_tolerance``, which is the near-exact baseline.

:func:`gradient_descent` is the classical baseline: plain gradient descent
on ``f`` with a constant stepsize (``1/L`` by default) and the same stopping
test ``||grad f(x)|| <= delta``, so that one of its iterations costs one
inner iteration of the proximal method.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray


@dataclass
class Trace:
    """Everything recorded about one run, one entry per outer iteration."""

    iterates: list[Array] = field(default_factory=list)
    w: list[Array] = field(default_factory=list)
    z_norm: list[float] = field(default_factory=list)
    inner: list[int] = field(default_factory=list)
    checks: list[int] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)
    e0_norm: list[float] = field(default_factory=list)
    y_minus_w: list[float] = field(default_factory=list)
    accepted_by: list[str] = field(default_factory=list)
    feasible_iterates: list[Array] = field(default_factory=list)
    converged: bool = False
    seconds: float = float("nan")
    gradient_evaluations: int = 0
    gamma: float = 1.0
    # The iteration number of each entry when the record is thinned (gradient
    # descent); ``None`` means one entry per iteration.
    iteration: list[int] | None = None

    @property
    def k_delta(self) -> int:
        """Updates before stopping; the paper's ``K_delta`` for Algorithm 1."""
        if self.iteration is not None:
            return self.iteration[-1]
        return len(self.inner) - 1


def inexact_ppm(
    problem,
    gamma: float,
    smoothness: float,
    sigma: float,
    b: int,
    delta: float,
    *,
    exact_tolerance: float | None = None,
    w0: Array | None = None,
    max_outer: int = 50_000,
    max_inner: int = 2_000_000,
) -> Trace:
    """Run Algorithm 1 for ``min f`` with gradient descent as the inner solver."""
    n = problem.n
    w = np.zeros(n) if w0 is None else np.array(w0, dtype=float)
    x = w.copy()
    coefficient = 2.0 / (gamma * smoothness + 2.0)
    trace = Trace(gamma=gamma)
    started = time.perf_counter()
    g = problem.gradient(x)
    evaluations = 1
    k = 0
    while True:
        e = gamma * g + (x - w)
        e0_norm = float(np.linalg.norm(e))
        inner = 0
        checks = 0
        while True:
            checks += 1
            z_norm = float(np.linalg.norm(g))
            y_minus_w = float(np.linalg.norm(x - w))
            if z_norm <= delta:
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
                g = problem.gradient(x)
                e = gamma * g + (x - w)
            inner += b
            evaluations += b
        trace.elapsed.append(time.perf_counter() - started)
        trace.iterates.append(x.copy())
        trace.w.append(w.copy())
        trace.z_norm.append(z_norm)
        trace.inner.append(inner)
        trace.checks.append(checks)
        trace.e0_norm.append(e0_norm)
        trace.y_minus_w.append(y_minus_w)
        trace.accepted_by.append(accepted_by)
        if accepted_by == "terminal":
            trace.converged = True
            break
        w = w - gamma * g
        k += 1
        if k >= max_outer:
            break
    trace.seconds = time.perf_counter() - started
    trace.gradient_evaluations = evaluations
    return trace


def gradient_descent(
    problem,
    smoothness: float,
    delta: float,
    *,
    stepsize_ratio: float = 1.0,
    w0: Array | None = None,
    max_iterations: int = 1_000_000,
    dense: int = 200,
    spacing: float = 1.02,
) -> Trace:
    """Plain gradient descent ``x <- x - (stepsize_ratio / L) grad f(x)``.

    Stops when ``||grad f(x)|| <= delta``, the test of :func:`inexact_ppm`.
    The trace records each of the first ``dense`` iterations, then
    iterations at the geometric spacing ``spacing``, and always the last
    one: ``Trace.iteration`` holds their numbers, ``inner`` and ``checks``
    the gradient steps and norm checks since the previous record, and
    ``z_norm`` the gradient norm, so the cumulative inner work of a record
    is its iteration number.
    """
    n = problem.n
    x = np.zeros(n) if w0 is None else np.array(w0, dtype=float)
    stepsize = stepsize_ratio / smoothness
    trace = Trace(gamma=stepsize, iteration=[])
    started = time.perf_counter()
    g = problem.gradient(x)
    evaluations = 1
    k = 0
    previous = -1
    next_record = 0
    while True:
        norm = float(np.linalg.norm(g))
        done = norm <= delta
        capped = k >= max_iterations
        if done or capped or k >= next_record:
            trace.elapsed.append(time.perf_counter() - started)
            trace.iterates.append(x.copy())
            trace.w.append(x.copy())
            trace.z_norm.append(norm)
            trace.inner.append(k - previous if previous >= 0 else 0)
            trace.checks.append(k - previous if previous >= 0 else 1)
            trace.e0_norm.append(float("nan"))
            trace.y_minus_w.append(0.0)
            trace.accepted_by.append("terminal" if done else "step")
            trace.iteration.append(k)
            previous = k
            while next_record <= k:
                next_record = (
                    next_record + 1
                    if next_record < dense
                    else int(math.ceil(next_record * spacing))
                )
        if done:
            trace.converged = True
            break
        if capped:
            break
        x = x - stepsize * g
        g = problem.gradient(x)
        evaluations += 1
        k += 1
    trace.seconds = time.perf_counter() - started
    trace.gradient_evaluations = evaluations
    return trace


def accurate_resolvent(
    problem, gamma: float, smoothness: float, w: Array, tolerance: float = 1e-13
) -> tuple[Array, float]:
    """``J(w)`` for ``B = grad f`` to near machine precision, and ``||R(w)||``.

    Gradient descent on the proximal subproblem until the residual
    ``||gamma grad f(x) + x - w||`` drops below ``tolerance (1 + ||w||)`` or
    stops improving.  A diagnostic, never part of a timed run.
    """
    coefficient = 2.0 / (gamma * smoothness + 2.0)
    x = w.copy()
    floor = tolerance * (1.0 + float(np.linalg.norm(w)))
    best = float("inf")
    stalled = 0
    for _ in range(200_000):
        e = gamma * problem.gradient(x) + (x - w)
        norm = float(np.linalg.norm(e))
        if norm <= floor:
            break
        if norm < 0.999 * best:
            best = norm
            stalled = 0
        else:
            stalled += 1
            if stalled > 200:
                break
        x = x - coefficient * e
    return x, float(np.linalg.norm(x - w))
