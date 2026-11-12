"""Algorithms 1 and 3 of the paper, as run by the original studies.

Ports of ``numerical_experiments.admm`` (``inexact_admm``, ``exact_admm``) and
``numerical_experiments.ppm`` (``inexact_ppm``, ``gradient_descent``,
``accurate_resolvent``) with the same tests, the same hard stops, and the
same per-outer-iteration record, for problems with millions of variables:

* the ADMM inner solver is any solver of :mod:`studies.inner`, started each
  outer iteration at the previously accepted ``x`` with the residual
  ``-c e`` already known, and checked every ``s`` inner iterations (``b`` in
  the code and the tables);
* the original stores every iterate and evaluates the metrics of
  ``histories.csv`` after the run; at this scale the iterates do not fit,
  so ``Diagnostics`` evaluates the same metrics after every outer iteration
  with the clock paused.  ``elapsed`` and ``seconds`` therefore still time
  the algorithm alone, as in the original;
* the inner start may be reset at every outer iteration (the cluster's
  ``fixed`` and ``gaussian`` inner initializations; ``warm``, the default,
  is the original behavior).

ADMM tests (Algorithm 3): stop if ``||x - z_{k+1}|| <= delta``; accept if
``||e|| <= sigma ||z_{k+1} + grad f(x)/c - w_k||`` with
``e = x + grad f(x)/c - w_k``.  PPM tests (Algorithm 1): stop if
``||grad f(x)|| <= delta``; accept if ``||e|| <= sigma ||x - w_k||`` with
``e = gamma grad f(x) + x - w_k``.  ``exact_tolerance`` replaces the relative
test by ``||e|| <= exact_tolerance`` (the near-exact baseline).  Every inexact
run stops after ``max_total_inner`` inner iterations or ``time_limit`` seconds,
checked at every check, besides ``max_outer`` outer iterations; the reason is
recorded in ``Trace.stopped_by``.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from releases.regression_v9.experiments.initialization import settings as validate
from releases.regression_v9.experiments.initialization import starting_point

from .problems import exact_residual

Array = np.ndarray

# The hard stop of every inexact run, as in the original studies.
MAX_TOTAL_INNER = 1_000_000
TIME_LIMIT = 300.0
nan = float("nan")


@dataclass
class Trace:
    """Everything recorded about one run, one entry per outer iteration.

    The fields of the original ``ppm.Trace`` without the stored iterates, and
    the metrics that the original computes from them afterward.
    """

    z_norm: list[float] = field(default_factory=list)
    inner: list[int] = field(default_factory=list)
    checks: list[int] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)
    e0_norm: list[float] = field(default_factory=list)
    y_minus_w: list[float] = field(default_factory=list)
    accepted_by: list[str] = field(default_factory=list)
    suboptimality: list[float] = field(default_factory=list)
    suboptimality_z: list[float] = field(default_factory=list)
    feasibility: list[float] = field(default_factory=list)
    dist: list[float] = field(default_factory=list)
    residual_exact: list[float] = field(default_factory=list)
    converged: bool = False
    # Why the run ended: "converged", "max_outer", "max_total_inner",
    # "time_limit", or "max_iterations" (gradient descent).
    stopped_by: str = ""
    seconds: float = nan
    # Products with the Hessian (ADMM) or gradient evaluations (PPM), the
    # unit in which the original studies report inner work.
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


class Clock:
    """Wall-clock time of an algorithm, excluding paused diagnostics."""

    def __init__(self):
        self.started = time.perf_counter()
        self.paused = 0.0

    def elapsed(self) -> float:
        return time.perf_counter() - self.started - self.paused

    @contextmanager
    def pause(self):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.paused += time.perf_counter() - started


def stop_reason(
    total_inner: int,
    max_total_inner: int | None,
    clock: Clock,
    time_limit: float | None,
) -> str:
    """The hard-stop rule: the budget that is exhausted, or ``""`` if none is."""
    if max_total_inner is not None and total_inner >= max_total_inner:
        return "max_total_inner"
    if time_limit is not None and clock.elapsed() >= time_limit:
        return "time_limit"
    return ""


class Diagnostics:
    """The per-iteration metrics of ``histories.csv``, relative to the truth.

    ``suboptimality`` is ``f(x_k) - f*``; for ADMM also ``suboptimality_z``,
    ``f(z_{k+1}) - f*`` at the feasible iterate, ``feasibility`` of ``x_k``,
    and ``residual_exact``, the exact fixed-point residual ``||R(w_k)||``
    (one exact ``x``-update; ``nan`` when ``exact`` is ``None``).  ``dist``
    is ``||w_k - w*||``.
    """

    def __init__(self, instance, truth_value: float, w_star: Array, c=nan, exact=None):
        self.instance = instance
        self.value = truth_value
        self.w_star = w_star
        self.c = c
        self.exact = exact

    def admm(self, trace: Trace, x: Array, z: Array, w: Array) -> None:
        instance = self.instance
        trace.suboptimality.append(instance.objective(x) - self.value)
        trace.suboptimality_z.append(instance.objective(z) - self.value)
        trace.feasibility.append(instance.feasibility(x))
        trace.dist.append(float(np.linalg.norm(w - self.w_star)))
        trace.residual_exact.append(
            exact_residual(instance, self.exact, self.c, w)
            if self.exact is not None
            else nan
        )

    def ppm(self, trace: Trace, x: Array, w: Array) -> None:
        trace.suboptimality.append(self.instance.objective(x) - self.value)
        trace.suboptimality_z.append(nan)
        trace.feasibility.append(nan)
        trace.dist.append(float(np.linalg.norm(w - self.w_star)))
        trace.residual_exact.append(nan)


class Initialization:
    """The inner starting point of each outer iteration.

    ``warm`` keeps the previously accepted point (the original studies);
    ``fixed`` and ``gaussian`` reset it at every outer iteration exactly as the
    cluster engine does, with draws keyed by ``(initialization_seed, k)``.
    """

    def __init__(self, settings: dict, seed: int):
        seed_value = settings.get("initialization_seed")
        self.config = dict(
            seed=seed,
            inner_initialization=settings.get("inner_initialization", "warm"),
            initialization_seed=int(
                seed + 200_003 if seed_value is None else seed_value
            ),
            gaussian_scale=float(settings.get("gaussian_scale", 1.0)),
            fixed_initial_value=float(settings.get("fixed_initial_value", 0.0)),
        )
        validate(self.config)
        self.mode = self.config["inner_initialization"]

    @property
    def resets(self) -> bool:
        return self.mode != "warm"

    def point(self, previous: Array, outer: int) -> Array:
        return starting_point(previous, outer, self.config)


def inexact_admm(
    instance,
    c: float,
    solver,
    sigma: float,
    b: int,
    delta: float,
    *,
    exact_tolerance: float | None = None,
    max_outer: int = 50_000,
    max_inner: int = 2_000_000,
    max_total_inner: int | None = MAX_TOTAL_INNER,
    time_limit: float | None = TIME_LIMIT,
    diagnostics: Diagnostics | None = None,
    initialization: Initialization | None = None,
) -> Trace:
    """Run Algorithm 3 with ``solver`` on the ``x``-subproblem.

    Each outer iteration starts the solver at the previously accepted ``x``,
    whose residual ``-c e`` is already known, and checks the pair
    ``(x, grad f(x))`` every ``b`` inner iterations; a check forms the
    iterate and evaluates the gradient there.
    """
    n = instance.n
    q = instance.problem.q
    z = np.zeros(n)
    p = np.zeros(n)
    x = z.copy()
    trace = Trace(gamma=1.0)
    clock = Clock()
    a = instance.gradient(x)
    evaluations = 1
    total_inner = 0
    stopped = ""
    k = 0
    while True:
        w = z - p / c
        if initialization is not None and initialization.resets:
            x = initialization.point(x, k)
            a = instance.gradient(x)
            evaluations += 1
        e = x + a / c - w
        e0_norm = float(np.linalg.norm(e))
        rhs = c * w - q
        products = solver.products
        solver.start(x, rhs, -c * e)
        inner = 0
        checks = 0
        while True:
            checks += 1
            z_new = instance.project(x - a / c)
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
            stopped = stop_reason(total_inner, max_total_inner, clock, time_limit)
            if stopped:
                accepted_by = stopped
                break
            if inner >= max_inner:
                raise RuntimeError(f"inner loop exceeded {max_inner} iterations")
            # The product Hx at the checked iterate, which gradient descent
            # reuses for its first step instead of recomputing it.
            hx = a - q
            for _ in range(b):
                if solver.full:
                    # Restart (or recover from a breakdown) at the current
                    # iterate, with its true residual.
                    x = solver.solution()
                    a = instance.gradient(x)
                    evaluations += 1
                    solver.start(x, rhs, -c * (x + a / c - w))
                    if solver.full:
                        break
                    hx = a - q
                solver.step(hx)
                hx = None
            inner += b
            total_inner += b
            x = solver.solution()
            a = instance.gradient(x)
            evaluations += 1
            e = x + a / c - w
        evaluations += solver.products - products
        trace.elapsed.append(clock.elapsed())
        trace.z_norm.append(zeta_norm)
        trace.inner.append(inner)
        trace.checks.append(checks)
        trace.e0_norm.append(e0_norm)
        trace.y_minus_w.append(y_minus_w)
        trace.accepted_by.append(accepted_by)
        if diagnostics is not None:
            with clock.pause():
                diagnostics.admm(trace, x, z_new, w)
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
    trace.seconds = clock.elapsed()
    trace.gradient_evaluations = evaluations
    return trace


def exact_admm(
    instance,
    c: float,
    exact,
    delta: float,
    *,
    max_outer: int = 50_000,
    diagnostics: Diagnostics | None = None,
) -> Trace:
    """Standard ADMM with the ``x``-subproblem solved by ``exact.solve``."""
    n = instance.n
    q = instance.problem.q
    z = np.zeros(n)
    p = np.zeros(n)
    trace = Trace(gamma=1.0)
    clock = Clock()
    k = 0
    while True:
        w = z - p / c
        x = exact.solve(c * z - q - p)
        p_new = p + c * (x - z)
        z_new = instance.project(x + p_new / c)
        zeta_norm = float(np.linalg.norm(x - z_new))
        trace.elapsed.append(clock.elapsed())
        trace.z_norm.append(zeta_norm)
        trace.inner.append(0)
        trace.checks.append(1)
        trace.e0_norm.append(nan)
        trace.y_minus_w.append(float(np.linalg.norm(z_new + (-p_new) / c - w)))
        if diagnostics is not None:
            with clock.pause():
                diagnostics.admm(trace, x, z_new, w)
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
    trace.seconds = clock.elapsed()
    trace.gradient_evaluations = 0
    return trace


def inexact_ppm(
    instance,
    gamma: float,
    sigma: float,
    b: int,
    delta: float,
    *,
    exact_tolerance: float | None = None,
    max_outer: int = 50_000,
    max_inner: int = 2_000_000,
    max_total_inner: int | None = MAX_TOTAL_INNER,
    time_limit: float | None = TIME_LIMIT,
    gd_factor: float = 1.0,
    diagnostics: Diagnostics | None = None,
    initialization: Initialization | None = None,
) -> Trace:
    """Run Algorithm 1 for ``min f`` with gradient descent as the inner solver.

    The inner step on ``phi_k(x) = f(x) + ||x - w_k||^2 / (2 gamma)`` is the
    original's ``2 gamma / (gamma L + 2)``, scaled by ``gd_factor``: in the
    residual ``e = gamma grad phi_k(x)`` it reads ``x <- x - coefficient e``.
    The engine's logistic step also counts the ridge ``mu``; this one does not,
    so that it contracts by ``r/(r+2)``, ``r = gamma L``, for every ``mu >= 0``,
    exactly the constant of Proposition 5.1 (``inner.contraction``).
    """
    n = instance.n
    w = np.zeros(n)
    x = w.copy()
    coefficient = gd_factor * 2.0 / (gamma * instance.problem.L + 2.0)
    trace = Trace(gamma=gamma)
    clock = Clock()
    g = instance.gradient(x)
    evaluations = 1
    total_inner = 0
    stopped = ""
    k = 0
    while True:
        if initialization is not None and initialization.resets:
            x = initialization.point(x, k)
            g = instance.gradient(x)
            evaluations += 1
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
            stopped = stop_reason(total_inner, max_total_inner, clock, time_limit)
            if stopped:
                accepted_by = stopped
                break
            if inner >= max_inner:
                raise RuntimeError(f"inner loop exceeded {max_inner} iterations")
            for _ in range(b):
                x = x - coefficient * e
                g = instance.gradient(x)
                e = gamma * g + (x - w)
            inner += b
            total_inner += b
            evaluations += b
        trace.elapsed.append(clock.elapsed())
        trace.z_norm.append(z_norm)
        trace.inner.append(inner)
        trace.checks.append(checks)
        trace.e0_norm.append(e0_norm)
        trace.y_minus_w.append(y_minus_w)
        trace.accepted_by.append(accepted_by)
        if diagnostics is not None:
            with clock.pause():
                diagnostics.ppm(trace, x, w)
        if accepted_by == "terminal":
            trace.converged = True
            trace.stopped_by = "converged"
            break
        if stopped:
            trace.stopped_by = stopped
            break
        w = w - gamma * g
        k += 1
        if k >= max_outer:
            trace.stopped_by = "max_outer"
            break
    trace.seconds = clock.elapsed()
    trace.gradient_evaluations = evaluations
    return trace


def gradient_descent(
    instance,
    smoothness: float,
    delta: float,
    *,
    stepsize_ratio: float = 1.0,
    max_iterations: int = 1_000_000,
    dense: int = 200,
    spacing: float = 1.02,
    diagnostics: Diagnostics | None = None,
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
    x = np.zeros(instance.n)
    stepsize = stepsize_ratio / smoothness
    trace = Trace(gamma=stepsize, iteration=[])
    clock = Clock()
    g = instance.gradient(x)
    evaluations = 1
    k = 0
    previous = -1
    next_record = 0
    while True:
        norm = float(np.linalg.norm(g))
        done = norm <= delta
        capped = k >= max_iterations
        if done or capped or k >= next_record:
            trace.elapsed.append(clock.elapsed())
            trace.z_norm.append(norm)
            trace.inner.append(k - previous if previous >= 0 else 0)
            trace.checks.append(k - previous if previous >= 0 else 1)
            trace.e0_norm.append(nan)
            trace.y_minus_w.append(0.0)
            trace.accepted_by.append("terminal" if done else "step")
            trace.iteration.append(k)
            if diagnostics is not None:
                with clock.pause():
                    diagnostics.ppm(trace, x, x)
            previous = k
            while next_record <= k:
                next_record = (
                    next_record + 1
                    if next_record < dense
                    else int(math.ceil(next_record * spacing))
                )
        if done:
            trace.converged = True
            trace.stopped_by = "converged"
            break
        if capped:
            trace.stopped_by = "max_iterations"
            break
        x = x - stepsize * g
        g = instance.gradient(x)
        evaluations += 1
        k += 1
    trace.seconds = clock.elapsed()
    trace.gradient_evaluations = evaluations
    return trace


def accurate_resolvent(
    instance, gamma: float, w: Array, tolerance: float = 1e-13
) -> tuple[Array, float]:
    """``J(w)`` for ``B = grad f`` to near machine precision, and ``||R(w)||``.

    Gradient descent on the proximal subproblem until the residual
    ``||gamma grad f(x) + x - w||`` drops below ``tolerance (1 + ||w||)`` or
    stops improving.  A diagnostic, never part of a timed run.
    """
    coefficient = 2.0 / (gamma * instance.problem.L + 2.0)
    x = w.copy()
    floor = tolerance * (1.0 + float(np.linalg.norm(w)))
    best = float("inf")
    stalled = 0
    for _ in range(200_000):
        e = gamma * instance.gradient(x) + (x - w)
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
