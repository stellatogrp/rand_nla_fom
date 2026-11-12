"""Algorithm 3 (doubly inexact proximal ADMM) for a semidefinite program.

The SDP ``minimize <C, X>`` subject to ``E(X) = b`` and ``X`` positive
semidefinite is the composite problem of the paper with

    f(X) = <C, X>,    g(y, Z) = indicator of {y = b} and {Z psd},
    coupling map  X -> (E(X), X) = (y, Z),

penalty ``c > 0`` and proximal parameter ``rho > 0``.  The outer state is
``W_k = (w_k, sqrt(rho) t_k)`` with ``w_k = z_k - p_k / c = (w_y, w_Z)``.

**The ``f``-subproblem** minimizes
``<C, X> + c/2 ||E(X) - w_y||^2 + c/2 ||X - w_Z||^2 + c rho/2 ||X - t||^2``,
the positive definite linear system

    (E*E + (1 + rho) I) X = E*(w_y) + w_Z + rho t - C / c,

which conjugate gradients solves approximately in the Frobenius inner
product, warm-started at the previous ``X``.  Because ``f`` is linear its
subgradient certificate is ``a = C`` at every ``X``, so the pair ``(X, C)``
is on the graph exactly and ``r_f``, ``1/c`` times the gradient of the
subproblem, is the residual of that system.

**The ``g``-subproblem** is the projection of ``(2 E(X) - w_y, 2 X - w_Z)``.
Its ``y`` part is solved exactly: ``y = b`` with the subgradient
``b_y = c (2 E(X) - w_y - b)`` (every vector is normal to a point).  Its
``Z`` part is the projection of ``M = 2 X - w_Z`` onto the cone, done
approximately by block Lanczos (:mod:`sdp.projection`): the positive Ritz
pairs give ``Z``, the negative ones give ``B`` with ``b_Z = c B`` in the
normal cone at ``Z`` exactly, and ``r_g = Z + B - M`` is the inexactness.

With those choices the quantities of Algorithm 3 are

    d   = (E(X) - b, X - Z),
    q   = r_f + r_g - rho (X - t),
    Q^2 = ||d||^2 + ||q||^2 / rho,
    E^2 = ||r_g||^2 + ||r_f + r_g||^2 / rho,
    P^2 = ||E(X) - b||^2 + ||w_Z + B - X||^2 + rho ||X - t||^2.

Each check stops if ``Q <= delta``, accepts if ``E <= sigma P`` (then
``w <- w - (E(X), X) + (b, Z)`` and ``t <- t - q / rho``), and otherwise
improves the pair.  Since ``E <= e_f + e_g`` with ``e_f = ||r_f|| / sqrt(rho)``
and ``e_g = ||r_g|| sqrt(1 + 1/rho)``, the improvement first runs conjugate
gradients until ``e_f <= f_share sigma P`` and then extends the Krylov
subspace of the projection, ``stride`` steps of either between checks.  A
Krylov subspace belongs to one ``M``: after ``X`` or ``w_Z`` changes it is
rebuilt from the retained Ritz vectors (:func:`sdp.projection.restart_block`).
The first check of an outer iteration reuses the previous ``(X, Z, B)``
unchanged, the warm start of the paper.

``f_solver="exact"`` solves the linear system through the ``m x m`` Cholesky
factor of ``E E* + (1 + rho) I`` and ``projection="exact"`` uses a full
eigendecomposition; both together are exact proximal ADMM, whose updates are
accepted without a test.  ``exact_tolerance`` replaces the relative test by
``E <= exact_tolerance`` (the near-exact baseline).
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg

from .problems import SDP
from .projection import (
    GUARD_FACTOR,
    MARGIN,
    BlockLanczos,
    ConePair,
    certify,
    restart_block,
    restart_one_sided,
)

Array = np.ndarray
nan = float("nan")


@dataclass
class Trace:
    """One run: an entry per outer iteration, totals, and the returned point."""

    Q: list[float] = field(default_factory=list)
    E: list[float] = field(default_factory=list)
    P: list[float] = field(default_factory=list)
    e_f: list[float] = field(default_factory=list)
    e_g: list[float] = field(default_factory=list)
    cg_steps: list[int] = field(default_factory=list)
    matvecs: list[int] = field(default_factory=list)
    lanczos_steps: list[int] = field(default_factory=list)
    restarts: list[int] = field(default_factory=list)
    checks: list[int] = field(default_factory=list)
    # Dimension of the subspace behind the accepted (Z, B), and the number of
    # positive and negative Ritz values in it.
    subspace_dim: list[int] = field(default_factory=list)
    rank_positive: list[int] = field(default_factory=list)
    rank_negative: list[int] = field(default_factory=list)
    # Smallest subspace dimension that could pass the test, a lower bound from
    # the eigenvalues of M (diagnostic; -1 when not computed).
    rank_needed: list[int] = field(default_factory=list)
    # One-sided projection: pairs that passed a test but not the guard, the
    # Cholesky certificates attempted and failed, and (diagnostic, nan when
    # not computed) the largest eigenvalue of the accepted B relative to the
    # larger of ||B|| and ||Z||, which is positive only if B is not in the
    # normal cone.
    guard_rejections: list[int] = field(default_factory=list)
    certificates: list[int] = field(default_factory=list)
    certificate_failures: list[int] = field(default_factory=list)
    violation: list[float] = field(default_factory=list)
    accepted_by: list[str] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)
    # ||W_k - W*|| at the start of iteration k (nan without a planted solution).
    dist: list[float] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    converged: bool = False
    # "converged", "max_outer", "time_limit", or "stalled".
    stopped_by: str = ""
    seconds: float = nan
    forward: int = 0
    adjoint: int = 0
    eigendecompositions: int = 0
    # Wall-clock spent computing cone pairs (eigendecompositions, Lanczos
    # steps, Rayleigh-Ritz, forming Z and B) and in Cholesky certificates.
    projection_seconds: float = 0.0
    certificate_seconds: float = 0.0
    # The returned point (X, (b, Z), C, (b_y, c B)) with nu = -b_y.
    X: Array | None = None
    Z: Array | None = None
    B: Array | None = None
    nu: Array | None = None
    EX: Array | None = None

    @property
    def outer_iterations(self) -> int:
        """Accepted updates before stopping, the paper's ``K_delta``."""
        return len(self.Q) - 1


class Clock:
    """Wall-clock time of the algorithm, excluding paused diagnostics."""

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


class _Counted:
    """The constraint map with its applications counted."""

    def __init__(self, E):
        self.E = E
        self.forward = 0
        self.adjoint_calls = 0

    def apply(self, X: Array) -> Array:
        self.forward += 1
        return self.E.apply(X)

    def adjoint(self, u: Array) -> Array:
        self.adjoint_calls += 1
        return self.E.adjoint(u)


class ConjugateGradient:
    """Conjugate gradients on ``(E*E + alpha I) X = rhs`` over symmetric matrices.

    ``start`` takes the iterate, ``E(X)``, and the residual ``rhs - H X``;
    each ``step`` applies ``E`` and its adjoint once and updates all three.
    ``exhausted`` means that the residual has reached rounding error (or is
    zero), so further steps cannot improve it.
    """

    def __init__(self, E, alpha: float, operator_norm: float):
        self.E = E
        self.alpha = alpha
        self.scale = 8.0 * np.finfo(float).eps * (alpha + operator_norm**2)

    def start(self, X: Array, EX: Array, R: Array) -> None:
        self.X, self.EX, self.R = X, EX, R
        self.D = R.copy()
        self.rr = float(np.vdot(R, R))
        self._update_exhausted()

    def _update_exhausted(self) -> None:
        floor = self.scale * float(np.linalg.norm(self.X))
        self.exhausted = self.rr == 0.0 or math.sqrt(self.rr) <= floor

    def step(self) -> None:
        ED = self.E.apply(self.D)
        HD = self.E.adjoint(ED) + self.alpha * self.D
        curvature = float(np.vdot(self.D, HD))
        if curvature <= 0.0:
            self.exhausted = True
            return
        step = self.rr / curvature
        self.X = self.X + step * self.D
        self.EX = self.EX + step * ED
        self.R = self.R - step * HD
        rr = float(np.vdot(self.R, self.R))
        self.D = self.R + (rr / self.rr) * self.D
        self.rr = rr
        self._update_exhausted()


@dataclass
class _Measure:
    Q: float
    E: float
    P: float
    e_f: float
    e_g: float
    q: Array


def _measure(X, EX, R, pair: ConePair, w_Z, t, b, rho: float) -> _Measure:
    """The test quantities of Algorithm 3 at the current pair (module docstring)."""
    r_g = pair.Z + pair.B - (2.0 * X - w_Z)
    total = r_g - R  # r_f + r_g, with r_f = -R
    shift = X - t
    q = total - rho * shift
    d_y = float(np.linalg.norm(EX - b))
    norm_g = float(np.linalg.norm(r_g))
    return _Measure(
        Q=math.sqrt(
            d_y**2 + float(np.linalg.norm(X - pair.Z)) ** 2 + np.vdot(q, q) / rho
        ),
        E=math.sqrt(norm_g**2 + float(np.vdot(total, total)) / rho),
        P=math.sqrt(
            d_y**2
            + float(np.linalg.norm(w_Z + pair.B - X)) ** 2
            + rho * float(np.vdot(shift, shift))
        ),
        e_f=float(np.linalg.norm(R)) / math.sqrt(rho),
        e_g=norm_g * math.sqrt(1.0 + 1.0 / rho),
        q=q,
    )


def _rank_needed(M: Array, tau: float) -> int:
    """Smallest subspace dimension whose Rayleigh-Ritz pair can pass ``E <= tau``.

    For a subspace of dimension ``k`` the residual ``P M P - M`` has Frobenius
    norm at least that of the ``n - k`` eigenvalues of ``M`` smallest in
    magnitude, and ``E`` is at least the norm of that residual.  So no
    subspace passes the test with fewer dimensions than the number of
    eigenvalues, largest in magnitude first, that cannot be left out.
    """
    magnitudes = np.sort(np.abs(np.linalg.eigvalsh(M)))
    tail = np.sqrt(np.cumsum(magnitudes**2))
    return int(np.count_nonzero(tail > tau))


def proximal_admm(
    sdp: SDP,
    *,
    c: float = 1.0,
    rho: float = 1.0,
    sigma: float = 0.5,
    delta: float = 1e-6,
    stride: int = 1,
    f_solver: str = "cg",
    projection: str = "lanczos",
    block_size: int = 8,
    f_share: float = 0.5,
    drop: float = 0.1,
    exact_tolerance: float | None = None,
    cholesky: bool = False,
    guard_factor: float = GUARD_FACTOR,
    max_outer: int = 50_000,
    time_limit: float | None = 600.0,
    seed: int = 0,
    diagnostics: bool = True,
) -> Trace:
    """Run Algorithm 3 on ``sdp`` and return its record.

    ``stride`` is the paper's ``s``; ``block_size`` is the smallest block of
    the Lanczos method (a restart may start from more retained Ritz
    vectors); ``f_share`` is the fraction of ``sigma P`` granted to the
    ``f``-residual before the projection is improved; ``drop`` scales the
    tolerance under which a restart discards small Ritz values.  ``seed``
    draws the random starting and fill directions of the Lanczos method.
    ``diagnostics`` computes, off the clock, the distance to the planted
    solution and ``rank_needed`` at every outer iteration.

    ``projection="one_sided"`` uses the one-sided pair of
    :mod:`sdp.projection`: the positive Ritz pairs give ``Z``, the rest of
    ``M`` gives ``B``, and the restart keeps the positive Ritz vectors and
    ``block_size`` guard vectors.  A pair that passes a test is accepted only
    if it is guarded and, with ``cholesky``, once a Cholesky factorization has
    certified ``B <= 0``; a pair that fails either is replaced by one from a
    larger subspace.  ``guard_factor`` is the safety factor of
    the guard's estimate.  For the one-sided projection the diagnostics record
    the largest eigenvalue of every accepted ``B``.
    """
    if not (c > 0.0 and rho > 0.0):
        raise ValueError("the doubly inexact method needs c > 0 and rho > 0")
    if f_solver not in ("cg", "exact"):
        raise ValueError("f_solver is cg or exact")
    if projection not in ("lanczos", "one_sided", "exact"):
        raise ValueError("projection is lanczos, one_sided, or exact")
    if not 0.0 < f_share < 1.0 or stride < 1 or block_size < 1:
        raise ValueError("need 0 < f_share < 1, stride >= 1, and block_size >= 1")
    n, m = sdp.n, sdp.m
    E = _Counted(sdp.E)
    rng = np.random.default_rng(seed)
    alpha = 1.0 + rho
    g_weight = math.sqrt(1.0 + 1.0 / rho)
    C_scaled = sdp.C / c
    exact = f_solver == "exact" and projection == "exact"
    factor = None
    if f_solver == "exact":
        # numpy's Cholesky rather than SciPy's: see "Known issue" in
        # numerical_experiments/README.md.
        factor = np.linalg.cholesky(sdp.E.gram() + alpha * np.eye(m)), True
    cg = ConjugateGradient(E, alpha, sdp.operator_norm())
    star = None
    if diagnostics and sdp.planted:
        # W* = (w_y*, w_Z*, sqrt(rho) t*) with w* = z* - b*/c and t* = x*.
        star = (sdp.b + sdp.nu_star / c, sdp.X_star + sdp.S_star / c, sdp.X_star)

    # Outer state from z_0 = (b, 0), p_0 = 0, t_0 = 0; inner state x = 0 and
    # the cone pair (0, 0).
    w_y = sdp.b.copy()
    w_Z = np.zeros((n, n))
    t = np.zeros((n, n))
    X = np.zeros((n, n))
    EX = np.zeros(m)
    pair = ConePair.zero(n)
    subspace_dim = 0

    trace = Trace()
    clock = Clock()
    lanczos: BlockLanczos | None = None
    version = 0  # increases whenever M = 2 X - w_Z changes
    built = -1  # the version the current projection was computed for
    max_checks = 20 * (n + m) + 1000
    stopped = ""
    while True:
        if star is not None:
            with clock.pause():
                trace.dist.append(
                    math.sqrt(
                        float(np.linalg.norm(w_y - star[0])) ** 2
                        + float(np.linalg.norm(w_Z - star[1])) ** 2
                        + rho * float(np.linalg.norm(t - star[2])) ** 2
                    )
                )
        else:
            trace.dist.append(nan)
        if f_solver == "exact":
            rhs = E.adjoint(w_y) + w_Z + rho * t - C_scaled
            X = (rhs - E.adjoint(scipy.linalg.cho_solve(factor, E.apply(rhs)))) / alpha
            X = (X + X.T) / 2.0
            EX = E.apply(X)
        # The residual rhs - H X of the new linear system at the warm start.
        R = E.adjoint(w_y - EX) + w_Z + rho * t - C_scaled - alpha * X
        cg.start(X, EX, R)
        version += 1
        fresh = True  # E(X) and R are computed from X, not by recurrence
        if exact:
            started = time.perf_counter()
            pair = ConePair.exact(2.0 * X - w_Z)
            trace.projection_seconds += time.perf_counter() - started
            trace.eigendecompositions += 1
            subspace_dim = n
            built = version
        cg_steps = matvecs = lanczos_steps = restarts = checks = 0
        guard_rejections = certificates = certificate_failures = 0
        while True:
            checks += 1
            now = _measure(X, EX, R, pair, w_Z, t, sdp.b, rho)
            tau = sigma * now.P if exact_tolerance is None else exact_tolerance
            if now.Q <= delta:
                verdict = "terminal"
            elif exact:
                verdict = "exact"
            elif now.E <= tau:
                verdict = "relative" if exact_tolerance is None else "absolute"
            else:
                verdict = ""
            if verdict and not fresh:
                # Confirm the decision with E(X) and the residual recomputed
                # from X; the recurrences of conjugate gradients drift.
                EX = E.apply(X)
                R = E.adjoint(w_y - EX) + w_Z + rho * t - C_scaled - alpha * X
                cg.start(X, EX, R)
                fresh = True
                continue
            uncertified = False
            if verdict and not pair.certified:
                # A one-sided pair is on the graph only if B <= 0: the guard
                # estimates it, and one Cholesky factorization decides it.
                if pair.bound > 0.0:
                    guard_rejections += 1
                    uncertified = True
                elif not cholesky:
                    pair.certified = True
                else:
                    certificates += 1
                    started = time.perf_counter()
                    passed = certify(pair)
                    trace.certificate_seconds += time.perf_counter() - started
                    if not passed:
                        certificate_failures += 1
                        uncertified = True
                if uncertified:
                    verdict = ""
            if verdict:
                break
            if time_limit is not None and clock.elapsed() >= time_limit:
                stopped = "time_limit"
                break
            if checks > max_checks:
                stopped = "stalled"
                break
            f_can = f_solver == "cg" and not cg.exhausted
            g_can = not (
                built == version
                and (
                    projection == "exact" or (lanczos is not None and lanczos.exhausted)
                )
            )
            project = g_can and (uncertified or not (f_can and now.e_f > f_share * tau))
            if f_can and not project:
                for _ in range(stride):
                    if cg.exhausted:
                        break
                    cg.step()
                    cg_steps += 1
                X, EX, R = cg.X, cg.EX, cg.R
                version += 1
                fresh = False
            elif project:
                started = time.perf_counter()
                M = 2.0 * X - w_Z
                if projection == "exact":
                    pair = ConePair.exact(M)
                    trace.eigendecompositions += 1
                    subspace_dim = n
                else:
                    if built != version or lanczos is None:
                        if projection == "one_sided":
                            start = restart_one_sided(pair)
                        else:
                            tolerance = drop * (1.0 - f_share) * tau / g_weight
                            start = restart_block(pair, tolerance, block_size, rng)
                        lanczos = BlockLanczos(M, start, rng)
                        restarts += 1
                    before = lanczos.matvecs
                    for _ in range(stride):
                        if lanczos.exhausted:
                            break
                        lanczos.step()
                        lanczos_steps += 1
                    matvecs += lanczos.matvecs - before
                    if projection == "one_sided":
                        margin = MARGIN if cholesky else 0.0
                        pair = lanczos.one_sided(block_size, margin, guard_factor)
                    else:
                        pair = lanczos.ritz()
                    subspace_dim = lanczos.dim
                built = version
                trace.projection_seconds += time.perf_counter() - started
            else:
                stopped = "stalled"
                break
        trace.elapsed.append(clock.elapsed())
        trace.Q.append(now.Q)
        trace.E.append(now.E)
        trace.P.append(now.P)
        trace.e_f.append(now.e_f)
        trace.e_g.append(now.e_g)
        trace.cg_steps.append(cg_steps)
        trace.matvecs.append(matvecs)
        trace.lanczos_steps.append(lanczos_steps)
        trace.restarts.append(restarts)
        trace.checks.append(checks)
        trace.subspace_dim.append(subspace_dim)
        trace.rank_positive.append(pair.rank_positive)
        trace.rank_negative.append(pair.rank_negative)
        trace.guard_rejections.append(guard_rejections)
        trace.certificates.append(certificates)
        trace.certificate_failures.append(certificate_failures)
        trace.accepted_by.append(verdict or stopped)
        with clock.pause():
            trace.objective.append(float(np.vdot(sdp.C, pair.Z)))
            if diagnostics and projection == "one_sided":
                # Relative to the larger of ||B|| and ||Z||, so that a B of
                # rounding errors next to a large Z does not count.
                eigenvalues = np.linalg.eigvalsh(pair.B)
                norm = max(
                    abs(eigenvalues[0]),
                    abs(eigenvalues[-1]),
                    float(np.max(pair.theta, initial=0.0)),
                    1e-300,
                )
                trace.violation.append(float(eigenvalues[-1]) / norm)
                trace.rank_needed.append(-1)
            elif diagnostics:
                trace.violation.append(nan)
                trace.rank_needed.append(_rank_needed(2.0 * X - w_Z, tau))
            else:
                trace.violation.append(nan)
                trace.rank_needed.append(-1)
        if verdict == "terminal":
            trace.converged = True
            trace.stopped_by = "converged"
            break
        if stopped:
            trace.stopped_by = stopped
            break
        if len(trace.Q) > max_outer:
            trace.stopped_by = "max_outer"
            break
        w_y = w_y - EX + sdp.b
        w_Z = w_Z - X + pair.Z
        t = t - now.q / rho
    trace.seconds = clock.elapsed()
    trace.forward = E.forward
    trace.adjoint = E.adjoint_calls
    trace.X, trace.Z, trace.B, trace.EX = X, pair.Z, pair.B, EX
    # The y part of the returned subgradient is b_y = c (2 E(X) - w_y - b);
    # the multiplier of E(X) = b in the usual sign is nu = -b_y.
    trace.nu = c * (w_y + sdp.b - 2.0 * EX)
    return trace


def certificate(sdp: SDP, trace: Trace, c: float, rho: float, delta: float) -> dict:
    """What the returned point certifies, measured independently of the run.

    The graph inclusions of Algorithm 3 (``Z`` in the cone, ``c B`` in its
    normal cone there), the two bounds of Proposition 5.2 at termination, and
    the optimality conditions of the SDP for the primal point ``Z`` and the
    dual point ``(nu, S)`` with ``S = -c B``.
    """
    X, Z, B, nu = trace.X, trace.Z, trace.B, trace.nu
    S = -c * B
    stationarity = sdp.C - sdp.E.adjoint(nu) - S  # a + E'b of the paper
    coupling = math.hypot(
        float(np.linalg.norm(trace.EX - sdp.b)), float(np.linalg.norm(X - Z))
    )
    operator_norm = math.sqrt(1.0 + sdp.operator_norm() ** 2)
    stationarity_bound = c * delta * (operator_norm + math.sqrt(rho))
    scale = max(float(np.linalg.norm(Z)) * float(np.linalg.norm(B)), 1e-300)
    objective = float(np.vdot(sdp.C, Z))
    dual_objective = float(sdp.b @ nu)
    return dict(
        # Inclusions: both should be zero up to rounding error.
        min_eigenvalue_Z=float(np.linalg.eigvalsh(Z)[0]),
        max_eigenvalue_B=float(np.linalg.eigvalsh(B)[-1]),
        complementarity=float(np.linalg.norm(Z @ B)) / scale,
        # Proposition 5.2: ||(E(X), X) - (b, Z)|| <= delta and
        # ||a + E'b|| <= c delta (||E|| + sqrt(rho)) at termination.
        coupling_residual=coupling,
        stationarity_residual=float(np.linalg.norm(stationarity)),
        stationarity_bound=stationarity_bound,
        # The SDP itself, at the feasible-cone point Z.
        primal_feasibility=float(np.linalg.norm(sdp.E.apply(Z) - sdp.b)),
        objective=objective,
        dual_objective=dual_objective,
        duality_gap=objective - dual_objective,
        solution_error=(
            float(np.linalg.norm(Z - sdp.X_star)) if sdp.planted else float("nan")
        ),
    )


def theory_checks(trace: Trace, sigma: float, delta: float) -> dict:
    """The outer guarantees of Section 3 against the planted solution.

    For every accepted update the one-step decrease of
    Lemma ``one-step-decrease-inequality`` reads

        ||W_{k+1} - W*||^2 <= ||W_k - W*||^2 - (1 - s)/(1 + s) Q_k^2,

    and the update bound of Lemma ``exact-inexact-graph-pair-inequalities``
    reads ``(1 - s) P_k <= Q_k <= (1 + s) P_k``; ``s`` is ``sigma`` for an
    update accepted by the relative test and the achieved ratio ``E_k / P_k``
    otherwise.  The reported violations are the largest amounts by which an
    inequality fails (at most rounding error if the theory holds), the first
    relative to ``||W_0 - W*||^2`` and the second relative to ``P_k``.
    """
    accepted = len(trace.Q) - 1 if trace.converged else len(trace.Q)
    accepted = min(accepted, len(trace.dist) - 1)
    decrease = ratio = 0.0
    total = 0.0
    largest_sigma = 0.0
    for k in range(accepted):
        if trace.accepted_by[k] == "relative":
            s = sigma
        else:
            s = trace.E[k] / trace.P[k] if trace.P[k] > 0.0 else 0.0
        largest_sigma = max(largest_sigma, s)
        total += trace.Q[k] ** 2
        ratio = max(
            ratio,
            ((1.0 - s) * trace.P[k] - trace.Q[k]) / trace.P[k],
            (trace.Q[k] - (1.0 + s) * trace.P[k]) / trace.P[k],
        )
        if not math.isnan(trace.dist[0]):
            gap = (
                trace.dist[k + 1] ** 2
                - trace.dist[k] ** 2
                + (1.0 - s) / (1.0 + s) * trace.Q[k] ** 2
            )
            decrease = max(decrease, gap / trace.dist[0] ** 2)
    dist0 = trace.dist[0] if trace.dist else nan
    bound = nan
    k_plus = nan
    if not math.isnan(dist0) and largest_sigma < 1.0:
        bound = (1.0 + largest_sigma) / (1.0 - largest_sigma) * dist0**2
        k_plus = math.floor(bound / delta**2)
    return dict(
        initial_distance=dist0,
        decrease_violation=decrease if not math.isnan(dist0) else nan,
        ratio_violation=ratio,
        sum_Q_squared=total,
        sum_Q_squared_bound=bound,
        K_delta_plus=k_plus,
    )
