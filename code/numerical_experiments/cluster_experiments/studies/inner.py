"""Inner solvers of the ADMM ``x``-subproblem, their constants, and their costs.

The ``x``-subproblem of ADMM is the linear system ``(H + cI) x = c w - q``
with ``H`` the Hessian of the quadratic (``Q`` for the QP, ``[A, -A]'[A, -A]``
for the split LASSO).  Every inner solver of the cluster experiments is
available through one interface (``start`` from a point whose residual is
known, ``step``, ``solution``, and the ``full`` flag of a GMRES cycle):

``gmres``
    The original study's GMRES (``numerical_experiments.admm.Gmres``):
    classical Gram-Schmidt with one reorthogonalization, Givens rotations,
    and a restart at Krylov dimension 200, with the constants
    ``theory.gmres_constants``.  The basis grows on demand, so a short cycle
    does not hold ``201`` vectors of length ``n``.
``gd``, ``agd``, ``cg``, ``nystrom_pcg``, ``block_kaczmarz``, ``cyclic_kaczmarz``
    The sealed engine's ``Inner`` (one step per call, setup timed once per
    instance), started from the known residual ``-c e`` instead of
    recomputing it, as the original GMRES is.  The engine's verified
    contraction constants are used where they exist (``gd``, ``cg``, and the
    expected-value bound of ``block_kaczmarz``); ``agd``, ``nystrom_pcg``, and
    ``cyclic_kaczmarz`` have none, so ``sigma*`` and ``s*`` are not defined
    for them.

``calibrate`` measures the costs ``c_in`` of one inner iteration and
``c_test`` of one check behind ``s*`` exactly as the original studies do.
"""

from __future__ import annotations

import copy
import math
import time
from collections import OrderedDict

import numpy as np
import scipy.linalg

from releases.regression_v9.experiments import theory, tuning
from releases.regression_v9.experiments.inner import Inner

Array = np.ndarray

# Krylov dimension at which GMRES restarts, as in the original ADMM study.
RESTART = 200
# Krylov dimension at which the costs of a GMRES step and of a check are timed
# for ``s*``: about the inner iterations of one outer iteration.  The other
# solvers are timed on runs of this many steps.
CALIBRATION_DEPTH = 10
# Timings of each cost behind ``s*`` (``--calibration-repetitions``); the
# original studies took 300, and studies_v1 to studies_v3 took 100.
CALIBRATION_REPETITIONS = 30
METHODS = (
    "gmres",
    "gd",
    "agd",
    "cg",
    "nystrom_pcg",
    "block_kaczmarz",
    "cyclic_kaczmarz",
)
nan = float("nan")


class Gmres:
    """GMRES on ``(H + cI) x = rhs``, one Arnoldi step at a time.

    ``apply`` maps ``v`` to ``(H + cI) v``.  ``start`` begins a cycle at ``x0``
    given its residual ``rhs - (H + cI) x0``, which the caller already has, so
    ``solution`` costs one triangular solve and one product with the basis.
    """

    def __init__(self, apply, n: int, restart: int = RESTART):
        self.apply = apply
        self.n = n
        self.restart = restart
        self.V = np.empty((min(restart + 1, 16), n))
        self.R = np.zeros((restart + 1, restart))
        self.cs = np.empty(restart)
        self.sn = np.empty(restart)
        self.g = np.zeros(restart + 1)
        self.products = 0
        self.j = 0
        self.exhausted = True
        self.x0 = np.zeros(n)

    def start(self, x0: Array, rhs: Array | None, residual: Array) -> None:
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

    def step(self, hx: Array | None = None) -> None:
        """One Arnoldi step: one product with ``H + cI`` and the orthogonalization."""
        j = self.j
        V = self.V[: j + 1]
        u = self.apply(V[j])
        self.products += 1
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
            self._reserve(j + 2)
            self.V[j + 1] = u / norm

    def _reserve(self, rows: int) -> None:
        """Room for ``rows`` basis vectors, keeping the ``j`` already built."""
        if rows <= len(self.V):
            return
        size = min(self.restart + 1, max(rows, 2 * len(self.V)))
        grown = np.empty((size, self.n))
        grown[: self.j] = self.V[: self.j]
        self.V = grown

    def solution(self) -> Array:
        """The iterate minimizing the residual over ``x0`` plus the Krylov space."""
        if self.j == 0:
            return self.x0
        y = scipy.linalg.solve_triangular(self.R[: self.j, : self.j], self.g[: self.j])
        return self.x0 + y @ self.V[: self.j]


class EngineInner:
    """The engine's one-step solvers behind the interface of :class:`Gmres`.

    ``start`` installs the known residual where the engine's own ``start``
    would recompute it with one more product; the rest of its state is set as
    the engine sets it.  ``step`` hands gradient descent the product ``Hx``
    when the caller has it from the latest check, as the engine's solver does.
    """

    full = False

    def __init__(self, engine: Inner):
        self.engine = engine

    @property
    def products(self) -> int:
        return self.engine.products

    def start(self, x0: Array, rhs: Array, residual: Array) -> None:
        engine = self.engine
        engine.x = np.array(x0, dtype=float)
        engine.rhs = rhs
        engine.r = np.array(residual, dtype=float)
        engine.direction = engine.precondition(engine.r)
        engine.rz = float(engine.r @ engine.direction)
        engine.previous = engine.x.copy()
        engine.steps = 0

    def step(self, hx: Array | None = None) -> None:
        self.engine.step(q_x=hx if self.engine.method == "gd" else None)

    def solution(self) -> Array:
        return self.engine.x.copy()


class InnerFactory:
    """Sets up the inner solver of an instance once and hands out fresh copies.

    The original studies share an instance's setup among its configurations,
    and so does this: the Nystrom sketch and the Kaczmarz permutation are built
    once (``setup_seconds``), and every run starts from the same post-setup
    state, random stream and block cache included, so the runs of an
    instance differ only in their parameters.
    """

    def __init__(self, instance, c: float, settings: dict):
        self.method = settings["inner"]
        self.n = instance.n
        problem = instance.problem
        started = time.perf_counter()
        if self.method == "gmres":
            self.restart = int(settings.get("restart", RESTART))
            self.apply = lambda v: problem.hess(v) + c * v
            self.engine = None
        else:
            # The engine's stream for inner solvers, seeded as in its runs.
            rng = np.random.default_rng(instance.seed + 100_003)
            self.engine = Inner(
                problem,
                c,
                self.method,
                rng,
                int(settings.get("block_size", 32)),
                int(settings.get("rank", 64)),
                gd_factor=float(settings.get("gd_factor", 1.0)),
            )
            self.state = (
                copy.deepcopy(rng.bit_generator.state),
                OrderedDict(self.engine.cache),
                (
                    self.engine.products,
                    self.engine.row_updates,
                    self.engine.block_factorizations,
                ),
            )
        self.setup_seconds = time.perf_counter() - started

    def new(self, restart: int | None = None):
        """A solver in its post-setup state."""
        if self.engine is None:
            return Gmres(self.apply, self.n, restart or self.restart)
        rng_state, cache, counts = self.state
        self.engine.rng.bit_generator.state = copy.deepcopy(rng_state)
        self.engine.cache = OrderedDict(cache)
        (
            self.engine.products,
            self.engine.row_updates,
            self.engine.block_factorizations,
        ) = counts
        return EngineInner(self.engine)


def contraction(instance, c: float, ratio: float, settings: dict):
    """``(alpha, C_0, kind)`` of the inner-contraction assumption of the paper.

    ``kind`` is ``"deterministic"`` when ``||e^l||^2 <= C_0 alpha^l ||e^0||^2``
    holds for every run, ``"expected"`` when it holds in expectation (block
    Kaczmarz), and ``"none"`` when no verified constants exist, in which case
    ``alpha`` and ``C_0`` are ``nan``.
    """
    if instance.algorithm == "ppm":
        return _ppm_constants(ratio, float(settings.get("gd_factor", 1.0)))
    method = settings["inner"]
    if method == "gmres":
        alpha, c0 = theory.gmres_constants(ratio)
        return alpha, c0, "deterministic"
    bound = tuning.constants(
        instance.problem,
        c,
        method,
        int(settings.get("block_size", 32)),
        1.0,
        float(settings.get("gd_factor", 1.0)),
    )
    if bound is None:
        return nan, nan, "none"
    alpha, c0, deterministic = bound
    # As in the engine: keep log(1/alpha) finite and positive.
    alpha = min(max(alpha, 1e-15), 1.0 - 1e-15)
    return alpha, c0, "deterministic" if deterministic else "expected"


def _ppm_constants(ratio: float, gd_factor: float):
    """The constants of the PPM inner step ``x <- x - coefficient e``.

    With ``e = gamma grad phi_k(x)`` and ``r = gamma L``, the step contracts
    ``x - x_k*`` by ``rho = max(|1 - t|, |1 - t (1 + r)|)``, ``t = 2 gd_factor /
    (r + 2)``, over the spectrum ``[1, 1 + r]`` of ``gamma Hess phi_k`` (``f``
    convex), and ``||e^l|| <= (1 + r) rho^l ||e^0||``.  At ``gd_factor = 1``
    this is ``theory.inner_constants`` (Proposition 5.1); a step that does not
    contract has no constants.
    """
    if gd_factor == 1.0:
        alpha, c0 = theory.inner_constants(ratio)
        return alpha, c0, "deterministic"
    step = 2.0 * gd_factor / (ratio + 2.0)
    rho = max(abs(1.0 - step), abs(1.0 - step * (1.0 + ratio)))
    if not 0.0 < rho < 1.0:
        return nan, nan, "none"
    return rho**2, (1.0 + ratio) ** 2, "deterministic"


def stride_scan_limit(b_star: float) -> int:
    """The largest ``s`` that Remark 4.4's scan for its integer minimizer covers.

    1000, as in the original studies, unless Lemma 4.2's ``s*`` is large
    (Kaczmarz, whose ``alpha`` is within ``1e-5`` of one): the unrelaxed
    minimizer lies near ``1.4 s*`` there, so the scan covers ``4 s*``, up to
    ``10^6``, rather than report its own end point as the minimizer.
    """
    if not math.isfinite(b_star):
        return 1000
    return int(min(max(1000, math.ceil(4.0 * b_star)), 1_000_000))


def calibrate(
    instance,
    c: float,
    gamma: float,
    factory: InnerFactory | None,
    repetitions: int = CALIBRATION_REPETITIONS,
    depth: int = CALIBRATION_DEPTH,
) -> tuple[float, float, float]:
    """Median seconds of one inner iteration and of one check (Lemma 4.2).

    Also returns the seconds the timed operations took, the cost of the
    measurements that ``s*`` needs.  PPM: an inner iteration is a gradient
    evaluation and the residual, a check the three norms of the tests.  ADMM:
    a check forms the iterate, evaluates the gradient there, projects, and
    takes the norms of the tests, and an inner iteration is one step of the
    inner solver.  A GMRES step costs more the larger its Krylov space, so it
    is timed at dimension ``depth``, after ``depth - 1`` untimed steps, as in
    the original studies.  The step of any other solver costs the same
    wherever it falls, so every step of runs of ``depth`` steps is timed, each
    followed by a timed check, with no untimed steps.  Each cost is sampled
    ``repetitions + 3`` times, the first three discarded as warm-up.
    """
    rng = np.random.default_rng(12345)
    n = instance.n
    v = rng.standard_normal(n)
    w = rng.standard_normal(n)
    inner, check = [], []
    samples = repetitions + 3

    def result() -> tuple[float, float, float]:
        return (
            float(np.median(inner[3:])),
            float(np.median(check[3:])),
            float(sum(inner) + sum(check)),
        )

    if instance.algorithm == "ppm":
        for _ in range(samples):
            started = time.perf_counter()
            g = instance.gradient(v)
            e = gamma * g + (v - w)
            inner.append(time.perf_counter() - started)
            started = time.perf_counter()
            float(np.linalg.norm(g))
            float(np.linalg.norm(v - w))
            float(np.linalg.norm(e))
            check.append(time.perf_counter() - started)
        return result()

    def timed_step(solver) -> None:
        started = time.perf_counter()
        solver.step()
        inner.append(time.perf_counter() - started)

    def timed_check(solver) -> None:
        started = time.perf_counter()
        x = solver.solution()
        g = instance.gradient(x)
        e = x + g / c - w
        z = instance.project(x - g / c)
        float(np.linalg.norm(x - z))
        float(np.linalg.norm(z + g / c - w))
        float(np.linalg.norm(e))
        check.append(time.perf_counter() - started)

    if factory.method != "gmres":
        solver = factory.new()
        while len(inner) < samples:
            # Any residual will do: only the costs are measured.  A fresh start
            # every ``depth`` steps keeps CG-like solvers from converging into
            # steps that do nothing.
            solver.start(v, w, w)
            for _ in range(min(depth, samples - len(inner))):
                timed_step(solver)
                timed_check(solver)
        return result()
    solver = factory.new(restart=depth + 1)

    def advance():
        # A tiny test problem can exhaust a GMRES cycle (a happy breakdown)
        # within ``depth`` steps; start a fresh one rather than step past it.
        if solver.full:
            solver.start(v, w, w)
        solver.step()

    for _ in range(samples):
        solver.start(v, w, w)
        for _ in range(depth - 1):
            advance()
        if solver.full:
            solver.start(v, w, w)
        timed_step(solver)
        timed_check(solver)
    return result()


def recommendations(
    alpha: float,
    c0: float,
    kind: str,
    settings: dict,
    c_in: float,
    c_test: float,
) -> dict:
    """``sigma*`` (Proposition 5.1) and ``s*`` (Lemma 4.2, Remark 4.4).

    As in the original studies: ``sigma*`` uses the deterministic constant
    ``chi = s log(1/alpha)`` of Theorem 4.3, except for a solver whose
    contraction holds only in expectation, where it uses the Theorem 4.2
    constant ``chi = s log(1/alpha) / (1 - alpha^s)``;
    ``sigma_star_expected`` is always the latter.  Without constants every
    entry is ``nan`` (``b_best`` and ``b_star_exact`` are ``None``).
    """
    b_sigma = int(settings["sigma_sweep_b"])
    sigma_b = float(settings["b_sweep_sigma"])
    if kind == "none":
        return dict(
            sigma_star=nan,
            sigma_star_expected=nan,
            sigma_upper_bound=nan,
            b_star=nan,
            b_best=None,
            b_star_exact=None,
        )
    deterministic = kind == "deterministic"
    b_star = theory.b_star_asymptotic(c_test, c_in, sigma_b, c0, alpha)
    return dict(
        sigma_star=theory.optimal_sigma(alpha, c0, b_sigma, deterministic),
        sigma_star_expected=theory.optimal_sigma(
            alpha, c0, b_sigma, deterministic=False
        ),
        sigma_upper_bound=theory.sigma_upper_bound(alpha, c0, b_sigma, deterministic),
        b_star=b_star,
        b_best=theory.best_integer_b(
            b_star,
            lambda b: theory.b_cost_per_outer(b, c_test, c_in, sigma_b, c0, alpha),
        ),
        b_star_exact=theory.optimal_b_exact(
            c_test, c_in, sigma_b, c0, alpha, b_max=stride_scan_limit(b_star)
        ),
    )


def sigma_star_by_b(alpha: float, c0: float, kind: str, bs) -> dict[int, float]:
    """``sigma*`` for every stride of the grid (``sigma_star_b<s>`` columns)."""
    if kind == "none":
        return {int(b): nan for b in bs}
    deterministic = kind == "deterministic"
    return {int(b): theory.optimal_sigma(alpha, c0, int(b), deterministic) for b in bs}


def finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)
