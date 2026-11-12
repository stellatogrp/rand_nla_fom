"""The cluster problem families seen through the original studies.

``Instance`` wraps the sealed engine's ``Problem`` with what the original
studies record about it: which algorithm runs (ADMM on the equality QP and on
the split LASSO, the proximal point method on ridge logistic regression), the
objective and the feasibility measure of ``histories.csv``, the smoothness
constant by Lanczos, the exact ``x``-update of ADMM, and a ground truth
together with its accuracy and time.

The engine's problems are unchanged; only two things are added on top of it.

``smoothness``
    The original studies set ``c = L / ratio`` (ADMM) or ``gamma = ratio / L``
    (PPM) from ``L`` computed by Lanczos.  The engine instead carries a
    certified upper bound for the regression designs (``||A||_1 ||A||_inf``
    or ``||A||_F^2``), which can be several times larger; it is kept as
    ``smoothness_certified``.  ``use_smoothness`` replaces the engine's ``L``
    by the Lanczos value, so the inner solvers use it too.
``exact_solver``
    The analog of the Cholesky factor of ``Q + cI`` in the original ADMM
    study.  The QP Hessian ``Q = I/kappa + (1 - 1/kappa) L_3D / 12`` is
    circulant on the periodic grid, so ``(Q + cI)^{-1}`` is applied exactly
    by the three-dimensional FFT.  The split LASSO Hessian ``H = B'B`` with
    ``B = [A, -A]`` decouples in ``t = u - v`` and ``s = u + v``: the system
    ``(H + cI) x = r`` is ``(2A'A + cI) t = r_u - r_v`` and ``c s = r_u + r_v``,
    solved by a Cholesky factor of the smaller Gram matrix when it fits
    (``--exact-direct-max``) and by conjugate gradients to ``rtol = 1e-13``
    otherwise (recorded as ``exact_solver``).

The ground truth (``truth_*`` columns of ``parameters.csv``) is a reference
solve, as CVXPY with Clarabel is in the original studies:

* QP: conjugate gradients on the reduced KKT system, the equality-fixed
  coordinates eliminated, compared with the planted KKT point
  (``truth_planted_error``);
* LASSO: FISTA with adaptive restart, then a polish on the detected support
  and signs (conjugate gradients on the normal equations), or CVXPY with
  Clarabel (``--truth-solver clarabel``) for small designs;
* logistic: Newton-CG with an Armijo line search, or CVXPY with Clarabel.

``truth_stationarity`` is the norm of the reduced gradient (QP), the natural
residual ``||b - soft(b - grad, lambda)||`` (LASSO), or the norm of the
gradient (logistic) at the reference point, so the accuracy of every
reference is on record.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import expit

from releases.regression_v9.experiments.dense import DenseDesign
from releases.regression_v9.experiments.problems import build

Array = np.ndarray

# The algorithm of the paper that each family runs.
FAMILIES = {"qp": "admm", "lasso": "admm", "logistic": "ppm"}
# The reference tolerances of the original ``solve_reference``, tightest first.
REFERENCE_TOLERANCES = (1e-12, 1e-10, 1e-8)
# Relative residual of the conjugate gradient solves that stand in for an
# exact solve (exact LASSO x-update, polish): about the attainable floor.
EXACT_CG_TOLERANCE = 1e-13

nan = float("nan")


@dataclass
class GroundTruth:
    """A reference solution together with how accurate and how expensive it was."""

    x: Array
    value: float
    status: str
    tolerance: float
    seconds: float
    stationarity: float
    feasibility: float
    planted_error: float
    solver: str


def engine_config(family: str, seed: int, settings: dict) -> dict:
    """The task configuration the sealed engine's ``build`` expects."""
    if family == "qp":
        return dict(
            family="qp",
            seed=seed,
            side=int(settings["side"]),
            kappa=float(settings["kappa"]),
        )
    return dict(
        family=family,
        seed=seed,
        dataset=settings["dataset"],
        orientation=settings["orientation"],
        matrix_source=settings["matrix_source"],
        lambda_ratio=float(settings.get("lambda_ratio", 0.01)),
    )


def _norm(value) -> float:
    return float(np.linalg.norm(value))


def _gram(A, v: Array) -> Array:
    """``A'A v`` with one pass over a dense design, or two products otherwise."""
    if isinstance(A, DenseDesign):
        return A.gram_product(v)
    return A.T @ (A @ v)


def _start(size: int) -> Array:
    """The fixed start vector of every Lanczos run."""
    return np.random.default_rng(20_261_105).standard_normal(size)


def _explicit(A):
    """The design as a SciPy sparse matrix (dense designs only when small)."""
    if isinstance(A, DenseDesign):
        return A.to_sparse()
    return sp.csr_matrix(A)


class Instance:
    """One drawn instance: the engine's problem plus the study's view of it."""

    def __init__(self, family: str, seed: int, settings: dict):
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}")
        self.family = family
        self.algorithm = FAMILIES[family]
        self.seed = int(seed)
        self.settings = settings
        self.problem = build(
            engine_config(family, seed, settings), settings.get("data_dir", "data")
        )
        p = self.problem
        self.n = int(p.n)
        self.m = int(len(p.indices)) if family == "qp" else int(p.A.shape[0])
        self.smoothness_certified = float(p.L)

    # -- the functions of histories.csv --------------------------------------

    def gradient(self, x: Array) -> Array:
        return self.problem.gradient(x)

    def objective(self, x: Array) -> float:
        return float(self.problem.objective(x))

    def project(self, v: Array) -> Array:
        """The ``z``-update of ADMM: projection onto the constraint set."""
        return self.problem.project(v)

    def feasibility(self, x: Array) -> float:
        """The constraint violation of an ADMM iterate ``x_k``.

        ``||x_I - t||`` for the coordinate equalities ``x_I = t`` of the QP
        (the original QP records ``||A x_k - b||``) and ``||min(x, 0)||`` for
        the nonnegativity of the split LASSO; ``nan`` for logistic
        regression, which is unconstrained, as for the original Huber study.
        """
        p = self.problem
        if self.family == "qp":
            return _norm(x[p.indices] - p.target)
        if self.family == "lasso":
            return _norm(np.minimum(x, 0.0))
        return nan

    # -- setup ---------------------------------------------------------------

    def smoothness(self) -> float:
        """``L`` by Lanczos, as in the original studies (not yet in use).

        ARPACK starts from a fixed seeded vector, so every process computes the
        same ``L`` to the last bit (its default start changes between calls).
        """
        p = self.problem
        if self.family == "qp":
            value = spla.eigsh(
                p.Q,
                k=1,
                which="LA",
                v0=_start(self.n),
                return_eigenvectors=False,
            )
            return float(value[0])
        sigma = self._largest_singular_value()
        if self.family == "lasso":
            # H = B'B with B = [A, -A], so ||H|| = ||B||^2 = 2 ||A||^2.
            return 2.0 * sigma**2
        # The logistic curvature is at most 1/4 per observation.
        return sigma**2 / (4.0 * p.A.shape[0]) + p.regularization

    def _largest_singular_value(self) -> float:
        A = self.problem.A
        rows, cols = A.shape
        if min(rows, cols) <= 2:
            dense = _explicit(A).toarray()
            return float(np.linalg.norm(dense, 2))
        operator = spla.LinearOperator(
            (rows, cols),
            matvec=lambda v: A @ np.ravel(v),
            rmatvec=lambda v: A.T @ np.ravel(v),
            dtype=float,
        )
        value = spla.svds(
            operator, k=1, v0=_start(min(rows, cols)), return_singular_vectors=False
        )
        return float(value[0])

    def use_smoothness(self, value: float) -> None:
        """Make ``value`` the ``L`` of the problem, for the inner solvers too."""
        if not math.isfinite(value) or value <= 0:
            raise ValueError("the smoothness constant must be finite and positive")
        self.problem.L = float(value)

    def exact_solver(self, c: float, direct_max: int = 5_000):
        """The exact ``x``-update ``(H + cI)^{-1}`` of ADMM, or ``None`` for PPM."""
        if self.family == "qp":
            return FourierSolver(
                int(self.settings["side"]), float(self.settings["kappa"]), c
            )
        if self.family == "lasso":
            return SplitLassoSolver(self.problem.A, c, direct_max)
        return None

    # -- ground truth ----------------------------------------------------------

    def ground_truth(self, settings: dict, smoothness: float | None = None):
        """The reference solution; ``smoothness`` sets the LASSO reference's step.

        Without it the step comes from the engine's certified bound, which is
        safe but can be far smaller than ``1/L`` on dense designs.
        """
        solver = settings.get("truth_solver", "iterative")
        tolerance = float(settings.get("truth_tolerance", REFERENCE_TOLERANCES[0]))
        time_limit = float(settings.get("truth_time_limit", 3600.0))
        iterations = int(settings.get("truth_max_iterations", 100_000))
        if self.family == "qp":
            return _qp_truth(self.problem, tolerance, iterations)
        if solver == "clarabel":
            return _clarabel_truth(self)
        if self.family == "lasso":
            # The split Hessian has norm 2 ||A||^2; FISTA on the unsplit LASSO
            # needs ||A||^2, with a margin for the rounding of the estimate.
            design = 0.5 * (self.problem.L if smoothness is None else smoothness)
            return _lasso_truth(
                self.problem, design * (1.0 + 1e-8), tolerance, iterations, time_limit
            )
        return _logistic_truth(self.problem, tolerance, iterations, time_limit)

    def stationarity(self, x: Array) -> float:
        """The first-order optimality residual used for ``truth_stationarity``."""
        p = self.problem
        if self.family == "qp":
            free = np.ones(self.n, dtype=bool)
            free[p.indices] = False
            return _norm(self.gradient(x)[free])
        if self.family == "lasso":
            d = p.A.shape[1]
            beta = x[:d] - x[d:]
            return _lasso_subgradient(beta, _gram(p.A, beta) - p.A.T @ p.target, p)
        return _norm(self.gradient(x))


def exact_residual(instance: Instance, exact, c: float, w: Array) -> float:
    """``||R(w)||`` for the ADMM operator: the exact step's ``||x - z||`` at ``w``.

    The same computation as ``admm.exact_residual`` of the original studies,
    with the Cholesky solve replaced by ``exact.solve``.
    """
    x = exact.solve(c * w - instance.problem.q)
    a = instance.gradient(x)
    z = instance.project(x - a / c)
    return _norm(x - z)


# ----------------------------------------------------------------------------
# Exact x-updates.


class FourierSolver:
    """``(Q + cI)^{-1}`` for the periodic-grid QP by the real three-dimensional FFT.

    The one-dimensional periodic second difference has eigenvalues
    ``2 - 2 cos(2 pi k / side)``; ``L_3D`` sums them over the three axes of
    the ``side^3`` grid in C order, which is how the engine's Kronecker
    products lay it out.
    """

    kind = "fft"

    def __init__(self, side: int, kappa: float, c: float):
        frequencies = 2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(side) / side)
        half = frequencies[: side // 2 + 1]
        laplacian = (
            frequencies[:, None, None]
            + frequencies[None, :, None]
            + half[None, None, :]
        )
        self.shape = (side, side, side)
        self.c = c
        self.eigenvalues = 1.0 / kappa + (1.0 - 1.0 / kappa) * laplacian / 12.0 + c

    def solve(self, rhs: Array) -> Array:
        transform = np.fft.rfftn(np.reshape(rhs, self.shape))
        return np.fft.irfftn(
            transform / self.eigenvalues, s=self.shape, axes=(0, 1, 2)
        ).ravel()


class SplitLassoSolver:
    """``(H + cI)^{-1}`` for the split LASSO Hessian ``H = [A, -A]'[A, -A]``."""

    def __init__(self, A, c: float, direct_max: int = 5_000):
        self.A = A
        self.c = c
        rows, cols = A.shape
        self.d = cols
        self.factor = None
        self.last = None
        self.kind = "cg"
        in_memory = not isinstance(A, DenseDesign) or A._resident is not None
        if in_memory and min(rows, cols) <= direct_max:
            try:
                self._factor(A, rows, cols)
            except MemoryError:
                self.factor = None
                self.kind = "cg"
        if self.factor is None:
            self.operator = spla.LinearOperator(
                (cols, cols),
                matvec=lambda v: 2.0 * _gram(A, np.ravel(v)) + c * np.ravel(v),
            )

    def _factor(self, A, rows: int, cols: int) -> None:
        explicit = _explicit(A)
        # numpy's Cholesky, as in the original studies: SciPy's routes to
        # Accelerate's dpotrf with uplo='U', which corrupts memory on macOS
        # 27.0 and 27.1 (see the original README, "Known issue").
        if cols <= rows:
            gram = (explicit.T @ explicit).toarray()
            matrix = 2.0 * gram + self.c * np.eye(cols)
            kind = "cholesky"
        else:
            gram = (explicit @ explicit.T).toarray()
            matrix = self.c * np.eye(rows) + 2.0 * gram
            kind = "cholesky_woodbury"
        self.factor = (np.asfortranarray(np.linalg.cholesky(matrix)), True)
        self.explicit = explicit
        self.kind = kind

    def _apply(self, r: Array) -> Array:
        """``(2A'A + cI)^{-1} r``."""
        if self.kind == "cholesky":
            return scipy.linalg.cho_solve(self.factor, r)
        if self.kind == "cholesky_woodbury":
            # (2A'A + cI)^{-1} = (I - 2A'(cI + 2AA')^{-1} A) / c.
            inner = scipy.linalg.cho_solve(self.factor, self.explicit @ r)
            return (r - 2.0 * (self.explicit.T @ inner)) / self.c
        solution, info = spla.cg(
            self.operator,
            r,
            x0=self.last,
            rtol=EXACT_CG_TOLERANCE,
            atol=0.0,
            maxiter=100_000,
        )
        if info != 0:
            warnings.warn(
                "exact LASSO x-update: CG did not reach its tolerance", stacklevel=2
            )
        self.last = solution
        return solution

    def solve(self, rhs: Array) -> Array:
        d = self.d
        upper, lower = rhs[:d], rhs[d:]
        t = self._apply(upper - lower)
        s = (upper + lower) / self.c
        return np.concatenate(((s + t) / 2.0, (s - t) / 2.0))


# ----------------------------------------------------------------------------
# Ground truths.


def _qp_truth(p, tolerance: float, iterations: int) -> GroundTruth:
    """Conjugate gradients on the reduced KKT system of the equality QP."""
    started = time.perf_counter()
    free = np.ones(p.n, dtype=bool)
    free[p.indices] = False
    Q = sp.csr_matrix(p.Q)
    rows = Q[free]
    reduced = rows[:, free]
    rhs = -p.q[free] - rows[:, ~free] @ p.target
    x = np.empty(p.n)
    x[p.indices] = p.target
    solution, info = spla.cg(reduced, rhs, rtol=tolerance, atol=0.0, maxiter=iterations)
    x[free] = solution
    seconds = time.perf_counter() - started
    gradient = Q @ x + p.q
    achieved = _norm(reduced @ solution - rhs) / _norm(rhs)
    return GroundTruth(
        x=x,
        value=float(0.5 * x @ (Q @ x) + p.q @ x),
        status="optimal" if info == 0 else "optimal_inaccurate",
        tolerance=tolerance if info == 0 else achieved,
        seconds=seconds,
        stationarity=_norm(gradient[free]),
        feasibility=_norm(x[p.indices] - p.target),
        planted_error=_norm(x - p.truth),
        solver="kkt_cg",
    )


def _soft(v: Array, threshold: float) -> Array:
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def _lasso_subgradient(beta: Array, gradient: Array, p) -> float:
    """The natural residual ``||b - soft(b - grad, lambda)||`` of the LASSO.

    It vanishes exactly at the minimizers of ``1/2||A b - y||^2 + lambda ||b||_1``
    and, unlike the minimum-norm subgradient, is continuous, so a reference
    point with tiny spurious nonzeros (an interior-point solution) is not
    reported as far from optimal.
    """
    return _norm(beta - _soft(beta - gradient, p.regularization))


def _split(beta: Array) -> Array:
    """The optimal split ``(beta+, beta-)`` of a LASSO coefficient vector."""
    return np.concatenate((np.maximum(beta, 0.0), np.maximum(-beta, 0.0)))


def _lasso_truth(
    p, design_smoothness: float, tolerance: float, iterations: int, time_limit: float
):
    """FISTA with adaptive restart, then a polish on the support and signs.

    ``design_smoothness`` is (an upper bound on) ``||A||^2``, the Lipschitz
    constant of the gradient of the unsplit least-squares term.
    """
    started = time.perf_counter()
    A, target, lam = p.A, p.target, p.regularization
    d = A.shape[1]
    step = 1.0 / design_smoothness
    atb = A.T @ target

    def gradient(beta):
        return _gram(A, beta) - atb

    def residual(beta, g=None):
        return _lasso_subgradient(beta, gradient(beta) if g is None else g, p)

    scale = 1.0 + residual(np.zeros(d), -atb)
    beta = np.zeros(d)
    momentum = beta.copy()
    t = 1.0
    achieved = scale - 1.0
    for iteration in range(iterations):
        previous = beta
        beta = _soft(momentum - step * gradient(momentum), step * lam)
        if np.dot(momentum - beta, beta - previous) > 0.0:
            t = 1.0  # gradient restart (O'Donoghue and Candes)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        momentum = beta + ((t - 1.0) / t_next) * (beta - previous)
        t = t_next
        if iteration % 10 == 9:
            achieved = residual(beta)
            if achieved <= tolerance * scale:
                break
            if time.perf_counter() - started >= time_limit:
                break
    beta, achieved = _polish(A, atb, lam, beta, residual(beta), p)
    seconds = time.perf_counter() - started
    x = _split(beta)
    met = achieved <= tolerance * scale
    return GroundTruth(
        x=x,
        value=float(p.objective(x)),
        status="optimal" if met else "optimal_inaccurate",
        tolerance=tolerance if met else achieved / scale,
        seconds=seconds,
        stationarity=achieved,
        feasibility=0.0,
        planted_error=nan,
        solver="fista_polish",
    )


def _polish(A, atb: Array, lam: float, beta: Array, achieved: float, p):
    """Solve the LASSO restricted to the support and signs of ``beta`` exactly.

    On the support ``S`` with signs ``s`` the optimality condition is the
    linear system ``A_S'A_S b_S = A_S'y - lambda s``; the solution is kept only
    if it keeps the signs and lowers the optimality residual.
    """
    support = np.flatnonzero(beta)
    if support.size == 0:
        return beta, achieved
    d = beta.size
    signs = np.sign(beta[support])

    def restricted(v):
        full = np.zeros(d)
        full[support] = np.ravel(v)
        return _gram(A, full)[support]

    operator = spla.LinearOperator((support.size, support.size), matvec=restricted)
    solution, _ = spla.cg(
        operator,
        atb[support] - lam * signs,
        x0=beta[support],
        rtol=EXACT_CG_TOLERANCE,
        atol=0.0,
        maxiter=10 * support.size + 1000,
    )
    if not np.array_equal(np.sign(solution), signs):
        return beta, achieved
    candidate = np.zeros(d)
    candidate[support] = solution
    residual = _lasso_subgradient(candidate, _gram(A, candidate) - atb, p)
    if residual < achieved:
        return candidate, residual
    return beta, achieved


def _logistic_truth(p, tolerance: float, iterations: int, time_limit: float):
    """Newton-CG with an Armijo backtracking line search on the ridge logistic loss."""
    started = time.perf_counter()
    A, labels, ridge = p.A, p.target, p.regularization
    rows, d = A.shape

    def evaluate(x):
        margins = labels * (A @ x)
        value = float(np.logaddexp(0.0, -margins).mean() + 0.5 * ridge * (x @ x))
        gradient = -(A.T @ (labels * expit(-margins))) / rows + ridge * x
        return value, gradient, margins

    x = np.zeros(d)
    value, gradient, margins = evaluate(x)
    scale = 1.0 + _norm(gradient)
    for _ in range(iterations):
        norm = _norm(gradient)
        if norm <= tolerance * scale or time.perf_counter() - started >= time_limit:
            break
        probability = expit(margins)
        weights = probability * (1.0 - probability) / rows
        hessian = spla.LinearOperator(
            (d, d),
            matvec=lambda v, weights=weights: (
                A.T @ (weights * (A @ np.ravel(v))) + ridge * np.ravel(v)
            ),
        )
        forcing = min(0.5, math.sqrt(norm / scale))
        direction, _ = spla.cg(
            hessian, -gradient, rtol=forcing, atol=0.0, maxiter=max(50, 10 * d)
        )
        slope = float(gradient @ direction)
        if slope >= 0.0:
            direction, slope = -gradient, -(norm**2)
        step = 1.0
        for _ in range(60):
            trial = x + step * direction
            trial_value = float(
                np.logaddexp(0.0, -labels * (A @ trial)).mean()
                + 0.5 * ridge * (trial @ trial)
            )
            if trial_value <= value + 1e-4 * step * slope:
                break
            step *= 0.5
        else:
            break  # no decrease at the floating-point floor
        x = trial
        value, gradient, margins = evaluate(x)
    norm = _norm(gradient)
    met = norm <= tolerance * scale
    return GroundTruth(
        x=x,
        value=value,
        status="optimal" if met else "optimal_inaccurate",
        tolerance=tolerance if met else norm / scale,
        seconds=time.perf_counter() - started,
        stationarity=norm,
        feasibility=0.0,
        planted_error=nan,
        solver="newton_cg",
    )


def _clarabel_truth(instance: Instance) -> GroundTruth:
    """CVXPY with Clarabel, tightest tolerance first, as in the original studies."""
    import cvxpy as cp

    p = instance.problem
    A = _explicit(p.A)
    rows, d = A.shape
    beta = cp.Variable(d)
    if instance.family == "lasso":
        objective = 0.5 * cp.sum_squares(A @ beta - p.target)
        objective = objective + p.regularization * cp.norm1(beta)
    else:
        objective = cp.sum(cp.logistic(-cp.multiply(p.target, A @ beta))) / rows + (
            0.5 * p.regularization
        ) * cp.sum_squares(beta)
    problem = cp.Problem(cp.Minimize(objective))
    started = time.perf_counter()
    status, tolerance = None, nan
    for candidate in REFERENCE_TOLERANCES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            problem.solve(
                solver=cp.CLARABEL,
                tol_gap_abs=candidate,
                tol_gap_rel=candidate,
                tol_feas=candidate,
                max_iter=500,
            )
        if problem.status == cp.OPTIMAL:
            status, tolerance = problem.status, candidate
            break
    if status is None:
        raise RuntimeError(f"reference solve ended with status {problem.status!r}")
    seconds = time.perf_counter() - started
    solution = np.asarray(beta.value, dtype=float)
    x = _split(solution) if instance.family == "lasso" else solution
    return GroundTruth(
        x=x,
        value=instance.objective(x),
        status=status,
        tolerance=tolerance,
        seconds=seconds,
        stationarity=instance.stationarity(x),
        feasibility=0.0,
        planted_error=nan,
        solver="clarabel",
    )
