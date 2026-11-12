"""Small synthetic semidefinite programs.

Every instance is

    minimize <C, X>  subject to  E(X) = b,  X positive semidefinite,

with ``X`` an ``n x n`` symmetric matrix and ``E`` a linear map into ``R^m``.
Its dual is ``maximize b'nu`` subject to ``S = C - E*(nu)`` positive
semidefinite, and ``(X, nu, S)`` is optimal exactly when it is feasible for
both and ``X S = 0``.

``planted`` draws the solution first and builds the data around it, so the
optimal value and a primal-dual solution are known to rounding error without
a solver.  ``maxcut`` is the max-cut relaxation of a random graph, whose
solution is not known in advance; :func:`reference` solves any instance with
CVXPY for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


class DenseMap:
    """``E(X)_i = <A_i, X>`` for dense symmetric ``A_i``.

    The ``A_i`` are stored as the rows of an ``m x n^2`` matrix, so ``E`` and
    its adjoint ``E*(u) = sum_i u_i A_i`` are one matrix-vector product each.
    """

    def __init__(self, matrices: Array):
        self.m, self.n, _ = matrices.shape
        self.rows = np.ascontiguousarray(matrices.reshape(self.m, -1))

    def apply(self, X: Array) -> Array:
        return self.rows @ X.reshape(-1)

    def adjoint(self, u: Array) -> Array:
        return (u @ self.rows).reshape(self.n, self.n)

    def gram(self) -> Array:
        """The ``m x m`` matrix of ``E E*``."""
        return self.rows @ self.rows.T


class DiagonalMap:
    """``E(X) = diag(X)``, the constraint map of the max-cut relaxation."""

    def __init__(self, n: int):
        self.m = self.n = n

    def apply(self, X: Array) -> Array:
        return np.diag(X).copy()

    def adjoint(self, u: Array) -> Array:
        return np.diag(u)

    def gram(self) -> Array:
        return np.eye(self.n)


@dataclass
class SDP:
    """One instance, with its planted solution when there is one."""

    family: str
    n: int
    m: int
    seed: int
    E: DenseMap | DiagonalMap
    b: Array
    C: Array
    # Planted primal-dual solution (``None`` when it is not known), and the
    # ranks of ``X*`` and ``S*`` (-1 when not known).
    X_star: Array | None = None
    nu_star: Array | None = None
    S_star: Array | None = None
    rank: int = -1
    dual_rank: int = -1

    @property
    def planted(self) -> bool:
        return self.X_star is not None

    @property
    def objective_star(self) -> float:
        """The optimal value, ``nan`` when no solution is planted."""
        if self.X_star is None:
            return float("nan")
        return float(np.vdot(self.C, self.X_star))

    def operator_norm(self) -> float:
        """``||E||``, the square root of the largest eigenvalue of ``E E*``."""
        return float(np.sqrt(np.linalg.eigvalsh(self.E.gram())[-1]))


def planted(
    n: int,
    m: int,
    rank: int,
    dual_rank: int | None = None,
    seed: int = 0,
    spread: float = 10.0,
) -> SDP:
    """A random SDP built around a planted primal-dual solution.

    ``X* = U_1 diag(lambda) U_1'`` has rank ``rank`` and
    ``S* = U_2 diag(mu) U_2'`` has rank ``dual_rank``, with ``[U_1, U_2]``
    orthonormal columns of one random orthogonal matrix, so ``X* S* = 0``;
    ``lambda`` and ``mu`` are uniform in ``[0.5, 1.5]`` and both matrices are
    scaled to unit Frobenius norm.  ``dual_rank = n - rank`` (the default) is
    strict complementarity; a smaller one leaves a common null space.

    The ``A_i`` are symmetrized Gaussian matrices with Frobenius norms
    log-spaced in ``[1, spread]``, which spreads the spectrum of ``E E*`` over
    about ``[1, spread^2]`` and makes the linear system of the
    ``f``-subproblem worth an iterative solve.  The multiplier ``nu*`` is
    Gaussian, scaled so that ``E*(nu*)`` has about unit norm.  Then
    ``b = E(X*)`` and ``C = S* + E*(nu*)`` make ``(X*, nu*, S*)`` optimal.
    """
    if dual_rank is None:
        dual_rank = n - rank
    if not 0 < rank < n or not 0 <= dual_rank <= n - rank:
        raise ValueError("need 0 < rank < n and 0 <= dual_rank <= n - rank")
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    primal = U[:, :rank]
    X_star = (primal * rng.uniform(0.5, 1.5, rank)) @ primal.T
    X_star = (X_star + X_star.T) / (2.0 * np.linalg.norm(X_star))
    if dual_rank:
        dual = U[:, rank : rank + dual_rank]
        S_star = (dual * rng.uniform(0.5, 1.5, dual_rank)) @ dual.T
        S_star = (S_star + S_star.T) / (2.0 * np.linalg.norm(S_star))
    else:
        S_star = np.zeros((n, n))
    scales = np.logspace(0.0, np.log10(spread), m) if m > 1 else np.ones(1)
    matrices = rng.standard_normal((m, n, n))
    matrices = (matrices + matrices.transpose(0, 2, 1)) / 2.0
    norms = np.linalg.norm(matrices.reshape(m, -1), axis=1)
    matrices *= (scales / norms)[:, None, None]
    E = DenseMap(matrices)
    nu_star = rng.standard_normal(m) / (scales * np.sqrt(m))
    b = E.apply(X_star)
    C = S_star + E.adjoint(nu_star)
    C = (C + C.T) / 2.0
    return SDP(
        family="planted",
        n=n,
        m=m,
        seed=seed,
        E=E,
        b=b,
        C=C,
        X_star=X_star,
        nu_star=nu_star,
        S_star=S_star,
        rank=rank,
        dual_rank=dual_rank,
    )


def maxcut(n: int, seed: int = 0, density: float = 0.5) -> SDP:
    """The max-cut relaxation of an Erdos-Renyi graph ``G(n, density)``.

    ``minimize <C, X>`` subject to ``diag(X) = 1`` and ``X`` positive
    semidefinite, with ``C = -L / ||L||_2`` for the graph Laplacian ``L``
    (the usual ``-L/4`` up to a positive scale, so that the dual slack is of
    order one).  Here ``E E* = I``, so the ``f``-subproblem has two distinct
    eigenvalues and conjugate gradients solves it in two steps: this family
    exercises the projection alone.
    """
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.random((n, n)) < density, 1).astype(float)
    adjacency = upper + upper.T
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    C = -laplacian / np.linalg.eigvalsh(laplacian)[-1]
    return SDP(
        family="maxcut",
        n=n,
        m=n,
        seed=seed,
        E=DiagonalMap(n),
        b=np.ones(n),
        C=C,
    )


def reference(sdp: SDP) -> dict:
    """Solve the instance with CVXPY (Clarabel) and report what it achieved.

    Returns the optimal value and the solver's own residuals; ``value`` is
    ``nan`` when CVXPY or the solver is unavailable or fails, with the reason
    in ``status``.
    """
    try:
        import cvxpy as cp
    except ImportError as error:
        return dict(value=float("nan"), status=f"unavailable: {error}")
    X = cp.Variable((sdp.n, sdp.n), symmetric=True)
    if isinstance(sdp.E, DiagonalMap):
        constraints = [cp.diag(X) == sdp.b]
    else:
        # ``order="C"`` matches the row-major layout of ``DenseMap.rows``; the
        # matrices are symmetric, so the order does not change the values.
        constraints = [sdp.E.rows @ cp.vec(X, order="C") == sdp.b]
    constraints.append(X >> 0)
    problem = cp.Problem(cp.Minimize(cp.trace(sdp.C @ X)), constraints)
    try:
        problem.solve(solver=cp.CLARABEL)
    except (cp.SolverError, ValueError) as error:
        return dict(value=float("nan"), status=f"failed: {error}")
    if X.value is None:
        return dict(value=float("nan"), status=str(problem.status))
    solution = (X.value + X.value.T) / 2.0
    return dict(
        value=float(np.vdot(sdp.C, solution)),
        status=str(problem.status),
        feasibility=float(np.linalg.norm(sdp.E.apply(solution) - sdp.b)),
        min_eigenvalue=float(np.linalg.eigvalsh(solution)[0]),
    )
