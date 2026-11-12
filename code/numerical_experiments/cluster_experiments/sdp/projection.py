"""Approximate projection onto the positive semidefinite cone by block Lanczos.

The ``g``-subproblem of Algorithm 3 for an SDP is the projection of a
symmetric matrix ``M`` onto the cone.  Its exact solution is the Moreau
decomposition ``M = M_+ - M_-``: the projection is ``Z = M_+`` and
``B = -M_-`` is a normal vector of the cone at ``Z`` (``B`` is negative
semidefinite and ``Z B = 0``).

Algorithm 3 needs a pair ``(Z, B)`` with *exactly* these inclusions, but not
the exact decomposition.  A Rayleigh-Ritz step on any subspace gives one:
for an orthonormal basis ``Q`` and the eigendecomposition
``Q' M Q = S diag(theta) S'`` with Ritz vectors ``V = Q S``,

    Z = sum over theta_i > 0 of theta_i v_i v_i',
    B = sum over theta_i < 0 of theta_i v_i v_i'.

``Z`` is positive semidefinite, ``B`` is negative semidefinite, and
``Z B = 0`` because the Ritz vectors are orthonormal, whatever the subspace
is.  The only inexactness is the residual of the Moreau decomposition,

    Z + B - M = P M P - M,        P = Q Q',

which is what the relative error test of Algorithm 3 measures.  Its norm
satisfies ``||M - P M P||_F^2 = ||M||_F^2 - ||Q' M Q||_F^2``, so it is at
least the Frobenius norm of the eigenvalues of ``M`` that a subspace of
that dimension cannot hold: the block Krylov subspace must capture the most
positive *and* the most negative eigenvalues, and the eigenvalues it leaves
out must be negligible.

:class:`BlockLanczos` builds the block Krylov subspace of ``M`` from a
starting block, with full reorthogonalization, and :func:`restart_block`
chooses the starting block from the Ritz vectors of the previous pair (a
thick restart), which is the warm start when ``M`` changes a little.

**The one-sided pair.**  At an SDP solution ``M`` has no negligible
eigenvalues, so the pair above needs the whole space.  The one-sided pair
keeps only the positive Ritz pairs ``(Theta_+, V_+)`` explicit and takes the
rest of ``M`` as the normal vector:

    Z = V_+ Theta_+ V_+',    B = N = (I - P_+) M (I - P_+),    P_+ = V_+ V_+'.

``Z B = 0`` holds exactly again, and with ``R_+ = M V_+ - V_+ Theta_+`` the
residual is ``Z + B - M = -(V_+ R_+' + R_+ V_+')``, of norm
``sqrt(2) ||R_+||``: only the ``r`` positive Ritz pairs have to converge.
What is no longer automatic is ``B <= 0``.  It holds exactly when the
deflated matrix ``N`` has no positive eigenvalue, that is, when no positive
direction of ``M`` is left outside ``V_+``.  The safeguard has two layers
(:meth:`BlockLanczos.one_sided`, :func:`certify`):

* *The guard.*  The non-positive Ritz pairs of ``M`` are Ritz pairs of ``N``.
  The largest of them, ``(theta_0, v_0)`` with residual ``rho_0``, estimates
  the top of the spectrum of ``N`` as ``theta_0 + kappa rho_0`` (``kappa`` is
  ``GUARD_FACTOR``).  A pair is *guarded* when that estimate is not positive,
  and only guarded pairs are accepted: an unresolved or positive top of ``N``
  asks for more Lanczos steps, whatever the relative error test says.  The
  guard Ritz vectors are kept across restarts, so an eigenvalue of ``M`` that
  rises through zero is followed until it joins ``V_+``; the first
  decomposition of a run is exact, so that there is something to follow.
  This costs no product with ``M`` beyond the Krylov subspace, but it is an
  estimate: a Krylov subspace cannot prove that nothing positive hides in
  its complement.
* *The certificate.*  ``B <= 0`` on the complement of ``V_+`` is equivalent
  to ``-B + alpha P_+`` being positive definite, which one Cholesky
  factorization decides (``n^3 / 3`` operations, a fraction of an
  eigendecomposition).  A pair that fails is not accepted; the subspace is
  extended instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray

# A direction of a new block is kept when its singular value exceeds this
# fraction of the norm of the block it came from; the rest are deflated.
DEFLATION = 1e-10
# A one-sided pair that will be certified is shifted down by this margin on
# the complement of ``V_+``, relative to the largest Ritz value in magnitude,
# so that the Cholesky factorization tests a strict inequality.
MARGIN = 1e-12
# Safety factor of the guard: the top of the deflated spectrum is estimated
# as ``theta_0 + GUARD_FACTOR * rho_0``.  With factor one this is the usual
# residual bound, which holds only if ``theta_0`` already approximates the top
# eigenvalue; in the first outer iterations of local trials, where the matrix
# changes a lot between restarts, the top was up to 5.3 residuals above it.
GUARD_FACTOR = 4.0


@dataclass
class ConePair:
    """A point ``Z`` of the cone and ``B`` with ``c B`` in its normal cone there.

    ``theta`` (ascending) and the orthonormal columns of ``V`` are the Ritz
    pairs that define both: ``Z`` sums the positive ones and ``B`` the
    negative ones.  The dense ``Z`` and ``B`` are kept for the residuals.
    """

    theta: Array
    V: Array
    Z: Array
    B: Array
    # One-sided pairs only: the guard's estimate of the top of the deflated
    # spectrum (the pair is guarded when it is not positive).  ``theta`` and
    # ``V`` then hold the positive Ritz pairs and the guard.
    bound: float = float("nan")
    # ``B <= 0`` holds by construction (two-sided, exact), or the pair passed
    # the safeguard it was asked to pass.
    certified: bool = True

    @classmethod
    def from_ritz(cls, theta: Array, V: Array) -> ConePair:
        positive = V[:, theta > 0.0]
        negative = V[:, theta < 0.0]
        Z = (positive * theta[theta > 0.0]) @ positive.T
        B = (negative * theta[theta < 0.0]) @ negative.T
        return cls(theta=theta, V=V, Z=(Z + Z.T) / 2.0, B=(B + B.T) / 2.0)

    @classmethod
    def zero(cls, n: int) -> ConePair:
        """``Z = 0`` and ``B = 0``, a valid pair to start from."""
        return cls.from_ritz(np.empty(0), np.empty((n, 0)))

    @classmethod
    def exact(cls, M: Array) -> ConePair:
        """The Moreau decomposition of ``M`` by a full eigendecomposition."""
        theta, V = np.linalg.eigh(M)
        return cls.from_ritz(theta, V)

    @property
    def rank_positive(self) -> int:
        return int(np.count_nonzero(self.theta > 0.0))

    @property
    def rank_negative(self) -> int:
        return int(np.count_nonzero(self.theta < 0.0))


def _orthonormalize(block: Array, basis: Array) -> Array:
    """Orthonormal columns spanning ``block`` after projecting out ``basis``.

    Two passes of classical Gram-Schmidt against the basis, then a QR
    factorization; the columns stay orthogonal to the basis to rounding error.
    """
    for _ in range(2):
        block = block - basis @ (basis.T @ block)
    q, _ = np.linalg.qr(block)
    return q - basis @ (basis.T @ q)


class BlockLanczos:
    """The block Krylov subspace of a symmetric ``M``, one block at a time.

    ``start`` has orthonormal columns.  Each :meth:`step` multiplies the
    newest block by ``M`` (one product with ``M`` per column), adds the block
    to the basis, and forms the next block from the part of the product
    outside the basis.  The basis is fully reorthogonalized and the projected
    matrix ``T = Q' M Q`` is computed from the products rather than assumed
    block tridiagonal, so ``T`` is the exact compression of ``M`` to the
    subspace in floating point as well.

    A block that loses rank (an invariant subspace) is refilled with random
    directions, because the subspace has to grow until the residual
    ``P M P - M`` is small, not merely until the Ritz pairs converge; the
    method is ``exhausted`` only when the basis is all of ``R^n``, where the
    Rayleigh-Ritz step is the exact eigendecomposition.
    """

    def __init__(self, M: Array, start: Array, rng: np.random.Generator):
        n = M.shape[0]
        self.M = M
        self.n = n
        self.rng = rng
        self.Q = np.empty((n, n))
        self.W = np.empty((n, n))  # the products M Q, for the Ritz residuals
        self.T = np.zeros((n, n))
        self.dim = 0
        self.block = start[:, : min(start.shape[1], n)]
        self.matvecs = 0
        self.steps = 0

    @property
    def exhausted(self) -> bool:
        return self.block.shape[1] == 0

    def step(self) -> None:
        V = self.block
        size = V.shape[1]
        W = self.M @ V
        self.matvecs += size
        self.steps += 1
        k = self.dim
        cross = self.Q[:, :k].T @ W
        inner = V.T @ W
        self.T[:k, k : k + size] = cross
        self.T[k : k + size, :k] = cross.T
        self.T[k : k + size, k : k + size] = (inner + inner.T) / 2.0
        self.Q[:, k : k + size] = V
        self.W[:, k : k + size] = W
        self.dim = k + size
        self.block = self._next_block(W, min(size, self.n - self.dim))

    def _next_block(self, W: Array, size: int) -> Array:
        """The next block: the part of ``W`` outside the basis, ``size`` wide."""
        basis = self.Q[:, : self.dim]
        if size == 0:
            return np.empty((self.n, 0))
        residual = W - basis @ (basis.T @ W)
        residual -= basis @ (basis.T @ residual)
        left, singular, _ = np.linalg.svd(residual, full_matrices=False)
        keep = singular > DEFLATION * max(float(np.linalg.norm(W)), 1e-300)
        block = left[:, keep][:, :size]
        missing = size - block.shape[1]
        if missing:
            fill = self.rng.standard_normal((self.n, missing))
            block = np.hstack([block, fill])
        return _orthonormalize(block, basis)

    def ritz(self) -> ConePair:
        """The cone pair of the Rayleigh-Ritz step on the current basis."""
        k = self.dim
        theta, S = np.linalg.eigh(self.T[:k, :k])
        return ConePair.from_ritz(theta, self.Q[:, :k] @ S)

    def one_sided(
        self, guard: int, margin: float = 0.0, factor: float = GUARD_FACTOR
    ) -> ConePair:
        """The one-sided cone pair of the current basis (module docstring).

        ``guard`` is the number of non-positive Ritz pairs kept with the
        positive ones for the next restart.  ``margin`` is ``MARGIN`` for a
        pair that will be certified and zero otherwise.  ``factor`` is the
        safety factor of the guard's estimate.
        """
        if guard < 1:
            raise ValueError("the guard needs at least one Ritz pair")
        k, n = self.dim, self.n
        theta, S = np.linalg.eigh(self.T[:k, :k])
        r = int(np.count_nonzero(theta > 0.0))
        # eigh sorts ascending: the non-positive Ritz values come first, and
        # the guard is the top of them.
        keep = np.arange(max(k - r - guard, 0), k)
        V = self.Q[:, :k] @ S[:, keep]
        residual = self.W[:, :k] @ S[:, keep] - V * theta[keep]
        positive = theta[keep] > 0.0
        V_plus, R_plus = V[:, positive], residual[:, positive]
        scale = float(np.max(np.abs(theta)))
        if k == n:
            # The whole space: the Ritz pairs are the eigenpairs of M.
            bound = float(theta[k - r - 1]) if r < k else -scale
        elif r < k:
            top = k - r - 1 - keep[0]
            rho = float(np.linalg.norm(residual[:, top]))
            bound = float(theta[k - r - 1]) + factor * rho
        else:
            # Every Ritz value is positive, so nothing is known below them:
            # the smallest Ritz pair stands in for the top of the complement.
            bound = float(theta[0]) + factor * float(np.linalg.norm(residual[:, 0]))
        Z = (V_plus * theta[keep][positive]) @ V_plus.T
        cross = V_plus @ R_plus.T
        B = self.M - Z - cross - cross.T
        if margin:
            B -= margin * scale * (np.eye(n) - V_plus @ V_plus.T)
        return ConePair(
            theta=theta[keep],
            V=V,
            Z=(Z + Z.T) / 2.0,
            B=(B + B.T) / 2.0,
            bound=bound,
            certified=False,
        )


def restart_block(
    pair: ConePair,
    tolerance: float,
    block_size: int,
    rng: np.random.Generator,
) -> Array:
    """The starting block of a new Krylov subspace: a thick restart.

    Keeps the Ritz vectors of ``pair`` with the largest ``|theta|``, the most
    positive and the most negative ones, dropping the smallest as long as the
    Frobenius norm of the dropped Ritz values stays below ``tolerance`` (what
    dropping them adds to the residual when the matrix has not changed).
    The block is padded with random directions to at least ``block_size``
    columns.
    """
    n = pair.V.shape[0]
    order = np.argsort(-np.abs(pair.theta))
    squares = pair.theta[order] ** 2
    # tail[i] is the Frobenius norm of the Ritz values from position i on.
    tail = np.sqrt(np.cumsum(squares[::-1])[::-1])
    kept = pair.V[:, order[: int(np.count_nonzero(tail > tolerance))]]
    missing = min(block_size, n) - kept.shape[1]
    if missing <= 0:
        return kept
    fill = _orthonormalize(rng.standard_normal((n, missing)), kept)
    return np.hstack([kept, fill])


def restart_one_sided(pair: ConePair) -> Array:
    """The starting block after a one-sided pair: its positive Ritz vectors
    and its guard.

    A cold start (no pair yet) returns the whole space, which makes the first
    decomposition exact: a small random subspace says nothing about the top
    of the spectrum, so there would be nothing for the guard to follow.
    """
    if pair.V.shape[1] == 0:
        return np.eye(pair.V.shape[0])
    return pair.V


def certify(pair: ConePair) -> bool:
    """Whether ``B`` is negative definite on the complement of ``V_+``.

    One Cholesky factorization of ``-B + alpha P_+``: it exists exactly when
    that matrix is positive definite, and ``B V_+ = 0``, so it exists exactly
    when ``B < 0`` on the complement.  Marks the pair as certified.
    """
    V_plus = pair.V[:, pair.theta > 0.0]
    alpha = max(float(np.max(np.abs(pair.theta), initial=0.0)), 1e-300)
    try:
        np.linalg.cholesky(-pair.B + alpha * (V_plus @ V_plus.T))
    except np.linalg.LinAlgError:
        return False
    pair.certified = True
    return True
