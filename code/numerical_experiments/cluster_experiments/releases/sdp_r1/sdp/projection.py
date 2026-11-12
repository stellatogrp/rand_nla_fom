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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray

# A direction of a new block is kept when its singular value exceeds this
# fraction of the norm of the block it came from; the rest are deflated.
DEFLATION = 1e-10


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
