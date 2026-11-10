"""One-step inner solvers with shared, explicitly timed per-run setup."""

from collections import OrderedDict
from time import perf_counter

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .outcomes import SolverFailure


class Inner:
    def __init__(
        self,
        problem,
        c,
        method,
        rng,
        block_size=32,
        rank=64,
        cache_blocks=8,
        gd_factor=1.0,
    ):
        self.p, self.c, self.method, self.rng = problem, c, method, rng
        self.block_size = min(block_size, problem.n)
        self.cache_blocks = cache_blocks
        self.cache = OrderedDict()
        self.products = self.row_updates = self.block_factorizations = 0
        self.precondition = lambda r: r.copy()
        self.gd_factor = gd_factor
        started = perf_counter()
        if method == "nystrom_pcg":
            rank = min(rank, problem.n)
            omega, _ = la.qr(rng.normal(size=(problem.n, rank)), mode="economic")
            Y = np.empty_like(omega)
            for start in range(0, rank, 32):
                Y[:, start : start + 32] = self.qmul(omega[:, start : start + 32])
            # Stable shifted Nyström construction (Frangella/Tropp/Udell).
            shift = max(np.finfo(float).eps * la.norm(Y), 1e-15)
            Y += shift * omega
            gram = omega.T @ Y
            C = la.cholesky((gram + gram.T) / 2, lower=True)
            B = la.solve_triangular(C, Y.T, lower=True).T
            U, singular, _ = la.svd(B, full_matrices=False)
            eigen = np.maximum(singular**2 - shift, 0.0)
            weights = (eigen[-1] + c) / (eigen + c) - 1
            self.condition_bound = float((eigen[0] + c) / (eigen[-1] + c))
            self.precondition = lambda r: r + U @ (weights * (U.T @ r))
        elif method in ("block_kaczmarz", "cyclic_kaczmarz"):
            if problem.family != "qp":
                raise ValueError("block Kaczmarz currently supports sparse QP only")
            self.H = (problem.Q + c * sp.eye(problem.n, format="csr")).tocsr()
            self.permutation = rng.permutation(problem.n)
            self.blocks = (problem.n + self.block_size - 1) // self.block_size
            self.get_block(0)  # first block setup charged before iteration
        self.setup_seconds = perf_counter() - started

    def qmul(self, x):
        self.products += 1 if x.ndim == 1 else x.shape[1]
        return self.p.hess(x)

    def hmul(self, x):
        return self.qmul(x) + self.c * x

    def get_block(self, number):
        if number in self.cache:
            self.cache.move_to_end(number)
            return self.cache[number]
        rows = self.permutation[
            number * self.block_size : (number + 1) * self.block_size
        ]
        B = self.H[rows]
        support = np.unique(B.indices)
        B = B[:, support]
        gram = (B @ B.T).toarray()  # at most block_size squared
        factor = la.cho_factor(gram, lower=True)
        self.block_factorizations += 1
        self.cache[number] = (rows, B, factor, support)
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return rows, B, factor, support

    def start(self, x, rhs):
        self.x = x.copy()
        self.rhs = rhs
        self.r = rhs - self.hmul(x)
        self.direction = self.precondition(self.r)
        self.rz = float(self.r @ self.direction)
        self.previous = self.x.copy()
        self.steps = 0
        if self.method == "gmres":
            self.V = []
            self.hessenberg = np.zeros((51, 50))
            self.base = x.copy()
            self.beta = float(la.norm(self.r))
            if self.beta:
                self.V = [self.r / self.beta]

    def step(self, q_x=None):
        method = self.method
        if method in ("gd", "agd"):
            if method == "agd":
                cond = (self.p.L + self.c) / (self.p.mu + self.c)
                momentum = (np.sqrt(cond) - 1) / (np.sqrt(cond) + 1)
                y = self.x + momentum * (self.x - self.previous)
                new = y + (self.rhs - self.hmul(y)) / (self.p.L + self.c)
            else:
                eta = self.gd_factor * 2 / (self.p.L + self.p.mu + 2 * self.c)
                h_x = self.hmul(self.x) if q_x is None else q_x + self.c * self.x
                new = self.x + eta * (self.rhs - h_x)
            self.previous, self.x = self.x, new
        elif method in ("cg", "nystrom_pcg"):
            if self.rz > np.finfo(float).tiny:
                hp = self.hmul(self.direction)
                denominator = float(self.direction @ hp)
                if denominator <= 0:
                    raise SolverFailure("CG lost positive curvature")
                a = self.rz / denominator
                self.x += a * self.direction
                self.r -= a * hp
                z = self.precondition(self.r)
                rz = float(self.r @ z)
                self.direction = z + (rz / self.rz) * self.direction
                self.rz = rz
        elif method in ("block_kaczmarz", "cyclic_kaczmarz"):
            number = (
                int(self.rng.integers(self.blocks))
                if method == "block_kaczmarz"
                else self.steps % self.blocks
            )
            rows, B, factor, support = self.get_block(number)
            residual = self.rhs[rows] - B @ self.x[support]
            self.x[support] += B.T @ la.cho_solve(factor, residual)
            self.row_updates += len(rows)
        elif method == "gmres":
            j = len(self.V) - 1
            if j < 0:
                return
            v = self.hmul(self.V[j])
            basis = np.asarray(self.V)
            h = basis @ v
            v -= h @ basis
            correction = basis @ v
            v -= correction @ basis
            h += correction
            self.hessenberg[: j + 1, j] = h
            norm = la.norm(v)
            self.hessenberg[j + 1, j] = norm
            target = np.zeros(j + 2)
            target[0] = self.beta
            coefficients = la.lstsq(self.hessenberg[: j + 2, : j + 1], target)[0]
            self.x = self.base + coefficients @ basis
            if norm < 1e-14 or j == 49:
                self.start(self.x, self.rhs)
            else:
                self.V.append(v / norm)
        else:
            raise ValueError(method)
        self.steps += 1
