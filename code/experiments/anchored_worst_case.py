"""Worst-case residual of the anchored (Halpern) iteration with relative errors.

The anchored variant of the inexact primal-dual Douglas--Rachford method (see
``tex/acceleration.tex``) evaluates a firmly nonexpansive map ``F`` at a
perturbed point ``y = z + eps`` with ``||eps|| <= sigma ||F(y)||`` and then
sets ``z_{k+1} = z_0/(k+2) + (k+1)/(k+2) (z_k - theta F(y_k))``.

This script computes, for given ``theta``, ``sigma`` and horizon ``N``, the
largest possible value of ``N ||F(y_{N-1})|| / ||z_0 - z*||`` over all firmly
nonexpansive maps and all admissible perturbations.  That is a performance
estimation problem: a semidefinite program in the Gram matrix of the vectors
``z_0 - z*``, ``F(y_0), ..., F(y_{N-1})``, ``eps_1, ..., eps_N``, whose
constraints are the pairwise interpolation inequalities of firmly nonexpansive
maps and the error bounds.  With ``sigma = 0`` the optimal value is
``2N / (theta (N - 1) + 2)``, which is the tight bound proved in the note.

No SDP solver is assumed: the program is solved by a small ADMM loop that
alternates a projection onto the affine constraints with a projection onto
the positive semidefinite cone.  The returned value is that of the
cone-feasible iterate, and the affine residual is reported alongside it so the
accuracy is visible in the table.  Results go to
``results/anchored_worst_case/results.csv``; the run resumes if interrupted.
"""

from __future__ import annotations

import time

import numpy as np

from ._cli import base_parser
from ._paths import RESULTS, study_dir
from ._results import merge_rows, read_rows, write_rows

STUDY = "anchored_worst_case"
DEFAULT_THETA = 1.0
DEFAULT_SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
DEFAULT_HORIZONS = (4, 8, 16, 32)
DEFAULT_ITERATIONS = 150_000
DEFAULT_TOLERANCE = 3e-8


def _svec_index(dimension: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(dimension) for j in range(i, dimension)]


def _svec(matrix: np.ndarray, index: list[tuple[int, int]]) -> np.ndarray:
    root2 = np.sqrt(2.0)
    return np.array(
        [matrix[i, j] if i == j else root2 * matrix[i, j] for (i, j) in index]
    )


def _smat(vector: np.ndarray, index: list[tuple[int, int]], dim: int) -> np.ndarray:
    matrix = np.zeros((dim, dim))
    root2 = np.sqrt(2.0)
    for k, (i, j) in enumerate(index):
        if i == j:
            matrix[i, i] = vector[k]
        else:
            matrix[i, j] = matrix[j, i] = vector[k] / root2
    return matrix


def build_program(
    horizon: int, sigma: float, theta: float
) -> tuple[int, list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Return the dimension, inequality, equality and objective matrices."""
    dim = 1 + 2 * horizon
    basis = np.eye(dim)
    x0 = basis[0]
    steps = [None] + [basis[i] for i in range(1, horizon + 1)]
    errors = [None] + [basis[horizon + i] for i in range(1, horizon + 1)]
    iterates = [x0]
    for k in range(horizon):
        weight = 1.0 / (k + 2)
        iterates.append(
            weight * x0 + (1 - weight) * (iterates[k] - theta * steps[k + 1])
        )
    points = [(np.zeros(dim), np.zeros(dim))]
    points += [(iterates[i - 1] + errors[i], steps[i]) for i in range(1, horizon + 1)]
    inequalities = []
    for a in range(len(points)):
        for b in range(a + 1, len(points)):
            ya, ga = points[a]
            yb, gb = points[b]
            dg, dy = ga - gb, ya - yb
            inequalities.append(
                0.5 * (np.outer(dg, dy) + np.outer(dy, dg)) - np.outer(dg, dg)
            )
    for i in range(1, horizon + 1):
        inequalities.append(
            sigma**2 * np.outer(steps[i], steps[i]) - np.outer(errors[i], errors[i])
        )
    equalities = [np.outer(x0, x0)]
    objective = np.outer(steps[horizon], steps[horizon])
    return dim, inequalities, equalities, objective


def solve_worst_case(
    horizon: int,
    sigma: float,
    theta: float,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[float, float, float, int]:
    """Return the worst-case ratio, the two ADMM residuals and the iterations."""
    dim, inequalities, equalities, objective = build_program(horizon, sigma, theta)
    index = _svec_index(dim)
    n_gram = len(index)
    n_ineq, n_eq = len(inequalities), len(equalities)
    # Variables: the Gram matrix in svec form, then one slack per inequality.
    matrix = np.zeros((n_ineq + n_eq, n_gram + n_ineq))
    for i, constraint in enumerate(inequalities):
        matrix[i, :n_gram] = _svec(constraint, index)
        matrix[i, n_gram + i] = -1.0
    for j, constraint in enumerate(equalities):
        matrix[n_ineq + j, :n_gram] = _svec(constraint, index)
    rhs = np.zeros(n_ineq + n_eq)
    rhs[n_ineq:] = 1.0
    cost = np.zeros(n_gram + n_ineq)
    cost[:n_gram] = _svec(objective, index)
    gram_of_rows = matrix @ matrix.T
    chol = np.linalg.cholesky(gram_of_rows + 1e-12 * np.eye(gram_of_rows.shape[0]))

    def project_affine(point: np.ndarray) -> np.ndarray:
        residual = matrix @ point - rhs
        multiplier = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
        return point - matrix.T @ multiplier

    def project_cone(point: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(_smat(point[:n_gram], index, dim))
        psd = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        out = np.empty_like(point)
        out[:n_gram] = _svec(psd, index)
        out[n_gram:] = np.maximum(point[n_gram:], 0)
        return out

    rho, relaxation = 1.0, 1.6
    v = np.zeros(n_gram + n_ineq)
    w = np.zeros_like(v)
    u = np.zeros_like(v)
    primal = dual = np.inf
    for iteration in range(iterations):
        v = project_affine(w - u + cost / rho)
        relaxed = relaxation * v + (1 - relaxation) * w
        w_new = project_cone(relaxed + u)
        u = u + relaxed - w_new
        primal = float(np.linalg.norm(v - w_new))
        dual = float(rho * np.linalg.norm(w_new - w))
        w = w_new
        if iteration % 50 == 0:
            scale = max(1.0, float(np.linalg.norm(w)))
            if primal / scale < tolerance and dual / scale < tolerance:
                break
            if primal > 10 * dual:
                rho *= 2
                u /= 2
            elif dual > 10 * primal:
                rho /= 2
                u *= 2
    value = float(cost @ w)
    ratio = horizon * float(np.sqrt(max(value, 0.0)))
    return ratio, primal, dual, iteration + 1


def parse_args():
    parser = base_parser(
        "Worst-case scaled residual of the anchored iteration with relative errors",
        resumable=True,
    )
    parser.add_argument("--theta", type=float, default=DEFAULT_THETA)
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS)
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = study_dir(STUDY, args.results_dir) / RESULTS
    existing = [] if args.force else read_rows(path)
    done = {(row["theta"], row["sigma"], row["horizon"]) for row in existing}
    rows = list(existing)
    for sigma in args.sigmas:
        for horizon in args.horizons:
            key = (args.theta, sigma, horizon)
            if key in done:
                continue
            start = time.perf_counter()
            ratio, primal, dual, count = solve_worst_case(
                horizon,
                sigma,
                args.theta,
                iterations=args.iterations,
                tolerance=args.tolerance,
            )
            exact = 2 * horizon / (args.theta * (horizon - 1) + 2)
            row = dict(
                theta=args.theta,
                sigma=sigma,
                horizon=horizon,
                worst_scaled_residual=ratio,
                exact_scaled_residual=exact,
                ratio_to_exact=ratio / exact,
                admm_primal_residual=primal,
                admm_dual_residual=dual,
                admm_iterations=count,
                seconds=time.perf_counter() - start,
            )
            rows = merge_rows(
                rows, [row], key=lambda r: (r["theta"], r["sigma"], r["horizon"])
            )
            write_rows(path, rows)
            print(
                f"theta={args.theta} sigma={sigma} N={horizon}: "
                f"N*worst={ratio:.4f} (exact {exact:.4f}), "
                f"residuals {primal:.1e}/{dual:.1e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
