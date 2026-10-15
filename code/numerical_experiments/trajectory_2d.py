"""The paths of the iterates on a two-dimensional Huber instance.

    python -m numerical_experiments.trajectory_2d              # run and draw
    python -m numerical_experiments.trajectory_2d --plot-only  # redraw

The instance is drawn exactly like those of the main study (``make_huber``:
Gaussian design, noise, gross outliers, threshold ``M``) but with ``n = 2``
features and few observations, so that the loss is anisotropic and kinked
and the proximal steps are visible at full scale.  The default, ``m = 7``
observations with seed 6, is the instance with the most visible steps
among those whose near-exact run takes fewer than 50 outer iterations in
the ranking that ``--scan`` prints (seeds 0 to 39, several observation
counts).  From the origin, the near-exact proximal point method, the
inexact one at every ``sigma`` of the main study's sweep and at ``sigma*``
(all at ``b = 1``), and plain gradient descent run to the tolerance
``delta``; CVXPY gives the optimum.  Tables in
``results/numerical_experiments/ppm_huber_2d/``:

``instance.csv``
    the instance constants, the optimum, and the start;
``runs.csv``
    one row per run: outer iterations, gradient steps, time;
``trajectories.csv``
    one row per run and iterate: coordinates, ``f - f*``, gradient norm,
    and for the proximal runs the centre ``w_k`` of the subproblem the
    iterate came from, its exact solution ``J(w_k)`` (gradient descent to
    machine precision), and the distance between the two;
``contours.csv``
    ``f - f*`` on a grid over the full view and over the zoom window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import theory
from ._io import write_rows
from .ppm import accurate_resolvent, gradient_descent, inexact_ppm
from .problems import make_huber
from .study import DEFAULT_RESULTS_DIR, DELTA, NEAR_EXACT_TOLERANCE, SIGMAS

STUDY = "ppm_huber_2d"
SCAN_OBSERVATIONS = (6, 7, 8, 10, 12, 15, 20)
nan = float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="iterate paths on a two-dimensional Huber-loss instance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=6, help="instance seed")
    parser.add_argument("--m", type=int, default=7, help="observations")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Huber threshold M"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=10.0,
        help="gamma L: the stepsize is gamma = ratio / L",
    )
    parser.add_argument("--delta", type=float, default=DELTA, help="stopping tolerance")
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(SIGMAS))
    parser.add_argument(
        "--near-exact-tolerance",
        type=float,
        default=NEAR_EXACT_TOLERANCE,
        help="absolute inner residual tolerance of the near-exact run",
    )
    parser.add_argument(
        "--gd-step",
        type=float,
        default=1.0,
        help="stepsize of the gradient descent baseline, in units of 1/L",
    )
    parser.add_argument(
        "--grid", type=int, default=201, help="contour grid points per axis"
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=0.04,
        help="half-width of the zoom window as a fraction of the full view",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--name",
        default=STUDY,
        help="subdirectory of the results directory, so that several instances "
        "can be kept side by side",
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="only redraw from the tables"
    )
    parser.add_argument(
        "--scan",
        type=int,
        default=0,
        metavar="SEEDS",
        help="instead: rank the first SEEDS seeds of every observation count in "
        f"{SCAN_OBSERVATIONS} by visible proximal steps, print, and exit",
    )
    return parser


def scan(seeds: int, args: argparse.Namespace) -> list[dict]:
    """Rank small instances by how many proximal steps are visible at full scale.

    For every seed and observation count the near-exact proximal point
    method runs from the origin, and a step counts as visible when it moves
    the iterate by more than 2% of the distance from the start to the
    optimum.  Also reported: the outer iterations of the near-exact and of
    the ``sigma = 0.9`` runs, the gradient steps of the latter and of plain
    gradient descent, the condition number of the Hessian at the optimum
    (over the residuals within the threshold), and the gaps of the
    ``sigma = 0.9`` iterates to their exact subproblem solutions, summed and
    relative to that distance.
    """
    table: list[dict] = []
    for m in SCAN_OBSERVATIONS:
        for seed in range(seeds):
            problem = make_huber(seed, 2, m, args.threshold)
            smoothness = problem.smoothness()
            gamma = args.ratio / smoothness
            exact = inexact_ppm(
                problem,
                gamma,
                smoothness,
                0.0,
                1,
                args.delta,
                exact_tolerance=args.near_exact_tolerance,
            )
            if not exact.converged:
                continue
            points = np.array([np.zeros(2)] + list(exact.iterates))
            x_star = points[-1]
            distance = float(np.linalg.norm(x_star))
            steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
            residual = problem.A @ x_star - problem.b
            inlier = np.abs(residual) <= problem.threshold
            hessian = problem.A[inlier].T @ problem.A[inlier]
            eigenvalues = np.linalg.eigvalsh(hessian)
            loose = inexact_ppm(problem, gamma, smoothness, 0.9, 1, args.delta)
            gaps = [
                float(
                    np.linalg.norm(
                        x - accurate_resolvent(problem, gamma, smoothness, w)[0]
                    )
                )
                for x, w in zip(loose.iterates, loose.w, strict=True)
            ]
            descent = gradient_descent(
                problem, smoothness, args.delta, stepsize_ratio=args.gd_step
            )
            table.append(
                dict(
                    m=m,
                    seed=seed,
                    visible_steps=int((steps > 0.02 * distance).sum()),
                    k_exact=exact.k_delta,
                    k_loose=loose.k_delta,
                    steps_loose=int(sum(loose.inner)),
                    steps_gd=descent.k_delta,
                    kappa=float(eigenvalues[-1] / max(eigenvalues[0], 1e-12)),
                    relative_gap=float(sum(gaps)) / distance,
                    distance=distance,
                )
            )
    table.sort(key=lambda r: (-r["visible_steps"], -r["relative_gap"]))
    return table


def run(args: argparse.Namespace) -> Path:
    """Solve the instance every way, and write the tables."""
    directory = Path(args.results_dir) / args.name
    directory.mkdir(parents=True, exist_ok=True)
    problem = make_huber(args.seed, 2, args.m, args.threshold)
    truth = problem.ground_truth()
    smoothness = problem.smoothness()
    gamma = args.ratio / smoothness
    alpha, c0 = theory.inner_constants(gamma * smoothness)
    sigma_star = theory.optimal_sigma(alpha, c0, 1)
    x0 = np.zeros(2)

    configs = [("near_exact", nan, False)]
    configs += [("inexact", float(sigma), False) for sigma in args.sigmas]
    configs += [("inexact", sigma_star, True), ("gd", nan, False)]
    runs: list[dict] = []
    trajectories: list[dict] = []
    for method, sigma, optimal in configs:
        if method == "gd":
            trace = gradient_descent(
                problem,
                smoothness,
                args.delta,
                stepsize_ratio=args.gd_step,
                dense=10**9,  # record every iteration
            )
            points = list(trace.iterates)
            centres = [None] * len(points)
            cumulative = np.cumsum(trace.inner)
        else:
            trace = inexact_ppm(
                problem,
                gamma,
                smoothness,
                0.0 if method == "near_exact" else sigma,
                1,
                args.delta,
                exact_tolerance=(
                    args.near_exact_tolerance if method == "near_exact" else None
                ),
            )
            points = [x0] + list(trace.iterates)
            centres = [None] + list(trace.w)
            cumulative = np.concatenate([[0], np.cumsum(trace.inner)])
        identity = dict(
            method=method,
            sweep="sigma" if method == "inexact" else "reference",
            sigma=sigma,
            b=0 if method == "gd" else 1,
            sigma_is_optimal=optimal,
        )
        runs.append(
            dict(
                **identity,
                converged=trace.converged,
                k_delta=trace.k_delta,
                total_inner=int(sum(trace.inner)),
                gradient_evaluations=trace.gradient_evaluations,
                runtime_seconds=trace.seconds,
            )
        )
        for k, (x, w, n) in enumerate(zip(points, centres, cumulative, strict=True)):
            value, g = problem.value_and_gradient(x)
            row = dict(
                **identity,
                iteration=k,
                x1=float(x[0]),
                x2=float(x[1]),
                f_minus_fstar=value - truth.value,
                grad_norm=float(np.linalg.norm(g)),
                cumulative_inner=int(n),
                w1=nan,
                w2=nan,
                exact1=nan,
                exact2=nan,
                inexactness=nan,
            )
            if w is not None:
                exact, _ = accurate_resolvent(problem, gamma, smoothness, w)
                row.update(
                    w1=float(w[0]),
                    w2=float(w[1]),
                    exact1=float(exact[0]),
                    exact2=float(exact[1]),
                    inexactness=float(np.linalg.norm(x - exact)),
                )
            trajectories.append(row)

    # The full view holds every iterate and the optimum; the zoom window is
    # centred on the optimum.  Both are square so that the geometry is true.
    visited = np.array(
        [[t["x1"], t["x2"]] for t in trajectories]
        + [[t["exact1"], t["exact2"]] for t in trajectories if t["w1"] == t["w1"]]
        + [truth.x]
    )
    low, high = visited.min(axis=0), visited.max(axis=0)
    span = float((high - low).max()) * 1.3
    centre = (low + high) / 2
    windows = dict(
        full=(centre - span / 2, centre + span / 2),
        zoom=(truth.x - args.zoom * span, truth.x + args.zoom * span),
    )
    contours: list[dict] = []
    for panel, (lower, upper) in windows.items():
        x1 = np.linspace(lower[0], upper[0], args.grid)
        x2 = np.linspace(lower[1], upper[1], args.grid)
        mesh1, mesh2 = np.meshgrid(x1, x2)
        values = problem.objectives(np.column_stack([mesh1.ravel(), mesh2.ravel()]))
        contours.extend(
            dict(panel=panel, x1=float(a), x2=float(b), f_minus_fstar=float(v))
            for a, b, v in zip(
                mesh1.ravel(), mesh2.ravel(), values - truth.value, strict=True
            )
        )

    instance = dict(
        study=STUDY,
        seed=args.seed,
        n=2,
        m=args.m,
        threshold=args.threshold,
        ratio=args.ratio,
        delta=args.delta,
        near_exact_tolerance=args.near_exact_tolerance,
        gd_step=args.gd_step,
        smoothness=smoothness,
        gamma=gamma,
        alpha=alpha,
        c0=c0,
        sigma_star=sigma_star,
        f_star=truth.value,
        x_star_1=float(truth.x[0]),
        x_star_2=float(truth.x[1]),
        x0_1=0.0,
        x0_2=0.0,
        truth_status=truth.status,
        truth_tolerance=truth.tolerance,
        truth_seconds=truth.seconds,
        truth_stationarity=truth.stationarity,
        span=span,
        zoom=args.zoom,
        grid=args.grid,
    )
    write_rows(directory / "instance.csv", [instance])
    write_rows(directory / "runs.csv", runs)
    write_rows(directory / "trajectories.csv", trajectories)
    write_rows(directory / "contours.csv", contours)
    return directory


def main() -> None:
    args = build_parser().parse_args()
    if args.scan:
        names = None
        for row in scan(args.scan, args):
            if names is None:
                names = list(row)
                print(" ".join(f"{name:>14s}" for name in names))
            print(" ".join(f"{row[name]:>14.4g}" for name in names))
        return
    directory = Path(args.results_dir) / args.name
    if not args.plot_only:
        directory = run(args)
    from .plots import trajectory_2d, trajectory_2d_exact

    for path in trajectory_2d(directory) + trajectory_2d_exact(directory):
        print(path)


if __name__ == "__main__":
    main()
