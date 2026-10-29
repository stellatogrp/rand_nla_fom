"""The paths of the iterates on a two-dimensional rotation operator.

    python -m numerical_experiments.torsion_operator              # run and draw
    python -m numerical_experiments.torsion_operator --plot-only  # redraw

The operator is the linear map ``B(x) = U (x - x*)`` with ``U`` the rotation
by a quarter turn, so that ``B`` is monotone (``U + U^T = 0``) but not a
gradient.

The inexact proximal point method of :func:`numerical_experiments.ppm.inexact_ppm`
runs on it with Richardson iteration ``x <- x - tau e`` on the resolvent
residual ``e = gamma B(x) + x - w`` as the inner solver, which is what
gradient descent on the proximal subproblem reduces to when ``B`` is a
gradient.  ``U`` is normal, so the residual contracts by
``rho = ||I - tau (I + gamma U)||`` per step, minimized at
``tau = 1 / (1 + gamma^2)`` where ``rho^2 = gamma^2 / (1 + gamma^2)``; the
inner-contraction assumption of the paper then holds deterministically with
``alpha = rho^2`` and ``C_0 = 1``, and ``sigma*`` follows from the same formula
as in the main studies.

Why this operator: on a rotation the exact proximal point step satisfies
``||w - x*||^2 - ||J(w) - x*||^2 = ||w - J(w)||^2`` with equality, so the
one-step decrease lemma of the paper (Lemma 3.2) is tight, whereas on the
Huber instance the decrease is tens of times the guaranteed amount.

Every run is deterministic; ``--seed`` fixes the direction of ``x*``.  Tables in
``results/numerical_experiments/torsion_operator/`` mirror those of
:mod:`numerical_experiments.trajectory_2d`: ``instance.csv``, ``runs.csv``,
``trajectories.csv`` (with ``distance = ||x - x*||`` in place of ``f - f*``),
``contours.csv`` (``||x - x*||`` on a grid), and ``field.csv`` (the direction
of ``-B`` on a coarse grid, drawn as arrows).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import theory
from ._io import write_rows
from .aggregate import config_key
from .ppm import MAX_TOTAL_INNER, TIME_LIMIT, inexact_ppm
from .study import DEFAULT_RESULTS_DIR, DELTA, NEAR_EXACT_TOLERANCE, SIGMAS

STUDY = "torsion_operator"
Array = np.ndarray
nan = float("nan")
QUARTER_TURN = np.array([[0.0, -1.0], [1.0, 0.0]])


@dataclass
class TorsionOperator:
    """``B(x) = U (x - x*)`` with ``U`` the quarter turn."""

    x_star: Array

    @property
    def n(self) -> int:
        return 2

    @property
    def matrix(self) -> Array:
        return QUARTER_TURN

    def gradient(self, x: Array) -> Array:
        """``B(x)``; the name is the one :func:`inexact_ppm` calls on its problem."""
        return self.matrix @ (x - self.x_star)

    def subproblem_matrix(self, gamma: float) -> Array:
        """``I + gamma U``, the matrix of the resolvent system."""
        return np.eye(2) + gamma * self.matrix

    def resolvent(self, gamma: float, w: Array) -> Array:
        """``J(w)``, the solution of ``x + gamma B(x) = w``."""
        return np.linalg.solve(
            self.subproblem_matrix(gamma), w + gamma * self.matrix @ self.x_star
        )

    def richardson(self, gamma: float) -> tuple[float, float]:
        """The step ``tau`` minimizing ``rho = ||I - tau (I + gamma U)||``, and ``rho``.

        ``I + gamma U`` is normal with eigenvalues ``1 +- i gamma``, so
        ``rho(tau)^2 = (1 - tau)^2 + (tau gamma)^2``.
        """
        tau = 1.0 / (1.0 + gamma**2)
        return tau, abs(gamma) / math.hypot(1.0, gamma)


def make_operator(seed: int, distance: float) -> TorsionOperator:
    """``x*`` at ``distance`` from the origin, in a direction drawn from ``seed``."""
    angle = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi)
    return TorsionOperator(distance * np.array([math.cos(angle), math.sin(angle)]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="iterate paths on a two-dimensional rotation operator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0, help="direction of x*")
    parser.add_argument(
        "--distance", type=float, default=5.0, help="||x* - x0||, with x0 = 0"
    )
    parser.add_argument(
        "--gamma", type=float, default=2.0, help="stepsize of the proximal point method"
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
        "--max-total-inner",
        type=int,
        default=MAX_TOTAL_INNER,
        help="hard stop: total inner steps of one proximal run",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=TIME_LIMIT,
        help="hard stop: wall-clock seconds of one run",
    )
    parser.add_argument(
        "--exact-sigma",
        type=float,
        default=0.9,
        help="the inexact run that trajectory_2d_exact singles out",
    )
    parser.add_argument(
        "--grid", type=int, default=201, help="contour grid points per axis"
    )
    parser.add_argument(
        "--field", type=int, default=13, help="arrows per axis of the direction field"
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=0.1,
        help="half-width of the zoom window as a fraction of the full view",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--name", default=STUDY, help="subdirectory of the results directory"
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="only redraw from the tables"
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    """Run every method on the operator, and write the tables."""
    directory = Path(args.results_dir) / args.name
    directory.mkdir(parents=True, exist_ok=True)
    operator = make_operator(args.seed, args.distance)
    gamma = args.gamma
    tau, rho = operator.richardson(gamma)
    alpha, c0 = rho**2, 1.0
    sigma_star = theory.optimal_sigma(alpha, c0, 1)
    x0 = np.zeros(2)

    configs = [("near_exact", nan, False)]
    configs += [("inexact", float(sigma), False) for sigma in args.sigmas]
    configs += [("inexact", sigma_star, True)]
    runs: list[dict] = []
    trajectories: list[dict] = []
    for method, sigma, optimal in configs:
        # The smoothness argument only sets the gradient step, which
        # ``inner_step`` replaces by the Richardson step.
        trace = inexact_ppm(
            operator,
            gamma,
            1.0,
            0.0 if method == "near_exact" else sigma,
            1,
            args.delta,
            exact_tolerance=(
                args.near_exact_tolerance if method == "near_exact" else None
            ),
            max_total_inner=args.max_total_inner,
            time_limit=None if method == "near_exact" else args.time_limit,
            inner_step=lambda x, w, e: x - tau * e,
        )
        points = [x0] + list(trace.iterates)
        centers = [None] + list(trace.w)
        cumulative = np.concatenate([[0], np.cumsum(trace.inner)])
        identity = dict(
            method=method,
            sweep="sigma" if method == "inexact" else "reference",
            sigma=sigma,
            b=1,
            sigma_is_optimal=optimal,
        )
        runs.append(
            dict(
                **identity,
                converged=trace.converged,
                stopped_by=trace.stopped_by,
                k_delta=trace.k_delta,
                total_inner=int(sum(trace.inner)),
                gradient_evaluations=trace.gradient_evaluations,
                runtime_seconds=trace.seconds,
            )
        )
        for k, (x, w, n) in enumerate(zip(points, centers, cumulative, strict=True)):
            row = dict(
                **identity,
                iteration=k,
                x1=float(x[0]),
                x2=float(x[1]),
                distance=float(np.linalg.norm(x - operator.x_star)),
                z_norm=float(np.linalg.norm(operator.gradient(x))),
                cumulative_inner=int(n),
                w1=nan,
                w2=nan,
                exact1=nan,
                exact2=nan,
                inexactness=nan,
            )
            if w is not None:
                exact = operator.resolvent(gamma, w)
                row.update(
                    w1=float(w[0]),
                    w2=float(w[1]),
                    exact1=float(exact[0]),
                    exact2=float(exact[1]),
                    inexactness=float(np.linalg.norm(x - exact)),
                )
            trajectories.append(row)

    # The full view holds every iterate of the runs that converged and the
    # zero; the zoom window is centered on the zero.  Both are square so that
    # the geometry is true.
    converged = {config_key(r) for r in runs if r["converged"]}
    kept = [t for t in trajectories if config_key(t) in converged]
    visited = np.array(
        [[t["x1"], t["x2"]] for t in kept]
        + [[t["exact1"], t["exact2"]] for t in kept if t["w1"] == t["w1"]]
        + [operator.x_star]
    )
    low, high = visited.min(axis=0), visited.max(axis=0)
    span = float((high - low).max()) * 1.3
    center = (low + high) / 2
    windows = dict(
        full=(center - span / 2, center + span / 2),
        zoom=(operator.x_star - args.zoom * span, operator.x_star + args.zoom * span),
    )
    contours: list[dict] = []
    field: list[dict] = []
    for panel, (lower, upper) in windows.items():
        x1 = np.linspace(lower[0], upper[0], args.grid)
        x2 = np.linspace(lower[1], upper[1], args.grid)
        mesh1, mesh2 = np.meshgrid(x1, x2)
        distance = np.hypot(mesh1 - operator.x_star[0], mesh2 - operator.x_star[1])
        contours.extend(
            dict(panel=panel, x1=float(a), x2=float(b), distance=float(v))
            for a, b, v in zip(
                mesh1.ravel(), mesh2.ravel(), distance.ravel(), strict=True
            )
        )
        # Arrows on an interior grid, so that none sits on the frame.
        f1 = np.linspace(lower[0], upper[0], args.field + 2)[1:-1]
        f2 = np.linspace(lower[1], upper[1], args.field + 2)[1:-1]
        mesh1, mesh2 = np.meshgrid(f1, f2)
        for a, b in zip(mesh1.ravel(), mesh2.ravel(), strict=True):
            direction = -operator.gradient(np.array([a, b]))
            length = float(np.linalg.norm(direction))
            if length > 0.0:
                field.append(
                    dict(
                        panel=panel,
                        x1=float(a),
                        x2=float(b),
                        v1=float(direction[0] / length),
                        v2=float(direction[1] / length),
                    )
                )

    instance = dict(
        study=STUDY,
        seed=args.seed,
        n=2,
        distance=args.distance,
        delta=args.delta,
        near_exact_tolerance=args.near_exact_tolerance,
        gamma=gamma,
        richardson_step=tau,
        rho=rho,
        alpha=alpha,
        c0=c0,
        sigma_star=sigma_star,
        x_star_1=float(operator.x_star[0]),
        x_star_2=float(operator.x_star[1]),
        x0_1=0.0,
        x0_2=0.0,
        span=span,
        zoom=args.zoom,
        grid=args.grid,
        field=args.field,
    )
    write_rows(directory / "instance.csv", [instance])
    write_rows(directory / "runs.csv", runs)
    write_rows(directory / "trajectories.csv", trajectories)
    write_rows(directory / "contours.csv", contours)
    write_rows(directory / "field.csv", field)
    return directory


def main() -> None:
    args = build_parser().parse_args()
    directory = Path(args.results_dir) / args.name
    if not args.plot_only:
        directory = run(args)
    from .plots import trajectory_figures

    for path in trajectory_figures(directory, args.exact_sigma):
        print(path)


if __name__ == "__main__":
    main()
