"""Sweep the scalar metric ``gamma_x = gamma_lambda = gamma``.

Writes ``results/gamma_sweep/results.csv`` for the full sweep and, for the
focused large inequality-QP run, ``inequality_qp_results.csv`` plus the
``histories.csv`` traces of its ``gamma = 1`` runs.  Figures come from
:mod:`plots.gamma_sweep`.
"""

from __future__ import annotations

import numpy as np

from rand_nla_fom import DRSResult, primal_dual_drs

from ._cli import base_parser, kind_argument
from ._metrics import history_rows
from ._paths import HISTORIES, RESULTS, study_dir
from ._results import write_rows
from .sweep_problems import (
    CONDITIONS,
    SweepProblem,
    make_equality_qp,
    make_inequality_qp,
    make_lp,
)

STUDY = "gamma_sweep"
INEQUALITY_RESULTS = "inequality_qp_results.csv"
GAMMA_MIN = -2.0
GAMMA_MAX = 1.0
NUM_GAMMAS = 25
SIGMA = 0.2
THETA = 1.0
INNER_SOLVER = "schur_cg"
INNER_ITERATION_FACTOR = 4
KINDS = ("equality QP", "LP", "inequality QP")
MAKERS = {
    "equality QP": make_equality_qp,
    "LP": make_lp,
    "inequality QP": make_inequality_qp,
}
# The focused study traces this gamma, which every sweep grid contains.
TRACED_GAMMA = 1.0


def gammas(args) -> np.ndarray:
    return np.logspace(args.gamma_min, args.gamma_max, args.num_gammas)


def run_problem(
    problem: SweepProblem,
    values: np.ndarray,
    sigma: float = SIGMA,
    theta: float = THETA,
    captured: dict[float, DRSResult] | None = None,
) -> list[dict]:
    """Solve ``problem`` at every gamma, recording the work each one costs."""
    rows = []
    for gamma in values:
        gamma = float(gamma)
        common = dict(
            problem=problem.kind,
            matrix=problem.condition,
            gamma=gamma,
            sigma=sigma,
            theta=theta,
            inner_solver=INNER_SOLVER,
            n=problem.dimension,
            m=problem.matrix.shape[0],
            solver_n=problem.matrix.shape[1],
            matrix_condition=problem.matrix_condition,
        )
        try:
            result = primal_dual_drs(
                problem.prox,
                problem.matrix,
                problem.rhs,
                gamma_x=gamma,
                gamma_lambda=gamma,
                sigma=sigma,
                theta=theta,
                linear_solver=INNER_SOLVER,
                objective=problem.objective,
                tolerance=problem.solver_tolerance,
                max_iterations=problem.max_iterations,
                max_inner_iterations=(INNER_ITERATION_FACTOR * problem.matrix.shape[0]),
            )
        except RuntimeError as error:
            rows.append(
                dict(
                    **common,
                    converged=False,
                    outer_iterations=np.nan,
                    inner_iterations=np.nan,
                    mean_inner_iterations=np.nan,
                    objective_error=np.nan,
                    feasibility=np.nan,
                    error=str(error),
                )
            )
            continue
        rows.append(
            dict(
                **common,
                converged=result.converged,
                outer_iterations=result.iterations,
                inner_iterations=int(result.inner_iterations.sum()),
                mean_inner_iterations=float(result.inner_iterations.mean()),
                objective_error=abs(result.objective_values[-1] - problem.optimum),
                feasibility=result.feasibility_norms[-1],
                error="",
            )
        )
        if captured is not None:
            captured[gamma] = result
    return rows


def _report_best(rows: list[dict], kinds: list[str]) -> None:
    for kind in kinds:
        for condition in CONDITIONS:
            candidates = [
                row
                for row in rows
                if row["problem"] == kind
                and row["matrix"] == condition
                and row["converged"]
            ]
            if not candidates:
                print(f"{kind:<14} {condition:<16} no convergence within cap")
                continue
            best_outer = min(candidates, key=lambda row: row["outer_iterations"])
            best_inner = min(candidates, key=lambda row: row["inner_iterations"])
            print(
                f"{kind:<14} {condition:<16} "
                f"outer best gamma={best_outer['gamma']:.3g} "
                f"({best_outer['outer_iterations']})  "
                f"inner best gamma={best_inner['gamma']:.3g} "
                f"({best_inner['inner_iterations']})"
            )


def run_study(args) -> None:
    output = study_dir(STUDY, args.results_dir)
    values = gammas(args)
    rows = [
        row
        for kind in args.kinds
        for condition in CONDITIONS
        for row in run_problem(MAKERS[kind](condition), values, args.sigma, args.theta)
    ]
    write_rows(output / RESULTS, rows)
    _report_best(rows, args.kinds)
    print(f"wrote {output}")


def run_inequality_study(args) -> None:
    """Run only the large inequality-QP portion, keeping its traces."""
    output = study_dir(STUDY, args.results_dir)
    values = gammas(args)
    rows, histories = [], []
    for condition in CONDITIONS:
        problem = make_inequality_qp(condition)
        captured: dict[float, DRSResult] = {}
        rows.extend(run_problem(problem, values, args.sigma, args.theta, captured))
        traced = min(captured, key=lambda gamma: abs(gamma - TRACED_GAMMA))
        histories.extend(
            history_rows(
                dict(
                    problem=problem.kind,
                    matrix=condition,
                    gamma=traced,
                    label=condition,
                ),
                captured[traced],
                problem.optimum,
            )
        )
    write_rows(output / INEQUALITY_RESULTS, rows)
    write_rows(output / HISTORIES, histories)
    _report_best(rows, ["inequality QP"])
    print(f"wrote focused large-scale results to {output}")


def build_parser():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument(
        "--gamma-min", type=float, default=GAMMA_MIN, help="log10 of the smallest gamma"
    )
    parser.add_argument(
        "--gamma-max", type=float, default=GAMMA_MAX, help="log10 of the largest gamma"
    )
    parser.add_argument("--num-gammas", type=int, default=NUM_GAMMAS)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--theta", type=float, default=THETA)
    kind_argument(parser, KINDS)
    return parser


def main(argv: list[str] | None = None) -> None:
    run_study(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
