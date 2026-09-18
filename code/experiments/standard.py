"""Standard parameter and inner-solver comparisons on small QP and LP instances.

Writes ``results/standard/results.csv`` (one row per setting) and
``results/standard/histories.csv`` (one row per setting and outer iteration).
The figures are drawn by :mod:`plots.standard`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rand_nla_fom import DRSResult, primal_dual_drs, quadratic_prox

from ._cli import base_parser
from ._metrics import history_rows
from ._paths import HISTORIES, RESULTS, study_dir
from ._results import write_rows
from .problems import (
    Prox,
    Vector,
    equality_qp_solution,
    nonnegative_linear_prox,
    standard_form_lp_optimum,
)

STUDY = "standard"
QP_SEED = 14
LP_SEED = 27
GROUPS = (
    (
        "M",
        (
            ("gx=.25, gl=1", 0.25, 1.0, 0.2, 1.0),
            ("gx=1, gl=1", 1.0, 1.0, 0.2, 1.0),
            ("gx=1, gl=.25", 1.0, 0.25, 0.2, 1.0),
        ),
    ),
    (
        "sigma",
        (
            ("sigma=.05", 0.5, 1.0, 0.05, 1.0),
            ("sigma=.25", 0.5, 1.0, 0.25, 1.0),
            ("sigma=.45", 0.5, 1.0, 0.45, 1.0),
        ),
    ),
    (
        "theta",
        (
            ("theta=.65", 0.5, 1.0, 0.15, 0.65),
            ("theta=1", 0.5, 1.0, 0.15, 1.0),
            ("theta=1.6", 0.5, 1.0, 0.15, 1.6),
        ),
    ),
)
SOLVERS = ("schur_cg", "coupled_gmres")
SOLVER_SETTING = (0.5, 1.0, 0.2, 1.0)


@dataclass(frozen=True)
class Problem:
    name: str
    matrix: Vector
    rhs: Vector
    prox: Prox
    objective: Callable[[Vector], float]
    optimum: float
    tolerance: float
    max_iterations: int


def make_qp(seed: int = QP_SEED) -> Problem:
    rng = np.random.default_rng(seed)
    constraints, variables = 12, 40
    matrix = rng.standard_normal((constraints, variables)) / np.sqrt(variables)
    factor = rng.standard_normal((variables, variables)) / np.sqrt(variables)
    hessian = factor.T @ factor + 0.35 * np.eye(variables)
    linear_term = rng.standard_normal(variables) / np.sqrt(variables)
    rhs = rng.standard_normal(constraints)
    solution = equality_qp_solution(hessian, linear_term, matrix, rhs)

    def objective(vector: Vector) -> float:
        return float(0.5 * vector @ hessian @ vector + linear_term @ vector)

    return Problem(
        "QP",
        matrix,
        rhs,
        quadratic_prox(hessian, linear_term),
        objective,
        objective(solution),
        2e-8,
        4_000,
    )


def make_lp(seed: int = LP_SEED) -> Problem:
    rng = np.random.default_rng(seed)
    constraints, variables = 9, 32
    matrix = np.vstack(
        (rng.standard_normal((constraints - 1, variables)), np.ones(variables))
    )
    feasible = rng.uniform(0.1, 1.0, variables)
    feasible /= feasible.sum()
    rhs = matrix @ feasible
    cost = rng.standard_normal(variables)

    def objective(vector: Vector) -> float:
        return float(cost @ vector)

    return Problem(
        "LP",
        matrix,
        rhs,
        nonnegative_linear_prox(cost),
        objective,
        standard_form_lp_optimum(cost, matrix, rhs),
        2e-7,
        12_000,
    )


def solve(
    problem: Problem,
    gamma_x: float,
    gamma_lambda: float,
    sigma: float,
    theta: float,
    solver: str,
) -> DRSResult:
    return primal_dual_drs(
        problem.prox,
        problem.matrix,
        problem.rhs,
        gamma_x=gamma_x,
        gamma_lambda=gamma_lambda,
        sigma=sigma,
        theta=theta,
        linear_solver=solver,
        objective=problem.objective,
        tolerance=problem.tolerance,
        max_iterations=problem.max_iterations,
    )


def _keys(problem: Problem, group: str, choice: str) -> dict:
    """Identify one run in both tables."""
    return dict(problem=problem.name, group=group, label=choice)


def result_row(
    problem: Problem,
    group: str,
    choice: str,
    gamma_x: float,
    gamma_lambda: float,
    sigma: float,
    theta: float,
    solver: str,
    result: DRSResult,
) -> dict:
    return dict(
        **_keys(problem, group, choice),
        gamma_x=gamma_x,
        gamma_lambda=gamma_lambda,
        sigma=sigma,
        theta=theta,
        solver=solver,
        converged=result.converged,
        outer_iterations=result.iterations,
        inner_iterations=int(result.inner_iterations.sum()),
        final_objective_error=abs(result.objective_values[-1] - problem.optimum),
        final_feasibility=result.feasibility_norms[-1],
        max_relative_error_ratio=float(result.relative_error_ratios.max()),
    )


def run_problem(problem: Problem) -> tuple[list[dict], list[dict]]:
    """Run every parameter group and every solver on one problem."""
    results, histories = [], []
    settings = [
        (group, label, gx, gl, sigma, theta, "schur_cg")
        for group, choices in GROUPS
        for label, gx, gl, sigma, theta in choices
    ]
    settings += [("solver", solver, *SOLVER_SETTING, solver) for solver in SOLVERS]
    for group, label, gx, gl, sigma, theta, solver in settings:
        result = solve(problem, gx, gl, sigma, theta, solver)
        results.append(
            result_row(problem, group, label, gx, gl, sigma, theta, solver, result)
        )
        histories.extend(
            history_rows(_keys(problem, group, label), result, problem.optimum)
        )
    return results, histories


def run_study(args) -> None:
    output = study_dir(STUDY, args.results_dir)
    results, histories = [], []
    for problem in (make_qp(args.qp_seed), make_lp(args.lp_seed)):
        problem_results, problem_histories = run_problem(problem)
        results.extend(problem_results)
        histories.extend(problem_histories)

    write_rows(output / RESULTS, results)
    write_rows(output / HISTORIES, histories)
    for row in results:
        print(
            f"{row['problem']:>2} {row['group']:<6} {row['label']:<18} "
            f"outer={row['outer_iterations']:5d} "
            f"inner={row['inner_iterations']:6d} "
            f"feas={row['final_feasibility']:.2e} "
            f"objerr={row['final_objective_error']:.2e}"
        )
    print(f"wrote {output}")


def build_parser():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--qp-seed", type=int, default=QP_SEED)
    parser.add_argument("--lp-seed", type=int, default=LP_SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    run_study(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
