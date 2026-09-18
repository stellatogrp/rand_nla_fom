"""Slack-form inequality-QP tuning study.

The slack formulation uses the Section 5 variable ``y=(x,s)``, constraint
matrix ``B=[A I]``, and proximal map

    prox_{gamma f}(x,s) = ((I + 2 gamma Q)^-1 x, max(s, 0)).

Dimensionless parameters are tuned on a proxy problem and transferred to a
larger instance using the RMS objective curvature and extreme singular values
of ``B``.  The transferred settings are then compared against progress-based
adaptation of ``sigma`` and ``theta``, which uses only step norms,
feasibility, recent contraction, and inner iteration counts -- never the
reference optimum.

Writes ``results/inequality_qp_tuning/results.csv`` and ``histories.csv``;
figures come from :mod:`plots.inequality_qp_tuning`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rand_nla_fom import DRSResult, primal_dual_drs

from ._cli import base_parser
from ._metrics import ACCURACY_FLOOR, history_rows, normalized_objective_error
from ._paths import HISTORIES, RESULTS, study_dir
from ._results import write_rows
from .problems import Prox, inequality_qp_optimum, slack_quadratic_prox

STUDY = "inequality_qp_tuning"
PROXY_SIZE = 100
PROXY_SEED = 311
LARGE_SIZE = 500
LARGE_SEED = 733
PROXY_ITERATIONS = 3_000
PROXY_TOLERANCE = 2e-7
LARGE_ITERATIONS = 12_000
LARGE_TOLERANCE = 3e-7
PRECONDITIONER_GRID = tuple(
    (x_scale, product_scale)
    for x_scale in (0.1, 0.3, 1.0)
    for product_scale in (0.1, 1.0, 10.0)
)
RELAXATION_GRID = (
    (0.05, 1.0),
    (0.2, 1.0),
    (0.45, 1.0),
    (0.1, 0.7),
    (0.1, 1.3),
    (0.1, 1.7),
)
WINNERS = 3


@dataclass(frozen=True)
class InequalityQP:
    A: np.ndarray
    b: np.ndarray
    Q: np.ndarray
    B: np.ndarray
    prox: Prox
    optimum: float
    curvature: float
    s_min: float
    s_max: float


@dataclass(frozen=True)
class Setting:
    label: str
    x_scale: float
    product_scale: float
    sigma: float
    theta: float
    adaptive: bool = False


def make_problem(n: int, seed: int, constraint_ratio: float = 0.2) -> InequalityQP:
    """Create a feasible dense strongly convex inequality QP."""
    rng = np.random.default_rng(seed)
    m = round(constraint_ratio * n)
    A = rng.standard_normal((m, n)) / np.sqrt(n)
    factor = rng.standard_normal((n, n)) / np.sqrt(n)
    Q = factor.T @ factor + 0.2 * np.eye(n)
    feasible = rng.standard_normal(n)
    slack = rng.uniform(0.05, 0.2, m)
    b = A @ feasible + slack
    B = np.hstack((A, np.eye(m)))

    singular_values = np.linalg.svd(B, compute_uv=False)
    return InequalityQP(
        A=A,
        b=b,
        Q=Q,
        B=B,
        prox=slack_quadratic_prox(Q),
        optimum=inequality_qp_optimum(Q, A, b),
        curvature=float(np.trace(2 * Q) / n),
        s_min=float(singular_values[-1]),
        s_max=float(singular_values[0]),
    )


def parameters(problem: InequalityQP, setting: Setting) -> tuple[float, float]:
    gamma_x = setting.x_scale / problem.curvature
    product = setting.product_scale / (problem.s_min * problem.s_max)
    return gamma_x, product / gamma_x


def solve(
    problem: InequalityQP,
    setting: Setting,
    max_iterations: int,
    tolerance: float,
) -> DRSResult:
    n = problem.Q.shape[0]
    gamma_x, gamma_lambda = parameters(problem, setting)
    return primal_dual_drs(
        problem.prox,
        problem.B,
        problem.b,
        gamma_x=gamma_x,
        gamma_lambda=gamma_lambda,
        sigma=setting.sigma,
        theta=setting.theta,
        linear_solver="schur_cg",
        adaptive=setting.adaptive,
        objective=lambda y: float(y[:n] @ problem.Q @ y[:n]),
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def violations(problem: InequalityQP, result: DRSResult) -> np.ndarray:
    """Return the inequality residual ``||max(Ax - b, 0)||`` per iterate."""
    n = problem.Q.shape[0]
    xs = result.primal_iterates[:, :n]
    return np.linalg.norm(np.maximum(xs @ problem.A.T - problem.b, 0.0), axis=1)


def diagnostics(
    problem: InequalityQP, result: DRSResult
) -> tuple[np.ndarray, np.ndarray]:
    inequality = violations(problem, result)
    error = normalized_objective_error(result.objective_values, problem.optimum)
    merit = np.maximum(error + inequality, ACCURACY_FLOOR)
    return merit, inequality


def work_score(problem: InequalityQP, result: DRSResult) -> tuple[float, int]:
    merit, _ = diagnostics(problem, result)
    # Accuracy is compared in half-decade bands; within a band prefer less work.
    return (
        round(float(np.log10(merit[-1])) * 2) / 2,
        int(result.inner_iterations.sum()),
    )


def tune(proxy: InequalityQP, max_iterations: int, tolerance: float):
    preconditioners = [
        Setting(f"x={x:g}, p={p:g}", x, p, 0.25, 1.0) for x, p in PRECONDITIONER_GRID
    ]
    pre_runs = [
        (s, solve(proxy, s, max_iterations, tolerance)) for s in preconditioners
    ]
    best = min(pre_runs, key=lambda pair: work_score(proxy, pair[1]))[0]
    relaxations = [
        Setting(
            f"sigma={sigma:g}, theta={theta:g}",
            best.x_scale,
            best.product_scale,
            sigma,
            theta,
        )
        for sigma, theta in RELAXATION_GRID
    ]
    parameter_runs = [
        (s, solve(proxy, s, max_iterations, tolerance)) for s in relaxations
    ]
    winners = [
        s
        for s, _ in sorted(parameter_runs, key=lambda pair: work_score(proxy, pair[1]))[
            :WINNERS
        ]
    ]
    return pre_runs, parameter_runs, winners


def _keys(problem_name: str, stage: str, setting: Setting) -> dict:
    return dict(
        problem=problem_name,
        stage=stage,
        label=setting.label,
        adaptive=setting.adaptive,
    )


def result_row(
    problem_name: str,
    stage: str,
    problem: InequalityQP,
    setting: Setting,
    result: DRSResult,
) -> dict:
    gamma_x, gamma_lambda = parameters(problem, setting)
    merit, inequality = diagnostics(problem, result)
    return dict(
        **_keys(problem_name, stage, setting),
        n=problem.Q.shape[0],
        m=problem.A.shape[0],
        gamma_x=gamma_x,
        gamma_lambda=gamma_lambda,
        x_scale=setting.x_scale,
        product_scale=setting.product_scale,
        sigma=setting.sigma,
        theta=setting.theta,
        converged=result.converged,
        outer_iterations=result.iterations,
        inner_iterations=int(result.inner_iterations.sum()),
        objective_error=abs(result.objective_values[-1] - problem.optimum),
        inequality_violation=inequality[-1],
        final_merit=merit[-1],
    )


def _tabulate(
    problem_name: str, stage: str, problem: InequalityQP, runs
) -> tuple[list[dict], list[dict]]:
    results = [result_row(problem_name, stage, problem, s, r) for s, r in runs]
    histories = [
        row
        for setting, result in runs
        for row in history_rows(
            _keys(problem_name, stage, setting),
            result,
            problem.optimum,
            violations(problem, result),
        )
    ]
    return results, histories


def run_study(args) -> None:
    output = study_dir(STUDY, args.results_dir)
    proxy = make_problem(args.proxy_size, args.proxy_seed)
    large = make_problem(args.large_size, args.large_seed)
    pre_runs, parameter_runs, winners = tune(
        proxy, args.proxy_iterations, args.proxy_tolerance
    )

    baseline = Setting("fixed baseline", 1.0, 1.0, 0.2, 1.0)
    tuned = winners[0]
    adaptive_settings = [
        Setting("adaptive baseline M", 1.0, 1.0, 0.2, 1.0, True),
        Setting("adaptive tuned M", tuned.x_scale, tuned.product_scale, 0.2, 1.0, True),
    ]
    large_settings = [baseline, *winners, *adaptive_settings]
    large_runs = [
        (s, solve(large, s, args.large_iterations, args.large_tolerance))
        for s in large_settings
    ]

    results, histories = [], []
    for problem_name, stage, problem, runs in (
        ("proxy", "preconditioner", proxy, pre_runs),
        ("proxy", "sigma_theta", proxy, parameter_runs),
        ("large", "transfer", large, large_runs),
    ):
        stage_results, stage_histories = _tabulate(problem_name, stage, problem, runs)
        results.extend(stage_results)
        histories.extend(stage_histories)
    write_rows(output / RESULTS, results)
    write_rows(output / HISTORIES, histories)

    for setting, result in large_runs:
        gamma_x, gamma_lambda = parameters(large, setting)
        merit, inequality = diagnostics(large, result)
        print(
            f"{setting.label:<24} gx={gamma_x:.3g} gl={gamma_lambda:.3g} "
            f"outer={result.iterations:5d} "
            f"inner={result.inner_iterations.sum():7d} "
            f"objerr={abs(result.objective_values[-1] - large.optimum):.2e} "
            f"ineq={inequality[-1]:.2e} merit={merit[-1]:.2e}"
        )
    print(f"wrote {output}")


def build_parser():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--proxy-size", type=int, default=PROXY_SIZE)
    parser.add_argument("--proxy-seed", type=int, default=PROXY_SEED)
    parser.add_argument("--large-size", type=int, default=LARGE_SIZE)
    parser.add_argument("--large-seed", type=int, default=LARGE_SEED)
    parser.add_argument("--proxy-iterations", type=int, default=PROXY_ITERATIONS)
    parser.add_argument("--proxy-tolerance", type=float, default=PROXY_TOLERANCE)
    parser.add_argument("--large-iterations", type=int, default=LARGE_ITERATIONS)
    parser.add_argument("--large-tolerance", type=float, default=LARGE_TOLERANCE)
    return parser


def main(argv: list[str] | None = None) -> None:
    run_study(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
