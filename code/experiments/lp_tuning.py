"""Large-LP tuning study.

The preconditioner is parameterized using problem scales instead of raw
numbers.  If ``s_min`` and ``s_max`` are the extreme singular values of A,
we use

    gamma_x = x_scale / rms(c),
    gamma_x * gamma_lambda = product_scale / (s_min * s_max).

The second relation places the transition in ``I + gamma_x*gamma_lambda
A*A.T`` near the geometric center of the spectrum.  A moderate proxy problem
selects the preconditioner and then ``(sigma, theta)``; only the best settings
are carried to the large instance.

Writes ``results/lp_tuning/results.csv`` and ``histories.csv``; figures come
from :mod:`plots.lp_tuning`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rand_nla_fom import DRSResult, primal_dual_drs

from ._cli import base_parser
from ._metrics import ACCURACY_FLOOR, history_rows, normalized_objective_error
from ._paths import HISTORIES, RESULTS, study_dir
from ._results import write_rows
from .problems import nonnegative_linear_prox, standard_form_lp_optimum

STUDY = "lp_tuning"
PROXY_SIZE = 150
PROXY_SEED = 101
LARGE_SIZE = 500
LARGE_SEED = 202
PROXY_ITERATIONS = 4_000
LARGE_ITERATIONS = 30_000
TOLERANCE = 3e-7
PRECONDITIONER_GRID = tuple(
    (x_scale, product_scale)
    for x_scale in (0.25, 1.0, 4.0)
    for product_scale in (0.1, 1.0, 10.0)
)
RELAXATION_GRID = (
    (0.05, 1.0),
    (0.2, 1.0),
    (0.45, 1.0),
    (0.15, 0.7),
    (0.15, 1.3),
    (0.14, 1.7),
)
WINNERS = 3
BASELINE_LABEL = "fixed baseline"


@dataclass(frozen=True)
class LP:
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    optimum: float
    s_min: float
    s_max: float
    c_rms: float


@dataclass(frozen=True)
class Setting:
    label: str
    gamma_x: float
    gamma_lambda: float
    sigma: float
    theta: float


def make_lp(n: int, seed: int) -> LP:
    """Make a bounded standard-form LP with m=round(.2*n)."""
    rng = np.random.default_rng(seed)
    m = round(0.2 * n)
    # The last row bounds the nonnegative feasible set.  Scaling it to the
    # same row norm as the random rows avoids inserting an artificial outlier.
    A = np.vstack((rng.standard_normal((m - 1, n)), np.ones(n))) / np.sqrt(n)
    feasible = rng.uniform(0.2, 1.0, n)
    feasible /= feasible.sum()
    b = A @ feasible
    c = rng.standard_normal(n)
    s = np.linalg.svd(A, compute_uv=False)
    return LP(
        A,
        b,
        c,
        standard_form_lp_optimum(c, A, b),
        float(s[-1]),
        float(s[0]),
        float(np.linalg.norm(c) / np.sqrt(n)),
    )


def data_setting(
    lp: LP,
    x_scale: float,
    product_scale: float,
    sigma: float = 0.2,
    theta: float = 1.0,
) -> Setting:
    gamma_x = x_scale / lp.c_rms
    product = product_scale / (lp.s_min * lp.s_max)
    gamma_lambda = product / gamma_x
    return Setting(
        f"x={x_scale:g}, p={product_scale:g}", gamma_x, gamma_lambda, sigma, theta
    )


def solve(lp: LP, setting: Setting, max_iterations: int, tolerance: float) -> DRSResult:
    return primal_dual_drs(
        nonnegative_linear_prox(lp.c),
        lp.A,
        lp.b,
        gamma_x=setting.gamma_x,
        gamma_lambda=setting.gamma_lambda,
        sigma=setting.sigma,
        theta=setting.theta,
        linear_solver="schur_cg",
        objective=lambda x: float(lp.c @ x),
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def merit(lp: LP, result: DRSResult) -> np.ndarray:
    error = normalized_objective_error(result.objective_values, lp.optimum)
    return np.maximum(error + result.feasibility_norms, ACCURACY_FLOOR)


def score(lp: LP, result: DRSResult) -> tuple[float, int]:
    # Prefer accuracy first; cumulative inner work breaks practically equal
    # ties.  The log score prevents small numerical noise from dominating.
    return (
        float(np.log10(merit(lp, result)[-1])),
        int(result.inner_iterations.sum()),
    )


def tune(proxy: LP, max_iterations: int, tolerance: float):
    preconditioners = [
        data_setting(proxy, x_scale, product_scale)
        for x_scale, product_scale in PRECONDITIONER_GRID
    ]
    pre_runs = [
        (setting, solve(proxy, setting, max_iterations, tolerance))
        for setting in preconditioners
    ]
    best_m, _ = min(pre_runs, key=lambda item: score(proxy, item[1]))

    parameters = [
        Setting(
            f"sigma={sigma:g}, theta={theta:g}",
            best_m.gamma_x,
            best_m.gamma_lambda,
            sigma,
            theta,
        )
        for sigma, theta in RELAXATION_GRID
    ]
    parameter_runs = [
        (setting, solve(proxy, setting, max_iterations, tolerance))
        for setting in parameters
    ]
    ranked = sorted(parameter_runs, key=lambda item: score(proxy, item[1]))
    return pre_runs, parameter_runs, [item[0] for item in ranked[:WINNERS]]


def transfer(setting: Setting, source: LP, target: LP) -> Setting:
    """Transfer dimensionless scale factors from proxy to target data."""
    x_scale = setting.gamma_x * source.c_rms
    product_scale = setting.gamma_x * setting.gamma_lambda * source.s_min * source.s_max
    transferred = data_setting(
        target, x_scale, product_scale, setting.sigma, setting.theta
    )
    return Setting(
        setting.label,
        transferred.gamma_x,
        transferred.gamma_lambda,
        setting.sigma,
        setting.theta,
    )


def _keys(problem: str, stage: str, setting: Setting) -> dict:
    return dict(problem=problem, stage=stage, label=setting.label)


def result_row(
    problem: str, stage: str, lp: LP, setting: Setting, result: DRSResult
) -> dict:
    return dict(
        **_keys(problem, stage, setting),
        n=lp.A.shape[1],
        m=lp.A.shape[0],
        s_min=lp.s_min,
        s_max=lp.s_max,
        c_rms=lp.c_rms,
        gamma_x=setting.gamma_x,
        gamma_lambda=setting.gamma_lambda,
        sigma=setting.sigma,
        theta=setting.theta,
        converged=result.converged,
        outer_iterations=result.iterations,
        inner_iterations=int(result.inner_iterations.sum()),
        objective_error=abs(result.objective_values[-1] - lp.optimum),
        feasibility=result.feasibility_norms[-1],
        final_merit=merit(lp, result)[-1],
    )


def _tabulate(problem: str, stage: str, lp: LP, runs) -> tuple[list[dict], list[dict]]:
    results = [result_row(problem, stage, lp, s, r) for s, r in runs]
    histories = [
        row
        for setting, result in runs
        for row in history_rows(_keys(problem, stage, setting), result, lp.optimum)
    ]
    return results, histories


def run_study(args) -> None:
    output = study_dir(STUDY, args.results_dir)
    proxy = make_lp(args.proxy_size, args.proxy_seed)
    large = make_lp(args.large_size, args.large_seed)
    pre_runs, parameter_runs, winners = tune(
        proxy, args.proxy_iterations, args.tolerance
    )

    baseline = Setting(BASELINE_LABEL, 0.5, 1.0, 0.2, 1.0)
    large_settings = [baseline] + [transfer(s, proxy, large) for s in winners]
    large_runs = [
        (setting, solve(large, setting, args.large_iterations, args.tolerance))
        for setting in large_settings
    ]

    results, histories = [], []
    for problem, stage, lp, runs in (
        ("proxy", "preconditioner", proxy, pre_runs),
        ("proxy", "sigma_theta", proxy, parameter_runs),
        ("large", "transfer", large, large_runs),
    ):
        stage_results, stage_histories = _tabulate(problem, stage, lp, runs)
        results.extend(stage_results)
        histories.extend(stage_histories)
    write_rows(output / RESULTS, results)
    write_rows(output / HISTORIES, histories)

    print(
        f"proxy: n={proxy.A.shape[1]}, m={proxy.A.shape[0]}, "
        f"singular range=[{proxy.s_min:.3g}, {proxy.s_max:.3g}]"
    )
    print(
        f"large: n={large.A.shape[1]}, m={large.A.shape[0]}, "
        f"singular range=[{large.s_min:.3g}, {large.s_max:.3g}]"
    )
    for setting, result in large_runs:
        print(
            f"{setting.label:<24} gx={setting.gamma_x:.3g} "
            f"gl={setting.gamma_lambda:.3g} sigma={setting.sigma:g} "
            f"theta={setting.theta:g} outer={result.iterations:5d} "
            f"inner={result.inner_iterations.sum():7d} "
            f"merit={merit(large, result)[-1]:.2e}"
        )
    print(f"wrote {output}")


def build_parser():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--proxy-size", type=int, default=PROXY_SIZE)
    parser.add_argument("--proxy-seed", type=int, default=PROXY_SEED)
    parser.add_argument("--large-size", type=int, default=LARGE_SIZE)
    parser.add_argument("--large-seed", type=int, default=LARGE_SEED)
    parser.add_argument("--proxy-iterations", type=int, default=PROXY_ITERATIONS)
    parser.add_argument("--large-iterations", type=int, default=LARGE_ITERATIONS)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    return parser


def main(argv: list[str] | None = None) -> None:
    run_study(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
