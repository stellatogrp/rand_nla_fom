"""LASSO in split form: metric, tolerance, inner-solver, size, and penalty studies.

The LASSO ``min ||D x - y||^2 / 2 + tau ||x||_1`` is written over the pair
``u = (x, z)`` with the residual ``z = D x - y`` as its own variable, so it
takes the form ``min f(u)`` subject to ``A u = b`` with

    f(x, z) = tau ||x||_1 + ||z||^2 / 2,   A = [D  -I],   b = y.

The proximal map of ``f`` is a soft threshold on ``x`` and a scaling of ``z``,
and ``A A^T = D D^T + I`` has every eigenvalue at least one, so the Schur
complement is well conditioned no matter how the columns of ``D`` correlate.
Two designs contrast this: independent Gaussian columns and columns following
an autoregressive correlation ``rho^|i-j|``, which makes ``D`` ill conditioned
without changing ``kappa(A)`` much.

The same split covers the hybrid penalty

    c1 ||D x - y||^2 + c2 ||D x - y||_1 + c3 ||x||_1 + c4 ||x||_2,

whose proximal map is still separable and closed form; the plain LASSO is
``(c1, c2, c3, c4) = (1/2, 0, tau, 0)``.

The stages, all on the same seeded instances:

``gamma``, ``sigma``
    sweep the shared metric ``gamma_x = gamma_lambda`` and the inner tolerance
    with block Kaczmarz, recording the outer and cumulative inner iterations at
    the first iterate meeting the target accuracy.  The tolerance grid starts
    above zero: ``sigma = 0`` asks Kaczmarz for a machine-precision solve,
    which costs tens of times more than any admissible tolerance and is covered
    by the ``sigma-sweep`` study.
``solver``
    compares the four inner solvers at one setting on both designs and keeps
    the convergence traces, whose per-iteration inner counts test the bounded
    expected inner work of Section 4.
``size``
    repeats the solver comparison on the independent design at growing
    ``(m, p)``.
``hybrid``
    repeats it for several coefficient vectors of the hybrid penalty.

Every reference optimum comes from CVXPY, see :mod:`experiments.problems`.

Writes ``results/lasso/results.csv`` and ``histories.csv``; figures come from
:mod:`plots.lasso`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rand_nla_fom import DRSResult, primal_dual_drs

from ._cli import base_parser
from ._metrics import accuracy_history, history_rows, target_index
from ._paths import HISTORIES, RESULTS, study_dir
from ._results import read_rows, write_rows
from .problems import (
    Coefficients,
    Prox,
    Vector,
    hybrid_optimum,
    hybrid_prox,
    hybrid_split_objective,
)

STUDY = "lasso"
SAMPLES = 100
FEATURES = 400
NONZEROS = 10
NOISE = 0.01
PENALTY_FRACTION = 0.1
CORRELATION = 0.9
SEED = 51
DESIGNS = ("independent", "correlated")
STAGES = ("gamma", "sigma", "solver", "size", "hybrid")
TARGET_ACCURACY = 1e-7
MAX_OUTER_ITERATIONS = 20_000
INNER_ITERATION_FACTOR = 500
GAMMA = 1.0
SIGMA = 0.2
THETA = 1.0
GAMMA_MIN = -1.0
GAMMA_MAX = 1.0
NUM_GAMMAS = 9
SIGMAS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)
SWEEP_SOLVER = "schur_block_kaczmarz"
SOLVERS = (
    "schur_cg",
    "schur_block_kaczmarz",
    "schur_randomized_kaczmarz",
    "coupled_gmres",
)
BLOCK_FRACTION = 0.05
KACZMARZ_SEED = 0
ZERO_LEVEL = 1e-6
# (samples, features, nonzeros) of the size stage; the first is the default
# instance of the other stages.
SIZES = ((100, 400, 10), (200, 800, 20), (400, 1_600, 40))
# Coefficients of the hybrid stage as (c1, c2, c3, c4), with c3 and c4 given
# as multiples of the LASSO penalty tau of the instance.
HYBRID_SIZE = (200, 800, 20)
HYBRID_PENALTIES = {
    "LASSO": (0.5, 0.0, 1.0, 0.0),
    "+ l2 norm": (0.5, 0.0, 1.0, 0.5),
    "+ l1 residual": (0.5, 0.05, 1.0, 0.0),
    "all four terms": (0.5, 0.05, 1.0, 0.5),
}


@dataclass(frozen=True)
class Lasso:
    design: str
    variant: str
    samples: int
    features: int
    matrix: Vector
    rhs: Vector
    prox: Prox
    objective: Callable[[Vector], float]
    optimum: float
    penalty: float
    coefficients: Coefficients
    design_condition: float
    matrix_condition: float
    solution_nonzeros: int
    zero_residuals: int


def make_lasso(
    design: str,
    samples: int = SAMPLES,
    features: int = FEATURES,
    nonzeros: int = NONZEROS,
    seed: int = SEED,
    noise: float = NOISE,
    penalty_fraction: float = PENALTY_FRACTION,
    correlation: float = CORRELATION,
    penalty: tuple[float, float, float, float] | None = None,
    variant: str | None = None,
) -> Lasso:
    """Draw a sparse regression instance and put it in split form.

    ``penalty`` gives the hybrid coefficients with ``c3`` and ``c4`` as
    multiples of ``tau = penalty_fraction * ||D^T y||_inf``; ``None`` is the
    plain LASSO.
    """
    rng = np.random.default_rng(seed)
    columns = rng.standard_normal((samples, features))
    if design == "correlated":
        # An AR(1) recursion across columns gives covariance rho^|i-j|.
        for index in range(1, features):
            columns[:, index] = (
                correlation * columns[:, index - 1]
                + np.sqrt(1.0 - correlation**2) * columns[:, index]
            )
    elif design != "independent":
        raise ValueError(f"unknown design {design!r}")
    columns /= np.sqrt(samples)

    truth = np.zeros(features)
    support = rng.choice(features, nonzeros, replace=False)
    truth[support] = rng.choice([-1.0, 1.0], nonzeros) * rng.uniform(0.5, 1.5, nonzeros)
    response = columns @ truth + noise * rng.standard_normal(samples)
    tau = penalty_fraction * float(np.abs(columns.T @ response).max())
    c1, c2, c3, c4 = penalty if penalty is not None else (0.5, 0.0, 1.0, 0.0)
    coefficients: Coefficients = (c1, c2, c3 * tau, c4 * tau)

    optimum, solution = hybrid_optimum(columns, response, coefficients)
    # An interior-point solution is sparse only to its tolerance, so entries
    # and residuals below this level count as zero.
    residual = columns @ solution - response
    solution_nonzeros = int(
        (np.abs(solution) > ZERO_LEVEL * np.abs(solution).max()).sum()
    )
    zero_residuals = int((np.abs(residual) < ZERO_LEVEL).sum())

    singular = np.linalg.svd(columns, compute_uv=False)
    return Lasso(
        design,
        variant if variant is not None else design,
        samples,
        features,
        np.hstack((columns, -np.eye(samples))),
        response,
        hybrid_prox(features, coefficients),
        hybrid_split_objective(features, coefficients),
        optimum,
        tau,
        coefficients,
        float(singular[0] / singular[-1]),
        float(np.sqrt((1.0 + singular[0] ** 2) / (1.0 + singular[-1] ** 2))),
        solution_nonzeros,
        zero_residuals,
    )


def block_size(problem: Lasso, fraction: float = BLOCK_FRACTION) -> int:
    return max(1, round(fraction * problem.samples))


def _keys(problem: Lasso, stage: str, label: str) -> dict:
    """Identify one run in both tables."""
    return dict(
        problem="LASSO",
        design=problem.design,
        stage=stage,
        variant=problem.variant,
        label=label,
    )


def run_setting(
    problem: Lasso,
    stage: str,
    label: str,
    gamma: float,
    sigma: float,
    theta: float,
    solver: str,
    args,
) -> tuple[dict, DRSResult | None]:
    """Measure the work one setting needs to reach the target accuracy."""
    c1, c2, c3, c4 = problem.coefficients
    common = dict(
        **_keys(problem, stage, label),
        gamma_x=gamma,
        gamma_lambda=gamma,
        sigma=sigma,
        theta=theta,
        inner_solver=solver,
        m=problem.samples,
        p=problem.features,
        n=problem.features + problem.samples,
        penalty=problem.penalty,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        optimum=problem.optimum,
        solution_nonzeros=problem.solution_nonzeros,
        zero_residuals=problem.zero_residuals,
        design_condition=problem.design_condition,
        matrix_condition=problem.matrix_condition,
        target_accuracy=args.target_accuracy,
        block_size=block_size(problem, args.block_fraction),
        random_seed=args.kaczmarz_seed,
    )
    started = time.perf_counter()
    try:
        result = primal_dual_drs(
            problem.prox,
            problem.matrix,
            problem.rhs,
            gamma_x=gamma,
            gamma_lambda=gamma,
            sigma=sigma,
            theta=theta,
            linear_solver=solver,
            objective=problem.objective,
            tolerance=args.target_accuracy / 100,
            max_iterations=args.max_outer_iterations,
            max_inner_iterations=INNER_ITERATION_FACTOR * problem.samples,
            block_size=block_size(problem, args.block_fraction),
            random_seed=args.kaczmarz_seed,
            store_iterates=False,
        )
    except RuntimeError as error:
        row = dict(
            **common,
            reached_accuracy=False,
            outer_iterations=np.nan,
            inner_iterations=np.nan,
            mean_inner_iterations=np.nan,
            max_inner_iterations=np.nan,
            achieved_accuracy=np.nan,
            solver_iterations=np.nan,
            runtime_seconds=time.perf_counter() - started,
            error=str(error),
        )
        return row, None

    accuracy = accuracy_history(problem.optimum, result)
    index = target_index(accuracy, args.target_accuracy)
    if index is None:
        counts = result.inner_iterations
        outer_iterations = inner_iterations = np.nan
        achieved_accuracy = float(accuracy[-1])
    else:
        counts = result.inner_iterations[: index + 1]
        outer_iterations = index + 1
        inner_iterations = int(counts.sum())
        achieved_accuracy = float(accuracy[index])
    row = dict(
        **common,
        reached_accuracy=index is not None,
        outer_iterations=outer_iterations,
        inner_iterations=inner_iterations,
        mean_inner_iterations=float(counts.mean()),
        max_inner_iterations=int(counts.max()),
        achieved_accuracy=achieved_accuracy,
        solver_iterations=result.iterations,
        runtime_seconds=time.perf_counter() - started,
        error="" if index is not None else "target accuracy not reached",
    )
    return row, result


def print_progress(row: dict) -> None:
    if row["reached_accuracy"]:
        status = (
            f"outer={row['outer_iterations']}, inner={row['inner_iterations']}, "
            f"mean inner/outer={row['mean_inner_iterations']:.2f}"
        )
    else:
        reason = row["error"] or f"best accuracy {row['achieved_accuracy']:.2e}"
        status = f"missed target ({reason})"
    print(
        f"LASSO {row['variant']:<16} {row['stage']:<6} {row['label']:<28} "
        f"{status}, time={row['runtime_seconds']:.1f}s",
        flush=True,
    )


def print_instance(problem: Lasso) -> None:
    extra = (
        f", nnz(x*)={problem.solution_nonzeros}, "
        f"zero residuals={problem.zero_residuals}"
    )
    print(
        f"LASSO {problem.variant}: m={problem.samples}, p={problem.features}, "
        f"c={tuple(round(c, 4) for c in problem.coefficients)}, "
        f"kappa(D)={problem.design_condition:.3g}, "
        f"kappa(A)={problem.matrix_condition:.3g}, F*={problem.optimum:.6g}{extra}",
        flush=True,
    )


def settings(stage: str, args) -> list[tuple[str, float, float, float, str]]:
    """Return ``(label, gamma, sigma, theta, solver)`` for every run of a stage."""
    if stage == "gamma":
        return [
            (f"gamma={gamma:.3g}", float(gamma), args.sigma, THETA, SWEEP_SOLVER)
            for gamma in np.logspace(args.gamma_min, args.gamma_max, args.num_gammas)
        ]
    if stage == "sigma":
        return [
            (f"sigma={sigma:.3g}", args.gamma, float(sigma), THETA, SWEEP_SOLVER)
            for sigma in args.sigmas
        ]
    return [(solver, args.gamma, args.sigma, THETA, solver) for solver in SOLVERS]


def run_stage(
    problem: Lasso, stage: str, args, results: list[dict], histories: list[dict]
) -> None:
    """Run every setting of ``stage`` on ``problem`` and record the rows."""
    for label, gamma, sigma, theta, solver in settings(stage, args):
        row, result = run_setting(
            problem, stage, label, gamma, sigma, theta, solver, args
        )
        results.append(row)
        print_progress(row)
        if stage in ("solver", "size", "hybrid") and result is not None:
            keys = _keys(problem, stage, label)
            histories.extend(history_rows(keys, result, problem.optimum))


def _checkpointed(path: Path, stages: list[str]) -> list[dict]:
    """Return the rows of an earlier run that belong to stages not run now."""
    return [row for row in read_rows(path) if row["stage"] not in stages]


def run_study(args) -> None:
    """Run the requested stages, rewriting both tables after every instance.

    Rows of stages not requested are kept from an earlier run, so the study
    can be completed stage by stage and resumed after an interruption.
    """
    output = study_dir(STUDY, args.results_dir)
    results = _checkpointed(output / RESULTS, args.stages)
    histories = _checkpointed(output / HISTORIES, args.stages)

    def checkpoint() -> None:
        write_rows(output / RESULTS, results)
        if histories:
            write_rows(output / HISTORIES, histories)

    design_stages = [stage for stage in args.stages if stage in DESIGN_STAGES]
    if design_stages:
        for design in args.designs:
            problem = make_lasso(
                design, args.samples, args.features, args.nonzeros, args.seed
            )
            print_instance(problem)
            for stage in design_stages:
                run_stage(problem, stage, args, results, histories)
                checkpoint()
    if "size" in args.stages:
        for samples, features, nonzeros in SIZES:
            problem = make_lasso(
                "independent",
                samples,
                features,
                nonzeros,
                args.seed,
                variant=f"{samples}x{features}",
            )
            print_instance(problem)
            run_stage(problem, "size", args, results, histories)
            checkpoint()
    if "hybrid" in args.stages:
        samples, features, nonzeros = HYBRID_SIZE
        for name, penalty in HYBRID_PENALTIES.items():
            problem = make_lasso(
                "independent",
                samples,
                features,
                nonzeros,
                args.seed,
                penalty=penalty,
                variant=name,
            )
            print_instance(problem)
            run_stage(problem, "hybrid", args, results, histories)
            checkpoint()
    if not results:
        print("nothing to do")
        return
    checkpoint()
    print(f"wrote {output}")


DESIGN_STAGES = ("gamma", "sigma", "solver")


def build_parser():
    parser = base_parser(__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--features", type=int, default=FEATURES)
    parser.add_argument("--nonzeros", type=int, default=NONZEROS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--designs",
        nargs="+",
        choices=DESIGNS,
        default=list(DESIGNS),
        metavar="DESIGN",
        help="column designs of the gamma, sigma, and solver stages",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
        metavar="STAGE",
        help="stages to run",
    )
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument(
        "--gamma-min", type=float, default=GAMMA_MIN, help="log10 of the smallest gamma"
    )
    parser.add_argument(
        "--gamma-max", type=float, default=GAMMA_MAX, help="log10 of the largest gamma"
    )
    parser.add_argument("--num-gammas", type=int, default=NUM_GAMMAS)
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=list(SIGMAS),
        metavar="SIGMA",
        help="inner tolerances of the sigma stage",
    )
    parser.add_argument("--block-fraction", type=float, default=BLOCK_FRACTION)
    parser.add_argument(
        "--kaczmarz-seed",
        type=int,
        default=KACZMARZ_SEED,
        help="seed for the Kaczmarz row sampling",
    )
    parser.add_argument(
        "--max-outer-iterations", type=int, default=MAX_OUTER_ITERATIONS
    )
    parser.add_argument("--target-accuracy", type=float, default=TARGET_ACCURACY)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if max(args.sigmas) >= (2 - THETA) / 2:
        raise SystemExit(
            f"every sigma must be below (2 - theta) / 2 = {(2 - THETA) / 2}"
        )
    run_study(args)


if __name__ == "__main__":
    main()
