"""Compare inner linear solvers over many random problem instances.

The preconditioner scales, the relaxation, and the inner tolerance are held
fixed at ``gamma_x = gamma_lambda = gamma``, ``theta``, and ``sigma``; the only
thing that varies within one problem family is the solver used for equation
(58).  For every family the study draws ``--instances`` independent problems of
the same size and records, for each solver, the outer and cumulative inner
iterations needed to reach a fixed accuracy, so the spread over instances can
be shown with error bars.

By default the instances of one family share a fixed geometric singular-value
ladder, so they differ only in their singular vectors and their solution
draws.  ``--randomize-spectrum`` pins the extreme singular values, leaving
``kappa(A)`` unchanged, and redraws the interior log-uniformly per instance;
its results go to a separate study directory and carry a
``randomize_spectrum`` column so the two samplings stay separable.

Instances run on a process pool with one BLAS thread each, so the recorded
times are comparable across solvers but measured under load.  Every instance
is checkpointed to ``results.csv``; figures come from
:mod:`plots.solver_sweep`.
"""

from __future__ import annotations

import os
import time
from multiprocessing import get_context
from pathlib import Path

# Worker processes are spawned, so they re-import this module and pick these
# up before NumPy loads: one BLAS thread each keeps the recorded times
# comparable across solvers.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402

from rand_nla_fom import primal_dual_drs  # noqa: E402

from ._cli import base_parser, kind_argument  # noqa: E402
from ._metrics import accuracy_history, target_index  # noqa: E402
from ._paths import RESULTS, SUMMARY, study_dir  # noqa: E402
from ._results import read_rows, select, write_rows  # noqa: E402
from .sweep_problems import (  # noqa: E402
    CONDITIONS,
    SweepProblem,
    make_large_equality_qp,
    make_large_inequality_qp,
    make_large_lp,
)

STUDY = "solver_sweep"
RANDOMIZED_STUDY = "solver_sweep_randomized_spectrum"
CONSTRAINTS = 100
VARIABLES = 500
GAMMA = 1.0
THETA = 1.0
SIGMA = 0.2
TARGET_ACCURACY = 1e-7
INSTANCES = 100
BASE_SEED = 1_000
MAX_OUTER_ITERATIONS = 20_000
INNER_ITERATION_FACTOR = 1_000
BLOCK_FRACTION = 0.05
KACZMARZ_SEED = 0
WORKERS = 6

KINDS = ("equality QP", "LP", "inequality QP")
MAKERS = {
    "equality QP": make_large_equality_qp,
    "LP": make_large_lp,
    "inequality QP": make_large_inequality_qp,
}
SOLVERS = (
    "schur_cg",
    "schur_block_kaczmarz",
    "schur_randomized_kaczmarz",
    "coupled_gmres",
)
STATISTICS = ("mean", "sd", "sem", "median", "low", "high")
AGGREGATED = {
    "outer_iterations": "outer",
    "inner_iterations": "inner",
    "runtime_seconds": "time",
}


def study_name(randomize_spectrum: bool) -> str:
    return RANDOMIZED_STUDY if randomize_spectrum else STUDY


def block_size(problem: SweepProblem, fraction: float = BLOCK_FRACTION) -> int:
    """Use about ``fraction`` of the Schur-complement dimension."""
    return max(1, round(fraction * problem.matrix.shape[0]))


def run_instance(problem: SweepProblem, seed: int, solver: str, settings: dict) -> dict:
    """Measure the work one solver needs on one problem instance."""
    size = block_size(problem, settings["block_fraction"])
    target = settings["target_accuracy"]
    common = dict(
        problem=problem.kind,
        matrix=problem.condition,
        inner_solver=solver,
        instance_seed=seed,
        sigma=settings["sigma"],
        theta=settings["theta"],
        gamma_x=settings["gamma"],
        gamma_lambda=settings["gamma"],
        target_accuracy=target,
        n=problem.dimension,
        m=problem.matrix.shape[0],
        solver_n=problem.matrix.shape[1],
        matrix_condition=problem.matrix_condition,
        block_size=size,
        block_fraction=settings["block_fraction"],
        random_seed=settings["kaczmarz_seed"],
    )
    started = time.perf_counter()
    try:
        result = primal_dual_drs(
            problem.prox,
            problem.matrix,
            problem.rhs,
            gamma_x=settings["gamma"],
            gamma_lambda=settings["gamma"],
            sigma=settings["sigma"],
            theta=settings["theta"],
            linear_solver=solver,
            objective=problem.objective,
            tolerance=target / 100,
            max_iterations=settings["max_outer_iterations"],
            max_inner_iterations=(INNER_ITERATION_FACTOR * problem.matrix.shape[0]),
            block_size=size,
            random_seed=settings["kaczmarz_seed"],
            store_iterates=False,
        )
    except RuntimeError as error:
        return dict(
            **common,
            reached_accuracy=False,
            outer_iterations=np.nan,
            inner_iterations=np.nan,
            achieved_accuracy=np.nan,
            solver_iterations=np.nan,
            runtime_seconds=time.perf_counter() - started,
            error=str(error),
        )

    accuracy = accuracy_history(problem.optimum, result)
    index = target_index(accuracy, target)
    if index is None:
        outer_iterations = inner_iterations = np.nan
        achieved_accuracy = float(accuracy[-1])
    else:
        outer_iterations = float(index + 1)
        inner_iterations = float(result.inner_iterations[: index + 1].sum())
        achieved_accuracy = float(accuracy[index])

    return dict(
        **common,
        reached_accuracy=index is not None,
        outer_iterations=outer_iterations,
        inner_iterations=inner_iterations,
        achieved_accuracy=achieved_accuracy,
        solver_iterations=float(result.iterations),
        runtime_seconds=time.perf_counter() - started,
        error="" if index is not None else "target accuracy not reached",
    )


def _run_task(task: tuple) -> list[dict]:
    """Run the requested solvers on a single freshly drawn instance."""
    kind, condition, seed, solvers, settings = task
    problem = MAKERS[kind](
        condition,
        settings["constraints"],
        settings["variables"],
        seed=seed,
        randomize_spectrum=settings["randomize_spectrum"],
    )
    rows = [run_instance(problem, seed, solver, settings) for solver in solvers]
    for row in rows:
        row["randomize_spectrum"] = settings["randomize_spectrum"]
    return rows


def key(row: dict) -> tuple:
    return (
        row["problem"],
        row["matrix"],
        int(row["instance_seed"]),
        row["inner_solver"],
    )


def order(kinds) -> object:
    kind_rank = {kind: index for index, kind in enumerate(kinds)}
    condition_rank = {name: index for index, name in enumerate(CONDITIONS)}
    solver_rank = {name: index for index, name in enumerate(SOLVERS)}

    def sort_key(row: dict) -> tuple:
        return (
            kind_rank.get(row["problem"], len(kind_rank)),
            condition_rank.get(row["matrix"], len(condition_rank)),
            solver_rank.get(row["inner_solver"], len(solver_rank)),
            int(row["instance_seed"]),
        )

    return sort_key


def statistics(values: np.ndarray) -> dict:
    """Return mean, spread, and order statistics of a sample."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return dict.fromkeys(STATISTICS, np.nan) | {"count": 0}
    spread = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return dict(
        count=int(values.size),
        mean=float(values.mean()),
        sd=spread,
        sem=spread / np.sqrt(values.size),
        median=float(np.median(values)),
        low=float(values.min()),
        high=float(values.max()),
    )


def summarize(rows: list[dict], kinds) -> list[dict]:
    """Aggregate per-instance results into one row per solver and family."""
    summary = []
    for kind in kinds:
        for condition in CONDITIONS:
            for solver in SOLVERS:
                selected = select(
                    rows, problem=kind, matrix=condition, inner_solver=solver
                )
                if not selected:
                    continue
                reached = [row for row in selected if row["reached_accuracy"]]
                entry = dict(
                    problem=kind,
                    matrix=condition,
                    inner_solver=solver,
                    instances=len(selected),
                    reached=len(reached),
                )
                for field, prefix in AGGREGATED.items():
                    stats = statistics(
                        np.array([row[field] for row in reached], dtype=float)
                    )
                    for statistic in STATISTICS:
                        entry[f"{prefix}_{statistic}"] = stats[statistic]
                summary.append(entry)
    return summary


def print_summary(summary: list[dict]) -> None:
    header = (
        f"{'problem':<14}{'matrix':<24}{'solver':<28}"
        f"{'reached':>9}{'outer mean+-sd':>22}{'inner mean+-sd':>26}"
    )
    print(header)
    print("-" * len(header))
    for entry in summary:
        if entry["reached"] == 0:
            outer = inner = "all failed"
        else:
            outer = f"{entry['outer_mean']:.1f} +- {entry['outer_sd']:.1f}"
            inner = f"{entry['inner_mean']:.1f} +- {entry['inner_sd']:.1f}"
        print(
            f"{entry['problem']:<14}{entry['matrix']:<24}"
            f"{entry['inner_solver']:<28}"
            f"{entry['reached']:>4}/{entry['instances']:<4}"
            f"{outer:>22}{inner:>26}"
        )


def _settings(args) -> dict:
    """Bundle the run configuration passed to every worker process."""
    return dict(
        constraints=args.constraints,
        variables=args.variables,
        gamma=args.gamma,
        theta=args.theta,
        sigma=args.sigma,
        target_accuracy=args.target_accuracy,
        max_outer_iterations=args.max_outer_iterations,
        block_fraction=args.block_fraction,
        kaczmarz_seed=args.seed,
        randomize_spectrum=args.randomize_spectrum,
    )


def _checkpointed(path: Path, args) -> list[dict]:
    if args.force:
        return []
    return [
        row
        for row in read_rows(path)
        if row.get("m") == args.constraints
        and row.get("n") == args.variables
        and row.get("sigma") == args.sigma
        and row.get("theta") == args.theta
        and row.get("gamma_x") == args.gamma
        and bool(row.get("randomize_spectrum")) == args.randomize_spectrum
    ]


def run_study(args) -> None:
    """Run or resume the multi-instance inner-solver comparison."""
    output = study_dir(study_name(args.randomize_spectrum), args.results_dir)
    results_path = output / RESULTS
    rows = _checkpointed(results_path, args)
    completed = {key(row) for row in rows}
    sort_key = order(args.kinds)
    settings = _settings(args)

    tasks = []
    for kind in args.kinds:
        for condition in CONDITIONS:
            for index in range(args.instances):
                seed = args.base_seed + index
                pending = tuple(
                    solver
                    for solver in args.solvers
                    if (kind, condition, seed, solver) not in completed
                )
                if pending:
                    tasks.append((kind, condition, seed, pending, settings))
    print(
        f"{len(rows)} checkpointed results; "
        f"{len(tasks)} instances still to run on {args.workers} workers",
        flush=True,
    )

    if tasks:
        started = time.perf_counter()
        context = get_context("spawn")
        with context.Pool(args.workers) as pool:
            for finished, produced in enumerate(
                pool.imap_unordered(_run_task, tasks), start=1
            ):
                replaced = {key(new) for new in produced}
                rows = [row for row in rows if key(row) not in replaced]
                rows.extend(produced)
                write_rows(results_path, sorted(rows, key=sort_key))
                sample = produced[0]
                elapsed = time.perf_counter() - started
                print(
                    f"[{finished}/{len(tasks)}] {sample['problem']} "
                    f"{sample['matrix']} seed={sample['instance_seed']} "
                    f"({elapsed / finished:.1f}s/instance, "
                    f"{elapsed / 60:.1f} min elapsed)",
                    flush=True,
                )

    if not rows:
        print("nothing to do")
        return
    rows = sorted(rows, key=sort_key)
    write_rows(results_path, rows)
    summary = summarize(rows, args.kinds)
    write_rows(output / SUMMARY, summary)
    print_summary(summary)
    print(f"wrote {output}")


def build_parser():
    parser = base_parser(__doc__.splitlines()[0], resumable=True)
    parser.add_argument("--instances", type=int, default=INSTANCES)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--constraints", type=int, default=CONSTRAINTS)
    parser.add_argument("--variables", type=int, default=VARIABLES)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--theta", type=float, default=THETA)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--block-fraction", type=float, default=BLOCK_FRACTION)
    parser.add_argument(
        "--seed",
        type=int,
        default=KACZMARZ_SEED,
        help="seed for the Kaczmarz row sampling",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument(
        "--max-outer-iterations", type=int, default=MAX_OUTER_ITERATIONS
    )
    parser.add_argument("--target-accuracy", type=float, default=TARGET_ACCURACY)
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=SOLVERS,
        default=list(SOLVERS),
        metavar="SOLVER",
        help="inner solvers to compare",
    )
    parser.add_argument(
        "--randomize-spectrum",
        action="store_true",
        help="redraw the interior singular values for every instance",
    )
    kind_argument(parser, KINDS)
    return parser


def main(argv: list[str] | None = None) -> None:
    run_study(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
