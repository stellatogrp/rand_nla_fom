"""Sweep the relative inner-solve tolerance with fixed relaxation.

For every admissible ``sigma`` the sweep records the outer and cumulative
inner block-Kaczmarz iterations at the first iterate whose normalized
objective error plus feasibility is at most the target.  Each setting is
checkpointed to ``results/sigma_sweep/results.csv`` as it finishes, so an
interrupted run resumes where it stopped.  Figures come from
:mod:`plots.sigma_sweep`.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from rand_nla_fom import primal_dual_drs

from ._cli import base_parser, kind_argument
from ._metrics import accuracy_history, target_index
from ._paths import RESULTS, study_dir
from ._results import merge_rows, read_rows, write_rows
from .sweep_problems import (
    CONDITIONS,
    SweepProblem,
    make_large_equality_qp,
    make_large_inequality_qp,
    make_large_lp,
)

STUDY = "sigma_sweep"
SIGMA_MAX = 0.49
NUM_SIGMAS = 21
THETA = 1.0
GAMMA = 1.0
TARGET_ACCURACY = 1e-7
INNER_SOLVER = "schur_block_kaczmarz"
BLOCK_FRACTION = 0.05
RANDOM_SEED = 0
INNER_ITERATION_FACTOR = 1_000
CONSTRAINTS = 500
VARIABLES = 1_000
KINDS = ("equality QP", "LP", "inequality QP")
MAKERS = {
    "equality QP": make_large_equality_qp,
    "LP": make_large_lp,
    "inequality QP": make_large_inequality_qp,
}


def sigmas(args) -> np.ndarray:
    return np.linspace(0.0, args.sigma_max, args.num_sigmas)


def block_size(problem: SweepProblem, fraction: float = BLOCK_FRACTION) -> int:
    """Use about ``fraction`` of the Schur-complement dimension."""
    return max(1, round(fraction * problem.matrix.shape[0]))


def run_problem(
    problem: SweepProblem,
    values: Iterable[float],
    args,
    *,
    report: bool = False,
) -> list[dict]:
    """Measure work at the first iterate attaining the target accuracy."""
    rows = []
    size = block_size(problem, args.block_fraction)
    for sigma in values:
        sigma = float(sigma)
        common = dict(
            problem=problem.kind,
            matrix=problem.condition,
            sigma=sigma,
            theta=THETA,
            gamma_x=GAMMA,
            gamma_lambda=GAMMA,
            target_accuracy=args.target_accuracy,
            n=problem.dimension,
            m=problem.matrix.shape[0],
            solver_n=problem.matrix.shape[1],
            matrix_condition=problem.matrix_condition,
            inner_solver=INNER_SOLVER,
            block_size=size,
            block_fraction=args.block_fraction,
            random_seed=args.seed,
        )
        started = time.perf_counter()
        try:
            result = primal_dual_drs(
                problem.prox,
                problem.matrix,
                problem.rhs,
                gamma_x=GAMMA,
                gamma_lambda=GAMMA,
                sigma=sigma,
                theta=THETA,
                linear_solver=INNER_SOLVER,
                objective=problem.objective,
                tolerance=args.target_accuracy / 100,
                max_iterations=problem.max_iterations,
                max_inner_iterations=(INNER_ITERATION_FACTOR * problem.matrix.shape[0]),
                block_size=size,
                random_seed=args.seed,
                store_iterates=False,
            )
        except RuntimeError as error:
            rows.append(
                dict(
                    **common,
                    reached_accuracy=False,
                    outer_iterations=np.nan,
                    inner_iterations=np.nan,
                    achieved_accuracy=np.nan,
                    solver_iterations=np.nan,
                    runtime_seconds=time.perf_counter() - started,
                    error=str(error),
                )
            )
            if report:
                print_progress(rows[-1])
            continue

        accuracy = accuracy_history(problem.optimum, result)
        index = target_index(accuracy, args.target_accuracy)
        if index is None:
            outer_iterations = inner_iterations = np.nan
            achieved_accuracy = float(accuracy[-1])
        else:
            outer_iterations = index + 1
            inner_iterations = int(result.inner_iterations[: index + 1].sum())
            achieved_accuracy = float(accuracy[index])

        rows.append(
            dict(
                **common,
                reached_accuracy=index is not None,
                outer_iterations=outer_iterations,
                inner_iterations=inner_iterations,
                achieved_accuracy=achieved_accuracy,
                solver_iterations=result.iterations,
                runtime_seconds=time.perf_counter() - started,
                error="",
            )
        )
        if report:
            print_progress(rows[-1])
    return rows


def print_progress(row: dict) -> None:
    status = (
        f"outer={row['outer_iterations']}, inner={row['inner_iterations']}"
        if row["reached_accuracy"]
        else f"failed: {row['error'] or 'target not reached'}"
    )
    print(
        f"{row['problem']} {row['matrix']} sigma={row['sigma']:.4g}: "
        f"{status}, time={row['runtime_seconds']:.1f}s",
        flush=True,
    )


def key(row: dict) -> tuple:
    return (row["problem"], row["matrix"], float(row["sigma"]))


def order(kinds: Iterable[str]):
    kind_rank = {kind: index for index, kind in enumerate(kinds)}
    condition_rank = {name: index for index, name in enumerate(CONDITIONS)}

    def sort_key(row: dict) -> tuple:
        return (
            kind_rank.get(row["problem"], len(kind_rank)),
            condition_rank.get(row["matrix"], len(condition_rank)),
            float(row["sigma"]),
        )

    return sort_key


def print_summary(rows: list[dict], kinds: Iterable[str]) -> None:
    for kind in kinds:
        for condition in CONDITIONS:
            reached = [
                row
                for row in rows
                if row["problem"] == kind
                and row["matrix"] == condition
                and row["reached_accuracy"]
            ]
            if not reached:
                print(f"{kind:<14} {condition:<24} target not reached")
                continue
            best_outer = min(reached, key=lambda row: row["outer_iterations"])
            best_inner = min(reached, key=lambda row: row["inner_iterations"])
            print(
                f"{kind:<14} {condition:<24} "
                f"outer best sigma={best_outer['sigma']:.4g} "
                f"({best_outer['outer_iterations']})  "
                f"inner best sigma={best_inner['sigma']:.4g} "
                f"({best_inner['inner_iterations']})"
            )


def _checkpointed(path: Path, args) -> list[dict]:
    """Return previous rows that were produced by this configuration."""
    if args.force:
        return []
    return [
        row
        for row in read_rows(path)
        if row.get("m") == args.constraints
        and row.get("n") == args.variables
        and row.get("inner_solver") == INNER_SOLVER
        and row.get("block_size") == round(args.block_fraction * args.constraints)
    ]


def run_study(args) -> None:
    """Run or resume the large block-Kaczmarz sweep."""
    output = study_dir(STUDY, args.results_dir)
    results_path = output / RESULTS
    rows = _checkpointed(results_path, args)
    completed = {key(row) for row in rows}
    sort_key = order(args.kinds)

    for kind in args.kinds:
        for condition in CONDITIONS:
            problem = None
            for sigma in sigmas(args):
                identity = (kind, condition, float(sigma))
                if identity in completed:
                    print(f"checkpointed {identity}", flush=True)
                    continue
                if problem is None:
                    problem = MAKERS[kind](condition, args.constraints, args.variables)
                rows = merge_rows(
                    rows,
                    run_problem(problem, [sigma], args, report=True),
                    key,
                    sort_key,
                )
                write_rows(results_path, rows)
                completed.add(identity)
    if not rows:
        print("nothing to do")
        return
    write_rows(results_path, sorted(rows, key=sort_key))
    print_summary(rows, args.kinds)
    print(f"wrote {output}")


def build_parser():
    parser = base_parser(__doc__.splitlines()[0], resumable=True)
    parser.add_argument("--num-sigmas", type=int, default=NUM_SIGMAS)
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=SIGMA_MAX,
        help=f"largest sigma; must stay below {(2 - THETA) / 2}",
    )
    parser.add_argument("--constraints", type=int, default=CONSTRAINTS)
    parser.add_argument("--variables", type=int, default=VARIABLES)
    parser.add_argument("--block-fraction", type=float, default=BLOCK_FRACTION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--target-accuracy", type=float, default=TARGET_ACCURACY)
    kind_argument(parser, KINDS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.sigma_max >= (2 - THETA) / 2:
        raise SystemExit(
            f"--sigma-max must be below (2 - theta) / 2 = {(2 - THETA) / 2} "
            "for Theorem 5 to apply"
        )
    run_study(args)


if __name__ == "__main__":
    main()
