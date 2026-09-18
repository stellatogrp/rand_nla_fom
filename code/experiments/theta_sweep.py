"""Sweep the relaxation parameter ``theta`` at a fixed unit metric.

The metric is pinned to ``gamma_x = gamma_lambda = 1`` so that the only free
outer parameter is the over/under-relaxation ``theta``.  Equation (58) is
solved with randomized Kaczmarz on the Schur complement: one row per inner
step, drawn with probability proportional to its squared norm.

Theorem 5 couples the two parameters through ``sigma < (2 - theta) / 2``, so a
single ``sigma`` cannot be held fixed across the whole admissible range of
``theta`` unless it is small.  The sweep therefore runs two policies:

``fixed``
    ``sigma = 0.04``, admissible for every ``theta`` tested, which isolates
    the effect of ``theta`` alone.
``frontier``
    ``sigma = 0.9 * (2 - theta) / 2`` capped at ``0.45``, the loosest inner
    solve each ``theta`` admits, which shows the price over-relaxation pays
    in inner accuracy.

Each setting is repeated over ``--trials`` independent trials.  A trial
redraws the problem instance -- singular vectors and, through
``randomize_spectrum``, the interior singular values -- along with the
Kaczmarz row stream, so the spread covers problem-to-problem variation as
well as solver randomness.  The extreme singular values stay pinned, so every
trial has exactly the stated ``kappa(A)``.

Randomized Kaczmarz contracts by roughly
``1 - sigma_min(S)^2 / ||S||_F^2`` per row projection, so its inner cost is
set by the Frobenius-scaled condition number of the Schur complement rather
than by ``kappa(S)``.

Results are checkpointed per trial to ``results/theta_sweep/results.csv``;
figures come from :mod:`plots.theta_sweep`.
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
    Condition,
    SweepProblem,
    make_large_equality_qp,
    make_large_inequality_qp,
    make_large_lp,
)

STUDY = "theta_sweep"
THETA_MIN = 0.1
THETA_MAX = 1.9
THETA_STEP = 0.1
GAMMA = 1.0
CONSTRAINTS = 50
VARIABLES = 200
TARGET_ACCURACY = 1e-7
INNER_SOLVER = "schur_randomized_kaczmarz"
MAX_OUTER_ITERATIONS = 6_000
INNER_ITERATION_FACTOR = 500
FIXED_SIGMA = 0.04
FRONTIER_FRACTION = 0.9
FRONTIER_CAP = 0.45
POLICIES = ("fixed", "frontier")
TRIALS = 3
# Separates the problem seeds of different trials and different families.
SEED_STRIDE = 1_000
CONDITIONS: tuple[Condition, ...] = (
    "moderately-conditioned",
    "mildly-ill-conditioned",
)
KINDS = ("equality QP", "LP", "inequality QP")
MAKERS = {
    "equality QP": (make_large_equality_qp, 41),
    "LP": (make_large_lp, 42),
    "inequality QP": (make_large_inequality_qp, 43),
}
THETAS = np.round(np.arange(THETA_MIN, THETA_MAX + 1e-9, THETA_STEP), 2)


def sigma_for(policy: str, theta: float) -> float:
    """Return the inner tolerance this policy assigns to ``theta``."""
    if policy == "fixed":
        return FIXED_SIGMA
    return min(FRONTIER_CAP, FRONTIER_FRACTION * (2.0 - theta) / 2.0)


def thetas(args) -> np.ndarray:
    return np.round(
        np.arange(args.theta_min, args.theta_max + 1e-9, args.theta_step), 4
    )


def make_trial(kind: str, condition: Condition, trial: int, args) -> SweepProblem:
    """Build the problem instance used by one trial."""
    maker, base_seed = MAKERS[kind]
    return maker(
        condition,
        args.constraints,
        args.variables,
        seed=base_seed + SEED_STRIDE * trial,
        randomize_spectrum=True,
    )


def run_problem(
    problem: SweepProblem,
    policy: str,
    trial: int,
    values: Iterable[float],
    args,
    *,
    report: bool = False,
) -> list[dict]:
    """Measure work at the first iterate attaining the target accuracy."""
    rows = []
    for theta in values:
        theta = float(theta)
        sigma = sigma_for(policy, theta)
        common = dict(
            problem=problem.kind,
            matrix=problem.condition,
            policy=policy,
            trial=trial,
            theta=theta,
            sigma=sigma,
            gamma_x=GAMMA,
            gamma_lambda=GAMMA,
            target_accuracy=args.target_accuracy,
            n=problem.dimension,
            m=problem.matrix.shape[0],
            solver_n=problem.matrix.shape[1],
            matrix_condition=problem.matrix_condition,
            inner_solver=INNER_SOLVER,
            random_seed=trial,
            max_outer_iterations=args.max_outer_iterations,
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
                theta=theta,
                linear_solver=INNER_SOLVER,
                objective=problem.objective,
                tolerance=args.target_accuracy / 100,
                max_iterations=args.max_outer_iterations,
                max_inner_iterations=(INNER_ITERATION_FACTOR * problem.matrix.shape[0]),
                random_seed=trial,
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
    if row["reached_accuracy"]:
        status = f"outer={row['outer_iterations']}, inner={row['inner_iterations']}"
    else:
        reason = row["error"] or f"best accuracy {row['achieved_accuracy']:.2e}"
        status = f"missed target ({reason})"
    print(
        f"{row['problem']} {row['matrix']} {row['policy']} "
        f"trial={row['trial']} theta={row['theta']:.4g} "
        f"sigma={row['sigma']:.4g}: {status}, "
        f"time={row['runtime_seconds']:.1f}s",
        flush=True,
    )


def key(row: dict) -> tuple:
    return (
        row["problem"],
        row["matrix"],
        row["policy"],
        int(row["trial"]),
        float(row["theta"]),
    )


def order(kinds: Iterable[str]):
    ranks = {
        "kind": {kind: index for index, kind in enumerate(kinds)},
        "condition": {name: index for index, name in enumerate(CONDITIONS)},
        "policy": {name: index for index, name in enumerate(POLICIES)},
    }

    def sort_key(row: dict) -> tuple:
        return (
            ranks["kind"].get(row["problem"], len(ranks["kind"])),
            ranks["condition"].get(row["matrix"], len(ranks["condition"])),
            ranks["policy"].get(row["policy"], len(ranks["policy"])),
            int(row["trial"]),
            float(row["theta"]),
        )

    return sort_key


def print_summary(rows: list[dict], kinds: Iterable[str]) -> None:
    for kind in kinds:
        for condition in CONDITIONS:
            for policy in POLICIES:
                selected = [
                    row
                    for row in rows
                    if row["problem"] == kind
                    and row["matrix"] == condition
                    and row["policy"] == policy
                    and row["reached_accuracy"]
                ]
                if not selected:
                    print(f"{kind:<14} {condition:<24} {policy:<9} target not reached")
                    continue
                best_outer = min(selected, key=lambda row: row["outer_iterations"])
                best_inner = min(selected, key=lambda row: row["inner_iterations"])
                print(
                    f"{kind:<14} {condition:<24} {policy:<9} "
                    f"outer best theta={best_outer['theta']:.4g} "
                    f"({best_outer['outer_iterations']:.0f})  "
                    f"inner best theta={best_inner['theta']:.4g} "
                    f"({best_inner['inner_iterations']:.0f})"
                )


def _checkpointed(path: Path, args) -> list[dict]:
    if args.force:
        return []
    return [
        row
        for row in read_rows(path)
        if row.get("m") == args.constraints
        and row.get("n") == args.variables
        and row.get("inner_solver") == INNER_SOLVER
        and int(row.get("trial", -1)) < args.trials
    ]


def run_study(args) -> None:
    """Run or resume the theta sweep."""
    output = study_dir(STUDY, args.results_dir)
    results_path = output / RESULTS
    rows = _checkpointed(results_path, args)
    completed = {key(row) for row in rows}
    sort_key = order(args.kinds)
    values = thetas(args)

    # Trial is the outermost loop so that an interrupted run still leaves
    # complete theta curves for every problem, just at fewer trials.
    for trial in range(args.trials):
        for kind in args.kinds:
            for condition in CONDITIONS:
                problem = None
                for policy in args.policies:
                    for theta in values:
                        identity = (kind, condition, policy, trial, float(theta))
                        if identity in completed:
                            continue
                        if problem is None:
                            problem = make_trial(kind, condition, trial, args)
                        rows = merge_rows(
                            rows,
                            run_problem(
                                problem, policy, trial, [theta], args, report=True
                            ),
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
    parser.add_argument("--theta-min", type=float, default=THETA_MIN)
    parser.add_argument("--theta-max", type=float, default=THETA_MAX)
    parser.add_argument("--theta-step", type=float, default=THETA_STEP)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--constraints", type=int, default=CONSTRAINTS)
    parser.add_argument("--variables", type=int, default=VARIABLES)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
        metavar="POLICY",
        help="inner-tolerance policies to run",
    )
    parser.add_argument(
        "--max-outer-iterations", type=int, default=MAX_OUTER_ITERATIONS
    )
    parser.add_argument("--target-accuracy", type=float, default=TARGET_ACCURACY)
    kind_argument(parser, KINDS)
    return parser


def main(argv: list[str] | None = None) -> None:
    run_study(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
