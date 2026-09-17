"""Sweep the relaxation parameter ``theta`` at a fixed unit metric.

The metric is pinned to ``gamma_x = gamma_lambda = 1`` so that the only free
outer parameter is the over/under-relaxation ``theta``, and every problem uses
the same ``100 x 500`` constraint matrix.  Equation (58) is solved with
randomized Kaczmarz on the Schur complement: one row per inner step, drawn
with probability proportional to its squared norm.

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

Randomized Kaczmarz contracts by roughly
``1 - sigma_min(S)^2 / ||S||_F^2`` per row projection, so its inner cost is
set by the Frobenius-scaled condition number of the Schur complement rather
than by ``kappa(S)``.  For these matrices that quantity is about 150 when
``kappa(A) = 10`` but about 6e4 when ``kappa(A) = 100``, and the inner budget
of ``500 * m`` projections is only enough for the former.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from randomized_sketch_descent import primal_dual_drs

from .experiment_utils import accuracy_history, plt, save_figure, write_rows
from .sweep_problems import (
    CONDITIONS,
    SweepProblem,
    make_large_equality_qp,
    make_large_inequality_qp,
    make_large_lp,
)


THETAS = np.round(np.linspace(0.1, 1.9, 13), 2)
GAMMA = 1.0
CONSTRAINTS = 100
VARIABLES = 500
TARGET_ACCURACY = 1e-7
INNER_SOLVER = "schur_randomized_kaczmarz"
RANDOM_SEED = 0
MAX_OUTER_ITERATIONS = 6_000
INNER_ITERATION_FACTOR = 500
FIXED_SIGMA = 0.04
FRONTIER_FRACTION = 0.9
FRONTIER_CAP = 0.45
POLICIES = ("fixed", "frontier")
KINDS = ("equality QP", "LP", "inequality QP")
MAKERS = {
    "equality QP": make_large_equality_qp,
    "LP": make_large_lp,
    "inequality QP": make_large_inequality_qp,
}


def sigma_for(policy: str, theta: float) -> float:
    """Return the inner tolerance this policy assigns to ``theta``."""
    if policy == "fixed":
        return FIXED_SIGMA
    return min(FRONTIER_CAP, FRONTIER_FRACTION * (2.0 - theta) / 2.0)


def run_problem(
    problem: SweepProblem,
    policy: str,
    thetas: Iterable[float] = THETAS,
    *,
    report: bool = False,
) -> list[dict]:
    """Measure work at the first iterate attaining the target accuracy."""
    rows = []
    for theta in thetas:
        theta = float(theta)
        sigma = sigma_for(policy, theta)
        common = dict(
            problem=problem.kind,
            matrix=problem.condition,
            policy=policy,
            theta=theta,
            sigma=sigma,
            gamma_x=GAMMA,
            gamma_lambda=GAMMA,
            target_accuracy=TARGET_ACCURACY,
            n=problem.dimension,
            m=problem.matrix.shape[0],
            solver_n=problem.matrix.shape[1],
            matrix_condition=problem.matrix_condition,
            inner_solver=INNER_SOLVER,
            random_seed=RANDOM_SEED,
            max_outer_iterations=MAX_OUTER_ITERATIONS,
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
                tolerance=TARGET_ACCURACY / 100,
                max_iterations=MAX_OUTER_ITERATIONS,
                max_inner_iterations=(
                    INNER_ITERATION_FACTOR * problem.matrix.shape[0]
                ),
                random_seed=RANDOM_SEED,
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
                _print_progress(rows[-1])
            continue

        accuracy = accuracy_history(problem.optimum, result)
        reached = np.flatnonzero(accuracy <= TARGET_ACCURACY)
        if reached.size:
            index = int(reached[0])
            outer_iterations = index + 1
            inner_iterations = int(result.inner_iterations[: index + 1].sum())
            achieved_accuracy = float(accuracy[index])
        else:
            outer_iterations = inner_iterations = np.nan
            achieved_accuracy = float(accuracy[-1])

        rows.append(
            dict(
                **common,
                reached_accuracy=bool(reached.size),
                outer_iterations=outer_iterations,
                inner_iterations=inner_iterations,
                achieved_accuracy=achieved_accuracy,
                solver_iterations=result.iterations,
                runtime_seconds=time.perf_counter() - started,
                error="",
            )
        )
        if report:
            _print_progress(rows[-1])
    return rows


def _print_progress(row: dict) -> None:
    if row["reached_accuracy"]:
        status = f"outer={row['outer_iterations']}, inner={row['inner_iterations']}"
    else:
        reason = row["error"] or f"best accuracy {row['achieved_accuracy']:.2e}"
        status = f"missed target ({reason})"
    print(
        f"{row['problem']} {row['matrix']} {row['policy']} "
        f"theta={row['theta']:.4g} sigma={row['sigma']:.4g}: "
        f"{status}, time={row['runtime_seconds']:.1f}s",
        flush=True,
    )


NUMERIC_FIELDS = (
    "theta",
    "sigma",
    "gamma_x",
    "gamma_lambda",
    "target_accuracy",
    "matrix_condition",
    "outer_iterations",
    "inner_iterations",
    "achieved_accuracy",
    "solver_iterations",
    "runtime_seconds",
)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name in NUMERIC_FIELDS:
            row[name] = float(row[name]) if row.get(name) else np.nan
        for name in ("n", "m", "solver_n", "max_outer_iterations"):
            if row.get(name):
                row[name] = int(float(row[name]))
        row["reached_accuracy"] = row["reached_accuracy"].lower() == "true"
    return rows


def _key(row: dict) -> tuple[str, str, str, float]:
    return (row["problem"], row["matrix"], row["policy"], float(row["theta"]))


def _merge_rows(rows: list[dict], replacements: list[dict]) -> list[dict]:
    replaced = {_key(row) for row in replacements}
    merged = [row for row in rows if _key(row) not in replaced]
    merged.extend(replacements)
    order = {
        "kind": {kind: index for index, kind in enumerate(KINDS)},
        "condition": {name: index for index, name in enumerate(CONDITIONS)},
        "policy": {name: index for index, name in enumerate(POLICIES)},
    }
    return sorted(
        merged,
        key=lambda row: (
            order["kind"][row["problem"]],
            order["condition"][row["matrix"]],
            order["policy"][row["policy"]],
            float(row["theta"]),
        ),
    )


def _series(rows: list[dict], kind: str, condition: str, policy: str):
    selected = [
        row
        for row in rows
        if row["problem"] == kind
        and row["matrix"] == condition
        and row["policy"] == policy
    ]
    selected.sort(key=lambda row: float(row["theta"]))
    return selected


def plot_work(rows: list[dict], output: Path) -> None:
    """Plot outer work, inner projections, and runtime against ``theta``."""
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.5), sharex=True)
    styles = {"fixed": "-", "frontier": "--"}
    colors = {
        condition: f"C{index}" for index, condition in enumerate(CONDITIONS)
    }
    for column, kind in enumerate(KINDS):
        failures = {row: [] for row in range(3)}
        for condition in CONDITIONS:
            for policy in POLICIES:
                selected = _series(rows, kind, condition, policy)
                if not selected:
                    continue
                theta = np.array(
                    [row["theta"] for row in selected], dtype=float
                )
                quantities = (
                    np.array(
                        [row["outer_iterations"] for row in selected],
                        dtype=float,
                    ),
                    np.array(
                        [row["inner_iterations"] for row in selected],
                        dtype=float,
                    ),
                    np.array(
                        [row["runtime_seconds"] for row in selected],
                        dtype=float,
                    ),
                )
                condition_number = float(selected[0]["matrix_condition"])
                label = rf"$\kappa(A)={condition_number:g}$, {policy} $\sigma$"
                missed = np.array(
                    [not row["reached_accuracy"] for row in selected]
                )
                for index, values in enumerate(quantities):
                    axes[index, column].semilogy(
                        theta,
                        values,
                        styles[policy],
                        marker="o",
                        markersize=4,
                        color=colors[condition],
                        label=label,
                    )
                    failures[index].append((theta[missed], colors[condition]))

        # Runs that never reach the target leave a gap in the curve; mark the
        # offending theta on the frame so the gap is not read as missing data.
        for index in range(3):
            axis = axes[index, column]
            finite = [
                value
                for line in axis.get_lines()
                for value in line.get_ydata()
                if np.isfinite(value)
            ]
            if not finite:
                continue
            top = max(finite)
            for theta, color in failures[index]:
                if theta.size:
                    axis.scatter(
                        theta,
                        np.full(theta.shape, top),
                        marker="x",
                        s=55,
                        color=color,
                        zorder=3,
                    )
        axes[0, column].set_title(
            f"{kind}\n" + rf"$A$: ${CONSTRAINTS}\times{VARIABLES}$"
        )
        axes[2, column].set_xlabel(r"relaxation $\theta$")
        for index in range(3):
            axes[index, column].grid(True, which="both", alpha=0.3)
            axes[index, column].legend(fontsize=7)

    axes[0, 0].set_ylabel("outer iterations to target")
    axes[1, 0].set_ylabel("row projections to target")
    axes[2, 0].set_ylabel("solver time (seconds)")
    fig.suptitle(
        "Target: normalized objective error + feasibility "
        rf"$\leq 10^{{-7}}$; $\gamma_x=\gamma_\lambda={GAMMA:g}$; "
        "randomized Kaczmarz inner solver, "
        rf"seed {RANDOM_SEED}; $\times$ marks $\theta$ that missed the "
        rf"target within {MAX_OUTER_ITERATIONS} outer iterations"
    )
    save_figure(fig, output)


def print_summary(rows: list[dict]) -> None:
    for kind in KINDS:
        for condition in CONDITIONS:
            for policy in POLICIES:
                reached = [
                    row
                    for row in _series(rows, kind, condition, policy)
                    if row["reached_accuracy"]
                ]
                if not reached:
                    print(
                        f"{kind:<14} {condition:<24} {policy:<9} "
                        "target not reached"
                    )
                    continue
                best_outer = min(
                    reached, key=lambda row: row["outer_iterations"]
                )
                best_inner = min(
                    reached, key=lambda row: row["inner_iterations"]
                )
                print(
                    f"{kind:<14} {condition:<24} {policy:<9} "
                    f"outer best theta={best_outer['theta']:.4g} "
                    f"({best_outer['outer_iterations']:.0f})  "
                    f"inner best theta={best_inner['theta']:.4g} "
                    f"({best_inner['inner_iterations']:.0f})"
                )


def run_study() -> None:
    """Run or resume the theta sweep."""
    output = Path("figures/theta_sweep")
    results_path = output / "results.csv"
    rows = [
        row
        for row in _read_rows(results_path)
        if row["m"] == CONSTRAINTS
        and row["n"] == VARIABLES
        and row["inner_solver"] == INNER_SOLVER
    ]
    completed = {_key(row) for row in rows}
    for kind in KINDS:
        for condition in CONDITIONS:
            problem = MAKERS[kind](condition, CONSTRAINTS, VARIABLES)
            for policy in POLICIES:
                for theta in THETAS:
                    key = (kind, condition, policy, float(theta))
                    if key in completed:
                        print(f"checkpointed {key}", flush=True)
                        continue
                    rows = _merge_rows(
                        rows,
                        run_problem(problem, policy, [theta], report=True),
                    )
                    write_rows(results_path, rows)
                    completed.add(key)
    plot_work(rows, output / "iteration_counts.pdf")
    print_summary(rows)
    print(f"wrote {output}")
