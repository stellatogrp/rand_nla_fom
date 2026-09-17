"""Sweep the relative inner-solve tolerance with fixed relaxation."""

from __future__ import annotations

import csv
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from randomized_sketch_descent import DRSResult, primal_dual_drs

from .experiment_utils import (
    accuracy_history,
    plt,
    save_figure,
    write_rows,
)
from .sweep_problems import (
    CONDITIONS,
    SweepProblem,
    make_large_equality_qp,
    make_large_inequality_qp,
    make_large_lp,
)


SIGMAS = np.linspace(0.0, 0.49, 21)
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


def _block_size(problem: SweepProblem) -> int:
    """Use about five percent of the Schur-complement dimension."""
    return max(1, round(BLOCK_FRACTION * problem.matrix.shape[0]))


def run_problem(
    problem: SweepProblem,
    sigmas: Iterable[float] = SIGMAS,
    *,
    report: bool = False,
) -> list[dict]:
    """Measure work at the first iterate attaining the target accuracy."""
    rows = []
    block_size = _block_size(problem)
    for sigma in sigmas:
        sigma = float(sigma)
        common = dict(
            problem=problem.kind,
            matrix=problem.condition,
            sigma=sigma,
            theta=THETA,
            gamma_x=GAMMA,
            gamma_lambda=GAMMA,
            target_accuracy=TARGET_ACCURACY,
            n=problem.dimension,
            m=problem.matrix.shape[0],
            solver_n=problem.matrix.shape[1],
            matrix_condition=problem.matrix_condition,
            inner_solver=INNER_SOLVER,
            block_size=block_size,
            block_fraction=BLOCK_FRACTION,
            random_seed=RANDOM_SEED,
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
                tolerance=TARGET_ACCURACY / 100,
                max_iterations=problem.max_iterations,
                max_inner_iterations=(
                    INNER_ITERATION_FACTOR * problem.matrix.shape[0]
                ),
                block_size=block_size,
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
            row = rows[-1]
            if report:
                _print_progress(row)
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

        row = dict(
            **common,
            reached_accuracy=bool(reached.size),
            outer_iterations=outer_iterations,
            inner_iterations=inner_iterations,
            achieved_accuracy=achieved_accuracy,
            solver_iterations=result.iterations,
            runtime_seconds=time.perf_counter() - started,
            error="",
        )
        rows.append(row)
        if report:
            _print_progress(row)
    return rows


def _print_progress(row: dict) -> None:
    status = (
        f"outer={row['outer_iterations']}, inner={row['inner_iterations']}"
        if row["reached_accuracy"]
        else f"failed: {row['error'] or 'target not reached'}"
    )
    elapsed = row["runtime_seconds"]
    print(
        f"{row['problem']} {row['matrix']} sigma={row['sigma']:.4g}: "
        f"{status}, time={elapsed:.1f}s",
        flush=True,
    )


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name in (
            "sigma",
            "theta",
            "gamma_x",
            "gamma_lambda",
            "target_accuracy",
            "matrix_condition",
            "outer_iterations",
            "inner_iterations",
            "achieved_accuracy",
            "solver_iterations",
            "runtime_seconds",
        ):
            row[name] = float(row[name]) if row.get(name) else np.nan
        for name in ("n", "m", "solver_n"):
            if not row.get(name):
                continue
            row[name] = int(row[name])
        row["reached_accuracy"] = row["reached_accuracy"].lower() == "true"
        row.setdefault("inner_solver", "schur_cg")
        row.setdefault("block_size", "")
        row.setdefault("block_fraction", "")
        row.setdefault("random_seed", "")
    return rows


def _merge_rows(rows: list[dict], replacements: list[dict]) -> list[dict]:
    keys = {(row["problem"], row["matrix"], row["sigma"]) for row in replacements}
    merged = [
        row
        for row in rows
        if (row["problem"], row["matrix"], row["sigma"]) not in keys
    ]
    merged.extend(replacements)
    kind_order = {kind: index for index, kind in enumerate(KINDS)}
    condition_order = {
        condition: index for index, condition in enumerate(CONDITIONS)
    }
    return sorted(
        merged,
        key=lambda row: (
            kind_order[row["problem"]],
            condition_order[row["matrix"]],
            float(row["sigma"]),
        ),
    )


def _matrix_label(kind: str, rows: list[dict]) -> str:
    """Describe the common problem and solver matrix dimensions."""
    dimensions = {
        (
            int(float(row["m"])),
            int(float(row["n"])),
            int(float(row.get("solver_n") or row["n"])),
        )
        for row in rows
        if row["problem"] == kind
    }
    if len(dimensions) != 1:
        return "mixed matrix dimensions"
    constraints, variables, solver_variables = dimensions.pop()
    if solver_variables != variables:
        return (
            rf"$A$: ${constraints}\times{variables}$; "
            rf"$[A\ I]$: ${constraints}\times{solver_variables}$"
        )
    return rf"$A$: ${constraints}\times{variables}$"


def plot_work(rows: list[dict], output: Path) -> None:
    """Plot outer and cumulative inner work against ``sigma``."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5), sharex=True)
    for column, kind in enumerate(KINDS):
        for condition in CONDITIONS:
            selected = [
                row
                for row in rows
                if row["problem"] == kind and row["matrix"] == condition
            ]
            sigmas = np.array([row["sigma"] for row in selected], dtype=float)
            outer = np.array(
                [row["outer_iterations"] for row in selected], dtype=float
            )
            inner = np.array(
                [row["inner_iterations"] for row in selected], dtype=float
            )
            runtime = np.array(
                [row.get("runtime_seconds", np.nan) for row in selected],
                dtype=float,
            )
            condition_number = float(selected[0]["matrix_condition"])
            label = rf"$\kappa(A)={condition_number:g}$"
            axes[0, column].semilogy(sigmas, outer, "o-", label=label)
            axes[1, column].semilogy(sigmas, inner, "o-", label=label)
            axes[2, column].semilogy(sigmas, runtime, "o-", label=label)

        solvers = {
            row.get("inner_solver", "schur_cg")
            for row in rows
            if row["problem"] == kind
        }
        solver_names = {
            "schur_cg": "CG",
            "schur_block_kaczmarz": "block Kaczmarz",
        }
        labels = ", ".join(
            sorted(solver_names.get(name, name) for name in solvers)
        )
        dimensions = _matrix_label(kind, rows)
        axes[0, column].set_title(f"{kind}\n{labels}; {dimensions}")
        axes[2, column].set_xlabel(r"inner tolerance $\sigma$")
        for row in range(3):
            axes[row, column].grid(True, which="both", alpha=0.3)
            axes[row, column].legend(fontsize=8)

    axes[0, 0].set_ylabel("outer iterations to target")
    axes[1, 0].set_ylabel("block projections to target")
    axes[2, 0].set_ylabel("solver time (seconds)")
    block_sizes = sorted({int(row["block_size"]) for row in rows})
    block_label = ", ".join(map(str, block_sizes))
    fig.suptitle(
        "Target: normalized objective error + feasibility "
        rf"$\leq 10^{{-7}}$; $\theta={THETA:g}$, "
        rf"$\gamma_x=\gamma_\lambda={GAMMA:g}$; "
        rf"block size {block_label} ({100 * BLOCK_FRACTION:g}% of rows), "
        rf"seed {RANDOM_SEED}"
    )
    save_figure(fig, output)


def print_summary(rows: list[dict]) -> None:
    for kind in KINDS:
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


def run_study() -> None:
    """Run or resume the large block-Kaczmarz sweep."""
    output = Path("figures/sigma_sweep")
    results_path = output / "results.csv"
    rows = [
        row
        for row in _read_rows(results_path)
        if row["m"] == CONSTRAINTS
        and row["n"] == VARIABLES
        and row["inner_solver"] == INNER_SOLVER
        and int(row["block_size"]) == round(BLOCK_FRACTION * CONSTRAINTS)
    ]
    makers = (
        make_large_equality_qp,
        make_large_lp,
        make_large_inequality_qp,
    )
    completed = {
        (row["problem"], row["matrix"], float(row["sigma"])) for row in rows
    }
    for maker in makers:
        for condition in CONDITIONS:
            problem = maker(condition, CONSTRAINTS, VARIABLES)
            for sigma in SIGMAS:
                key = (problem.kind, condition, float(sigma))
                if key in completed:
                    print(f"checkpointed {key}", flush=True)
                    continue
                rows = _merge_rows(
                    rows,
                    run_problem(problem, [sigma], report=True),
                )
                write_rows(results_path, rows)
                completed.add(key)
    plot_work(rows, output / "iteration_counts.pdf")
    print_summary(rows)
    print(f"wrote {output}")


def run_lp_study() -> None:
    """Run only the LP sweep and merge checkpoints into the shared outputs."""
    output = Path("figures/sigma_sweep")
    results_path = output / "results.csv"
    rows = _read_rows(results_path)
    for condition in CONDITIONS:
        problem = make_large_lp(condition, CONSTRAINTS, VARIABLES)
        for sigma in SIGMAS:
            replacement = run_problem(problem, [sigma], report=True)
            rows = _merge_rows(rows, replacement)
            write_rows(results_path, rows)
    plot_work(rows, output / "iteration_counts.pdf")
    print_summary(rows)
    print(f"wrote {output}")
