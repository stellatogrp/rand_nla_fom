"""Sweep the relaxation parameter ``theta`` at a fixed unit metric.

The metric is pinned to ``gamma_x = gamma_lambda = 1`` so that the only free
outer parameter is the over/under-relaxation ``theta``, and every problem uses
a ``50 x 200`` constraint matrix.  Equation (58) is solved with randomized
Kaczmarz on the Schur complement: one row per inner step, drawn with
probability proportional to its squared norm.

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

Each setting is repeated over ``TRIALS`` independent trials (three).  A trial redraws
the problem instance -- singular vectors and, through
``randomize_spectrum``, the interior singular values -- along with the
Kaczmarz row stream, so the spread covers problem-to-problem variation as
well as solver randomness.  The extreme singular values stay pinned, so every
trial has exactly the stated ``kappa(A)``.  Curves show the median over trials
and whiskers span the observed range.

Randomized Kaczmarz contracts by roughly
``1 - sigma_min(S)^2 / ||S||_F^2`` per row projection, so its inner cost is
set by the Frobenius-scaled condition number of the Schur complement rather
than by ``kappa(S)``.  For these matrices that quantity is about 76 at
``kappa(A) = 10`` and 190 at ``kappa(A) = 20``, both comfortably inside the
inner budget of ``500 * m`` projections.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from randomized_sketch_descent import primal_dual_drs

from .experiment_utils import accuracy_history, plt, save_figure, write_rows
from .sweep_problems import (
    Condition,
    SweepProblem,
    make_large_equality_qp,
    make_large_inequality_qp,
    make_large_lp,
)


THETAS = np.round(np.arange(0.1, 1.9 + 1e-9, 0.1), 2)
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
QUANTITIES = (
    ("outer_iterations", "outer iterations to target"),
    ("inner_iterations", "row projections to target"),
    ("runtime_seconds", "solver time (seconds)"),
)


def sigma_for(policy: str, theta: float) -> float:
    """Return the inner tolerance this policy assigns to ``theta``."""
    if policy == "fixed":
        return FIXED_SIGMA
    return min(FRONTIER_CAP, FRONTIER_FRACTION * (2.0 - theta) / 2.0)


def make_trial(kind: str, condition: Condition, trial: int) -> SweepProblem:
    """Build the problem instance used by one trial."""
    maker, base_seed = MAKERS[kind]
    return maker(
        condition,
        CONSTRAINTS,
        VARIABLES,
        seed=base_seed + SEED_STRIDE * trial,
        randomize_spectrum=True,
    )


def run_problem(
    problem: SweepProblem,
    policy: str,
    trial: int,
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
            trial=trial,
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
            random_seed=trial,
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
        status = (
            f"outer={row['outer_iterations']}, inner={row['inner_iterations']}"
        )
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
        for name in ("n", "m", "solver_n", "max_outer_iterations", "trial"):
            if row.get(name):
                row[name] = int(float(row[name]))
        row["reached_accuracy"] = row["reached_accuracy"].lower() == "true"
    return rows


def _key(row: dict) -> tuple[str, str, str, int, float]:
    return (
        row["problem"],
        row["matrix"],
        row["policy"],
        int(row["trial"]),
        float(row["theta"]),
    )


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
            int(row["trial"]),
            float(row["theta"]),
        ),
    )


def _series(rows: list[dict], kind: str, condition: str, policy: str):
    return [
        row
        for row in rows
        if row["problem"] == kind
        and row["matrix"] == condition
        and row["policy"] == policy
    ]


def _aggregate(selected: list[dict]) -> dict:
    """Reduce per-trial rows to a median and an observed range per theta.

    Only trials that reached the target contribute: a run that never got
    there has no cost to report, and its wall time measures a failure rather
    than a solve.
    """
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[float(row["theta"])].append(row)

    thetas = np.array(sorted(grouped), dtype=float)
    summary = {
        name: {"median": [], "low": [], "high": []} for name, _ in QUANTITIES
    }
    incomplete = []
    for theta in thetas:
        trials = grouped[theta]
        reached = [row for row in trials if row["reached_accuracy"]]
        incomplete.append(len(reached) < len(trials))
        for name, _ in QUANTITIES:
            values = np.array(
                [row[name] for row in reached], dtype=float
            )
            values = values[np.isfinite(values)]
            for statistic, function in (
                ("median", np.median),
                ("low", np.min),
                ("high", np.max),
            ):
                summary[name][statistic].append(
                    float(function(values)) if values.size else np.nan
                )
    return {
        "theta": thetas,
        "incomplete": np.array(incomplete, dtype=bool),
        "trials": max((len(group) for group in grouped.values()), default=0),
        **{
            name: {key: np.array(value, dtype=float)
                   for key, value in statistics.items()}
            for name, statistics in summary.items()
        },
    }


def plot_work(rows: list[dict], output: Path) -> None:
    """Plot outer work, inner projections, and runtime against ``theta``."""
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.5), sharex=True)
    styles = {"fixed": "-", "frontier": "--"}
    colors = {
        condition: f"C{index}" for index, condition in enumerate(CONDITIONS)
    }
    trials = 0
    for column, kind in enumerate(KINDS):
        # NaN entries for thetas that never reached the target confuse the log
        # autoscale, so track the finite extent and set the limits explicitly.
        extents: dict[int, list[float]] = defaultdict(list)
        for condition in CONDITIONS:
            for policy in POLICIES:
                selected = _series(rows, kind, condition, policy)
                if not selected:
                    continue
                summary = _aggregate(selected)
                trials = max(trials, summary["trials"])
                condition_number = float(selected[0]["matrix_condition"])
                label = (
                    rf"$\kappa(A)={condition_number:g}$, {policy} $\sigma$"
                )
                for index, (name, _) in enumerate(QUANTITIES):
                    axis = axes[index, column]
                    median = summary[name]["median"]
                    lower = np.clip(median - summary[name]["low"], 0, None)
                    upper = np.clip(summary[name]["high"] - median, 0, None)
                    axis.errorbar(
                        summary["theta"],
                        median,
                        yerr=np.vstack((lower, upper)),
                        fmt=styles[policy],
                        marker="o",
                        markersize=3.5,
                        linewidth=1.4,
                        elinewidth=1.0,
                        capsize=2.5,
                        color=colors[condition],
                        label=label,
                    )
                    axis.set_yscale("log")
                    finite = np.concatenate(
                        (
                            summary[name]["low"][
                                np.isfinite(summary[name]["low"])
                            ],
                            summary[name]["high"][
                                np.isfinite(summary[name]["high"])
                            ],
                        )
                    )
                    if finite.size:
                        extents[index].extend((finite.min(), finite.max()))
                    # A theta where some but not all trials reached the
                    # target still plots a median; flag it so the thinner
                    # sample is visible.
                    partial = summary["incomplete"] & np.isfinite(median)
                    if partial.any():
                        axis.scatter(
                            summary["theta"][partial],
                            median[partial],
                            marker="x",
                            s=45,
                            color=colors[condition],
                            zorder=4,
                        )

        axes[0, column].set_title(
            f"{kind}\n" + rf"$A$: ${CONSTRAINTS}\times{VARIABLES}$"
        )
        axes[2, column].set_xlabel(r"relaxation $\theta$")
        for index in range(3):
            axes[index, column].grid(True, which="both", alpha=0.3)
            if extents[index]:
                axes[index, column].set_ylim(
                    min(extents[index]) / 1.6, max(extents[index]) * 1.6
                )

    for index, (_, ylabel) in enumerate(QUANTITIES):
        axes[index, 0].set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    fig.legend(
        seen.values(),
        seen.keys(),
        loc="lower center",
        ncol=len(seen),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Target: normalized objective error + feasibility "
        rf"$\leq 10^{{-7}}$; $\gamma_x=\gamma_\lambda={GAMMA:g}$; "
        "randomized Kaczmarz inner solver; "
        rf"median of {trials} trials, whiskers span the observed range; "
        rf"$\times$ marks $\theta$ where some trial missed the target "
        rf"within {MAX_OUTER_ITERATIONS} outer iterations"
    )
    save_figure(fig, output)


def print_summary(rows: list[dict]) -> None:
    for kind in KINDS:
        for condition in CONDITIONS:
            for policy in POLICIES:
                selected = _series(rows, kind, condition, policy)
                if not selected:
                    continue
                summary = _aggregate(selected)
                outer = summary["outer_iterations"]["median"]
                inner = summary["inner_iterations"]["median"]
                if not np.any(np.isfinite(outer)):
                    print(
                        f"{kind:<14} {condition:<24} {policy:<9} "
                        "target not reached"
                    )
                    continue
                best_outer = int(np.nanargmin(outer))
                best_inner = int(np.nanargmin(inner))
                print(
                    f"{kind:<14} {condition:<24} {policy:<9} "
                    f"outer best theta={summary['theta'][best_outer]:.4g} "
                    f"({outer[best_outer]:.0f} "
                    f"[{summary['outer_iterations']['low'][best_outer]:.0f}, "
                    f"{summary['outer_iterations']['high'][best_outer]:.0f}])  "
                    f"inner best theta={summary['theta'][best_inner]:.4g} "
                    f"({inner[best_inner]:.0f} "
                    f"[{summary['inner_iterations']['low'][best_inner]:.0f}, "
                    f"{summary['inner_iterations']['high'][best_inner]:.0f}])"
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
        and int(row["trial"]) < TRIALS
    ]
    completed = {_key(row) for row in rows}
    # Trial is the outermost loop so that an interrupted run still leaves
    # complete theta curves for every problem, just at fewer trials.
    for trial in range(TRIALS):
        for kind in KINDS:
            for condition in CONDITIONS:
                problem = None
                for policy in POLICIES:
                    for theta in THETAS:
                        key = (kind, condition, policy, trial, float(theta))
                        if key in completed:
                            continue
                        if problem is None:
                            problem = make_trial(kind, condition, trial)
                        rows = _merge_rows(
                            rows,
                            run_problem(
                                problem, policy, trial, [theta], report=True
                            ),
                        )
                        write_rows(results_path, rows)
                        completed.add(key)
    plot_work(rows, output / "iteration_counts.pdf")
    print_summary(rows)
    print(f"wrote {output}")
