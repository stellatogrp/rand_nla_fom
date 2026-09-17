"""Compare inner linear solvers over many random problem instances.

The preconditioner scales, the relaxation, and the inner tolerance are held
fixed at ``gamma_x = gamma_lambda = gamma``, ``theta``, and ``sigma``; the only
thing that varies within one problem family is the solver used for equation
(58).  For every family the study draws ``INSTANCES`` independent problems of
the same size and records, for each solver, the outer and cumulative inner
iterations needed to reach a fixed accuracy, so the spread over instances can
be shown with error bars.
"""

from __future__ import annotations

import csv
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
SOLVER_LABELS = {
    "schur_cg": "Schur CG",
    "schur_block_kaczmarz": "block Kaczmarz",
    "schur_randomized_kaczmarz": "randomized Kaczmarz",
    "coupled_gmres": "coupled GMRES",
}
# Fixed categorical slots, assigned by solver and never cycled.
SOLVER_COLORS = {
    "schur_cg": "#2a78d6",
    "schur_block_kaczmarz": "#eb6834",
    "schur_randomized_kaczmarz": "#1baf7a",
    "coupled_gmres": "#eda100",
}
CONDITION_LABELS = {
    "moderately-conditioned": r"$\kappa(A)=10$",
    "ill-conditioned": r"$\kappa(A)=100$",
}
FLOAT_FIELDS = (
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
)
KEY_FIELDS = ("problem", "matrix", "instance_seed", "inner_solver")


def _block_size(problem: SweepProblem) -> int:
    """Use about five percent of the Schur-complement dimension."""
    return max(1, round(BLOCK_FRACTION * problem.matrix.shape[0]))


def run_instance(problem: SweepProblem, seed: int, solver: str) -> dict:
    """Measure the work one solver needs on one problem instance."""
    block_size = _block_size(problem)
    common = dict(
        problem=problem.kind,
        matrix=problem.condition,
        inner_solver=solver,
        instance_seed=seed,
        sigma=SIGMA,
        theta=THETA,
        gamma_x=GAMMA,
        gamma_lambda=GAMMA,
        target_accuracy=TARGET_ACCURACY,
        n=problem.dimension,
        m=problem.matrix.shape[0],
        solver_n=problem.matrix.shape[1],
        matrix_condition=problem.matrix_condition,
        block_size=block_size,
        block_fraction=BLOCK_FRACTION,
        random_seed=KACZMARZ_SEED,
    )
    started = time.perf_counter()
    try:
        result = primal_dual_drs(
            problem.prox,
            problem.matrix,
            problem.rhs,
            gamma_x=GAMMA,
            gamma_lambda=GAMMA,
            sigma=SIGMA,
            theta=THETA,
            linear_solver=solver,
            objective=problem.objective,
            tolerance=TARGET_ACCURACY / 100,
            max_iterations=MAX_OUTER_ITERATIONS,
            max_inner_iterations=INNER_ITERATION_FACTOR * problem.matrix.shape[0],
            block_size=block_size,
            random_seed=KACZMARZ_SEED,
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
    reached = np.flatnonzero(accuracy <= TARGET_ACCURACY)
    if reached.size:
        index = int(reached[0])
        outer_iterations = float(index + 1)
        inner_iterations = float(result.inner_iterations[: index + 1].sum())
        achieved_accuracy = float(accuracy[index])
    else:
        outer_iterations = inner_iterations = np.nan
        achieved_accuracy = float(accuracy[-1])

    return dict(
        **common,
        reached_accuracy=bool(reached.size),
        outer_iterations=outer_iterations,
        inner_iterations=inner_iterations,
        achieved_accuracy=achieved_accuracy,
        solver_iterations=float(result.iterations),
        runtime_seconds=time.perf_counter() - started,
        error="" if reached.size else "target accuracy not reached",
    )


def _run_task(task: tuple[str, str, int, tuple[str, ...], bool]) -> list[dict]:
    """Run the requested solvers on a single freshly drawn instance."""
    kind, condition, seed, solvers, randomize_spectrum = task
    problem = MAKERS[kind](
        condition,
        CONSTRAINTS,
        VARIABLES,
        seed=seed,
        randomize_spectrum=randomize_spectrum,
    )
    rows = [run_instance(problem, seed, solver) for solver in solvers]
    for row in rows:
        row["randomize_spectrum"] = randomize_spectrum
    return rows


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name in FLOAT_FIELDS:
            row[name] = float(row[name]) if row.get(name) else np.nan
        for name in ("n", "m", "solver_n", "instance_seed", "block_size"):
            if row.get(name):
                row[name] = int(float(row[name]))
        row["reached_accuracy"] = str(row["reached_accuracy"]).lower() == "true"
        row["randomize_spectrum"] = (
            str(row.get("randomize_spectrum", "")).lower() == "true"
        )
    return rows


def _key(row: dict) -> tuple:
    return (
        row["problem"],
        row["matrix"],
        int(row["instance_seed"]),
        row["inner_solver"],
    )


def _sorted_rows(rows: list[dict]) -> list[dict]:
    kind_order = {kind: index for index, kind in enumerate(KINDS)}
    condition_order = {name: index for index, name in enumerate(CONDITIONS)}
    solver_order = {name: index for index, name in enumerate(SOLVERS)}
    return sorted(
        rows,
        key=lambda row: (
            kind_order[row["problem"]],
            condition_order[row["matrix"]],
            solver_order[row["inner_solver"]],
            int(row["instance_seed"]),
        ),
    )


def _select(rows: list[dict], kind: str, condition: str, solver: str) -> list[dict]:
    return [
        row
        for row in rows
        if row["problem"] == kind
        and row["matrix"] == condition
        and row["inner_solver"] == solver
    ]


def _statistics(values: np.ndarray) -> dict:
    """Return mean, spread, and order statistics of a sample."""
    if values.size == 0:
        return dict(
            count=0, mean=np.nan, sd=np.nan, sem=np.nan,
            median=np.nan, low=np.nan, high=np.nan,
        )
    return dict(
        count=int(values.size),
        mean=float(values.mean()),
        sd=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        sem=(
            float(values.std(ddof=1) / np.sqrt(values.size))
            if values.size > 1
            else 0.0
        ),
        median=float(np.median(values)),
        low=float(values.min()),
        high=float(values.max()),
    )


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate per-instance results into one row per solver and family."""
    summary = []
    for kind in KINDS:
        for condition in CONDITIONS:
            for solver in SOLVERS:
                selected = _select(rows, kind, condition, solver)
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
                for name in ("outer_iterations", "inner_iterations", "runtime_seconds"):
                    values = np.array(
                        [row[name] for row in reached], dtype=float
                    )
                    stats = _statistics(values[np.isfinite(values)])
                    prefix = {
                        "outer_iterations": "outer",
                        "inner_iterations": "inner",
                        "runtime_seconds": "time",
                    }[name]
                    for statistic, value in stats.items():
                        if statistic == "count":
                            continue
                        entry[f"{prefix}_{statistic}"] = value
                summary.append(entry)
    return summary


def _spectrum_label(rows: list[dict]) -> str:
    """Describe how the singular values were drawn across instances."""
    flags = {bool(row.get("randomize_spectrum")) for row in rows}
    if flags == {True}:
        return "randomized interior spectrum"
    if flags == {False}:
        return "fixed geometric spectrum"
    return "mixed spectrum sampling"


def _bar_panel(axis, rows: list[dict], kind: str, field: str) -> None:
    """Draw one grouped bar panel with mean +/- one standard deviation."""
    width = 0.2
    centers = np.arange(len(CONDITIONS))
    drawn_means: list[float] = []
    failures: list[tuple[float, str]] = []
    for offset, solver in enumerate(SOLVERS):
        means, errors, notes = [], [], []
        for condition in CONDITIONS:
            selected = [
                row
                for row in _select(rows, kind, condition, solver)
                if row["reached_accuracy"]
            ]
            values = np.array([row[field] for row in selected], dtype=float)
            values = values[np.isfinite(values)]
            stats = _statistics(values)
            means.append(stats["mean"])
            errors.append(0.0 if np.isnan(stats["sd"]) else stats["sd"])
            total = len(_select(rows, kind, condition, solver))
            notes.append(
                "" if stats["count"] == total else f"{stats['count']}/{total}"
            )
        positions = centers + (offset - 1.5) * width
        means = np.array(means, dtype=float)
        errors = np.array(errors, dtype=float)
        # Clip the lower whisker so a wide spread stays drawable on a log axis.
        lower = np.minimum(errors, np.nan_to_num(means) * 0.95)
        axis.bar(
            positions,
            np.nan_to_num(means),
            width * 0.9,
            yerr=np.vstack((lower, errors)),
            color=SOLVER_COLORS[solver],
            label=SOLVER_LABELS[solver],
            error_kw=dict(elinewidth=1.0, capsize=2.5, ecolor="#3d3d3a"),
        )
        for position, mean, note in zip(positions, means, notes):
            if np.isnan(mean):
                failures.append((position, SOLVER_COLORS[solver]))
            else:
                drawn_means.append(mean)
                if note:
                    axis.text(
                        position, mean, f"{note} ", rotation=90,
                        fontsize=6, ha="center", va="bottom",
                    )
    axis.set_xticks(centers)
    axis.set_xticklabels([CONDITION_LABELS[name] for name in CONDITIONS])
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.set_axisbelow(True)

    positive = [value for value in drawn_means if value > 0]
    if axis.get_yscale() == "log" and positive:
        # Failed solvers plot as a zero-height bar; keep the axis on the
        # decades that carry data instead of running down to the bar base.
        axis.set_ylim(bottom=min(positive) / 5)
    low, high = axis.get_ylim()
    middle = np.sqrt(low * high) if axis.get_yscale() == "log" else (low + high) / 2
    for position, color in failures:
        axis.text(
            position, middle, "no instance\nreached target",
            rotation=90, fontsize=5.5, ha="center", va="center", color=color,
            fontweight="bold",
        )


def plot_work(rows: list[dict], output: Path) -> None:
    """Plot outer iterations, inner iterations, and time for every family."""
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 10))
    fields = (
        ("outer_iterations", "outer iterations to target", False),
        ("inner_iterations", "inner iterations to target", True),
        ("runtime_seconds", "solver time (seconds)", True),
    )
    for column, kind in enumerate(KINDS):
        for row_index, (field, ylabel, logarithmic) in enumerate(fields):
            axis = axes[row_index, column]
            if logarithmic:
                axis.set_yscale("log")
            _bar_panel(axis, rows, kind, field)
            if column == 0:
                axis.set_ylabel(ylabel)
        sizes = {
            (int(row["m"]), int(row["n"]), int(row["solver_n"]))
            for row in rows
            if row["problem"] == kind
        }
        constraints, variables, solver_variables = sizes.pop()
        dimensions = (
            rf"$A$: ${constraints}\times{variables}$"
            if solver_variables == variables
            else rf"$[A\ I]$: ${constraints}\times{solver_variables}$"
        )
        axes[0, column].set_title(f"{kind}\n{dimensions}")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Inner solver comparison over "
        f"{len(_select(rows, KINDS[0], CONDITIONS[0], SOLVERS[0]))} "
        f"random instances per family ({_spectrum_label(rows)}); "
        rf"$\sigma={SIGMA:g}$, $\theta={THETA:g}$, "
        rf"$\gamma_x=\gamma_\lambda={GAMMA:g}$; "
        "target: normalized objective error + feasibility "
        rf"$\leq 10^{{-7}}$"
        "\nbars are means, error bars one standard deviation over instances"
    )
    save_figure(fig, output)


def plot_distributions(rows: list[dict], output: Path) -> None:
    """Show the per-instance spread behind the error bars."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), sharex="col")
    for column, kind in enumerate(KINDS):
        for row_index, field in enumerate(
            ("outer_iterations", "inner_iterations")
        ):
            axis = axes[row_index, column]
            if field == "inner_iterations":
                axis.set_yscale("log")
            labels, samples, colors, positions = [], [], [], []
            for slot, (condition, solver) in enumerate(
                (condition, solver)
                for condition in CONDITIONS
                for solver in SOLVERS
            ):
                selected = [
                    row
                    for row in _select(rows, kind, condition, solver)
                    if row["reached_accuracy"]
                ]
                values = np.array([row[field] for row in selected], dtype=float)
                values = values[np.isfinite(values)]
                labels.append(
                    f"{SOLVER_LABELS[solver]}\n{CONDITION_LABELS[condition]}"
                )
                if not values.size:
                    continue
                samples.append(values)
                colors.append(SOLVER_COLORS[solver])
                positions.append(slot)
            drawn = axis.boxplot(
                samples,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker=".", markersize=2),
            )
            for patch, color in zip(drawn["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
                patch.set_edgecolor("#3d3d3a")
            for element in ("medians", "whiskers", "caps"):
                for line in drawn[element]:
                    line.set_color("#3d3d3a")
            axis.set_xticks(np.arange(len(labels)))
            axis.set_xlim(-0.7, len(labels) - 0.3)
            axis.grid(True, axis="y", which="both", alpha=0.3)
            axis.set_axisbelow(True)
            if row_index == 1:
                axis.set_xticklabels(labels, rotation=90, fontsize=6)
        axes[0, column].set_title(kind)
    axes[0, 0].set_ylabel("outer iterations to target")
    axes[1, 0].set_ylabel("inner iterations to target")
    fig.suptitle(
        "Per-instance distributions over "
        f"{len(_select(rows, KINDS[0], CONDITIONS[0], SOLVERS[0]))} "
        f"random instances ({_spectrum_label(rows)}); "
        rf"$\sigma={SIGMA:g}$, $\theta={THETA:g}$, "
        rf"$\gamma_x=\gamma_\lambda={GAMMA:g}$"
    )
    save_figure(fig, output)


def print_summary(summary: list[dict]) -> None:
    header = (
        f"{'problem':<14}{'matrix':<24}{'solver':<22}"
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
            f"{SOLVER_LABELS[entry['inner_solver']]:<22}"
            f"{entry['reached']:>4}/{entry['instances']:<4}"
            f"{outer:>22}{inner:>26}"
        )


def run_study(randomize_spectrum: bool = False) -> None:
    """Run or resume the multi-instance inner-solver comparison.

    With ``randomize_spectrum`` the instances of a family differ in their
    interior singular values as well as their singular vectors, at the same
    ``kappa(A)``; the two samplings are kept in separate output directories.
    """
    output = Path(
        "figures/solver_sweep_randomized_spectrum"
        if randomize_spectrum
        else "figures/solver_sweep"
    )
    results_path = output / "results.csv"
    rows = [
        row
        for row in _read_rows(results_path)
        if row["m"] == CONSTRAINTS
        and row["n"] == VARIABLES
        and float(row["sigma"]) == SIGMA
        and float(row["theta"]) == THETA
        and float(row["gamma_x"]) == GAMMA
        and row["randomize_spectrum"] == randomize_spectrum
    ]
    completed = {_key(row) for row in rows}
    tasks = []
    for kind in KINDS:
        for condition in CONDITIONS:
            for index in range(INSTANCES):
                seed = BASE_SEED + index
                pending = tuple(
                    solver
                    for solver in SOLVERS
                    if (kind, condition, seed, solver) not in completed
                )
                if pending:
                    tasks.append(
                        (kind, condition, seed, pending, randomize_spectrum)
                    )
    print(
        f"{len(rows)} checkpointed results; "
        f"{len(tasks)} instances still to run on {WORKERS} workers",
        flush=True,
    )

    if tasks:
        started = time.perf_counter()
        context = get_context("spawn")
        with context.Pool(WORKERS) as pool:
            for finished, produced in enumerate(
                pool.imap_unordered(_run_task, tasks), start=1
            ):
                rows = [row for row in rows if _key(row) not in {
                    _key(new) for new in produced
                }]
                rows.extend(produced)
                write_rows(results_path, _sorted_rows(rows))
                sample = produced[0]
                elapsed = time.perf_counter() - started
                print(
                    f"[{finished}/{len(tasks)}] {sample['problem']} "
                    f"{sample['matrix']} seed={sample['instance_seed']} "
                    f"({elapsed / finished:.1f}s/instance, "
                    f"{elapsed / 60:.1f} min elapsed)",
                    flush=True,
                )

    rows = _sorted_rows(rows)
    write_rows(results_path, rows)
    summary = summarize(rows)
    write_rows(output / "summary.csv", summary)
    plot_work(rows, output / "solver_comparison.pdf")
    plot_distributions(rows, output / "solver_distributions.pdf")
    print_summary(summary)
    print(f"wrote {output}")


def run_randomized_study() -> None:
    """Repeat the comparison with the interior spectrum redrawn per instance."""
    run_study(randomize_spectrum=True)
