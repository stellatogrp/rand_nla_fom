"""Create mean-only, warm-started versions of the four fresh QP figures."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.io import write_csv

from . import style
from .main_summary import (
    BASELINES,
    METHODS,
    NAMES,
    contraction_rows,
    encoding,
    grouping,
    load,
    points,
    timing_parts,
)


def mean_curve(series):
    """Arithmetic mean of log-interpolated residuals on common observed support."""
    left = max(x[0] for x, _ in series)
    right = min(x[-1] for x, _ in series)
    if right < left:
        raise ValueError("replicate histories have no common support")
    grid = np.unique(np.concatenate([x[(x >= left) & (x <= right)] for x, _ in series]))
    values = np.array([10 ** np.interp(grid, x, np.log10(y)) for x, y in series])
    return grid, np.mean(values, axis=0)


def convergence(runs, output, axis):
    import matplotlib.pyplot as plt

    sizes = sorted({r["n"] for r in runs})
    groups = grouping(runs)
    fig, axes = plt.subplots(1, len(sizes), figsize=(12, 3.6), squeeze=False)
    handles, table = {}, []
    for j, n in enumerate(sizes):
        ax = axes[0, j]
        for name in METHODS + (("pdhg",) if axis == "inner" else ()):
            records = groups.get((n, "warm", name), [])
            if not records:
                continue
            grid, mean = mean_curve([points(r, axis) for r in records])
            color, marker = encoding(name)
            (line,) = ax.plot(
                grid,
                mean,
                color=color,
                marker=marker,
                markersize=3,
                markevery=max(1, len(grid) // 7),
            )
            handles[name] = line
            table.extend(
                dict(
                    n=n,
                    method=name,
                    coordinate=float(x),
                    mean_residual=float(y),
                    seeds=len(records),
                )
                for x, y in zip(grid, mean, strict=True)
            )
        ax.axhline(1e-6, color=style.MUTED, linestyle=":", linewidth=1)
        ax.set_yscale("log")
        ax.set_title(f"n = {n:,}")
        ax.set_xlabel(
            "Cumulative inner updates" if axis == "inner" else "Outer solves completed"
        )
        if j == 0:
            ax.set_ylabel("Mean max(stationarity, feasibility)")
    fig.legend(
        list(handles.values()), [NAMES[k] for k in handles], loc="lower center", ncol=4
    )
    fig.suptitle(
        "Warm-started QP convergence: "
        + ("inner iterations" if axis == "inner" else "outer iterations"),
        y=0.995,
    )
    note = "Arithmetic means across seeds 3, 4, 5; common observed support only."
    if axis == "inner":
        note += " PDHG uses native iterations."
    fig.text(0.5, 0.15, note, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.24, 1, 0.96))
    stem = "02_qp_inner_convergence" if axis == "inner" else "01_qp_outer_convergence"
    style.save(fig, output / stem)
    write_csv(output / f"{axis}_mean_convergence.csv", table)


def timing_components(row):
    """Disjoint measured phases whose sum is the independent worker wall time."""
    solve, setup, other, total = timing_parts(row)
    generation = float(row["generation_seconds"])
    calibration = float(row["tuning_seconds"])
    overhead = other - generation - calibration
    if min(generation, calibration) < 0 or overhead < -1e-6:
        raise ValueError("overlapping or negative timing components")
    return np.array([solve, setup, generation, calibration, max(0.0, overhead), total])


def wall_time(runs, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    sizes = sorted({r["n"] for r in runs})
    groups = grouping(runs)
    fig, axes = plt.subplots(1, len(sizes), figsize=(12, 4.8), squeeze=False)
    order, table = METHODS + BASELINES, []
    phases = (
        ("Solve (including diagnostics)", style.MEASURED, ""),
        ("Solver/preconditioner setup", style.ACCENT, "///"),
        ("Problem construction", style.OKABE_ITO["green"], "xx"),
        ("Calibration", style.OKABE_ITO["sky"], "\\\\"),
        ("Process and I/O overhead", style.OKABE_ITO["purple"], ".."),
    )
    for j, n in enumerate(sizes):
        ax = axes[0, j]
        for x, name in enumerate(order):
            records = groups.get((n, "warm", name), [])
            if not records:
                ax.text(
                    x,
                    0.03,
                    "not evaluated",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    transform=ax.get_xaxis_transform(),
                )
                continue
            if len(records) != 3 or any(
                r["row"]["converged"] != "True" for r in records
            ):
                raise ValueError("mean timing requires all three converged seeds")
            mean = np.mean([timing_components(r["row"]) for r in records], axis=0)
            bottom = 0.0
            for height, (_, color, hatch) in zip(mean[:5], phases, strict=True):
                ax.bar(
                    x,
                    height,
                    bottom=bottom,
                    width=0.72,
                    color=color,
                    edgecolor="black",
                    linewidth=0.35,
                    hatch=hatch,
                )
                bottom += height
            table.append(
                dict(
                    n=n,
                    method=name,
                    seeds=3,
                    mean_solve_seconds=mean[0],
                    mean_solver_setup_seconds=mean[1],
                    mean_generation_seconds=mean[2],
                    mean_calibration_seconds=mean[3],
                    mean_process_overhead_seconds=mean[4],
                    mean_build_calibration_overhead_seconds=sum(mean[2:5]),
                    mean_worker_wall_seconds=mean[5],
                )
            )
        ax.set_yscale("log")
        ax.set_title(f"n = {n:,}")
        ax.set_xticks(
            range(len(order)),
            [NAMES[m].replace("Nystrom-PCG", "Nys-PCG") for m in order],
            rotation=55,
            ha="right",
        )
        if j == 0:
            ax.set_ylabel("Mean independent worker wall time (s)")
    fig.suptitle("Warm-started QP wall-clock time", y=0.995)
    fig.legend(
        handles=[
            Patch(facecolor=color, edgecolor="black", hatch=hatch, label=label)
            for label, color, hatch in phases
        ],
        loc="lower center",
        ncol=3,
    )
    fig.text(
        0.5,
        0.115,
        "Arithmetic means across seeds 3, 4, 5; "
        "all setup, calibration, instrumentation, and process costs included.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 0.96))
    style.save(fig, output / "03_qp_wall_clock")
    write_csv(output / "mean_timings.csv", table)


def theory_means(rows):
    """Deduplicate audits, average observed outer solves, then weight seeds equally."""
    events = {}
    for row in rows:
        key = row["method"], row["n"], row["local_inner"], row["run_id"], row["outer"]
        # A true boundary audit replaces the recurrence event at the same point.
        if key not in events or str(row["outer_boundary"]) == "True":
            events[key] = row
    per_run = defaultdict(list)
    for (method, n, step, run_id, _outer), row in events.items():
        per_run[method, n, step, run_id].append(row["measured_squared_ratio"])
    groups = defaultdict(list)
    for (method, n, step, _run_id), values in per_run.items():
        groups[method, n, step].append(float(np.mean(values)))
    return [
        dict(
            method=m, n=n, local_inner=s, mean_squared_ratio=float(np.mean(v)), seeds=3
        )
        for (m, n, s), v in sorted(groups.items())
        if len(v) == 3
    ]


def theory(runs, output):
    import matplotlib.pyplot as plt

    rows, excluded = contraction_rows(runs)
    means = theory_means(rows)
    sizes = sorted({r["n"] for r in means})
    fig, axes = plt.subplots(2, len(sizes), figsize=(12, 5.8), squeeze=False)
    for i, name in enumerate(("gd", "cg")):
        for j, n in enumerate(sizes):
            ax = axes[i, j]
            selected = [r for r in means if r["method"] == name and r["n"] == n]
            constants = {
                (r["alpha"], r["c0"])
                for r in rows
                if r["method"] == name and r["n"] == n
            }
            if not selected or len(constants) != 1:
                raise ValueError("theory panel lacks matched three-seed observations")
            alpha, c0 = next(iter(constants))
            ax.plot(
                [r["local_inner"] for r in selected],
                [r["mean_squared_ratio"] for r in selected],
                color=style.MEASURED,
                marker="o",
                markersize=3,
                label="Observed mean: warm start",
            )
            steps = np.arange(max(r["local_inner"] for r in selected) + 1)
            ax.plot(
                steps,
                c0 * alpha**steps,
                color=style.BOUND,
                linestyle="--",
                label=r"Deterministic bound $C_0\alpha^{\ell}$",
            )
            ax.set_yscale("log")
            ax.set_title(f"{NAMES[name]} | n = {n:,}")
            ax.set_xlabel(r"Local inner updates $\ell$")
            if j == 0:
                ax.set_ylabel(r"Mean $\|e_\ell\|^2 / \|e_0\|^2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Warm-started inner convergence versus theoretical bounds", y=0.995)
    fig.text(
        0.5,
        0.065,
        "Mean of observed outer solves within each seed, "
        "then equal-weight mean across seeds 3, 4, 5.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    style.save(fig, output / "04_qp_theoretical_bounds")
    write_csv(output / "mean_theory_comparison.csv", means)
    write_csv(output / "theory_exclusions.csv", excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/main_figures_v8_20261103")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("figure_campaign/inputs/evaluate.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("main_figures/fresh_v8/warm_only")
    )
    args = parser.parse_args()
    runs, sources = load(args.results_dir, args.manifest)
    runs = [r for r in runs if r["mode"] == "warm"]
    for records in grouping(runs).values():
        if {r["config"]["seed"] for r in records} != {3, 4, 5} or len(records) != 3:
            raise ValueError("expected exactly seeds 3, 4, and 5 for each method/size")
    style.apply()
    convergence(runs, args.output, "outer")
    convergence(runs, args.output, "inner")
    wall_time(runs, args.output)
    theory(runs, args.output)
    write_csv(args.output / "input_hashes.csv", sources)
    (args.output / "README.md").write_text(
        "# Warm-started fresh_v8 figures\n\n"
        "Four PNG/PDF figures use only warm-started splitting methods, plus native "
        "baseline references. Each plotted series uses seeds 3, 4, and 5. "
        "Curves and stacked bar components show arithmetic means without error "
        "bars or shaded ranges.\n\n"
        "Convergence residuals are interpolated in log space within the common "
        "observed support, converted back to residual units, then averaged "
        "arithmetically. No terminal extrapolation or individual-seed markers "
        "are added. Inner updates are solver-specific; PDHG uses native iterations. "
        "OSQP/SCS expose insufficient native histories for convergence curves.\n\n"
        "Wall time includes every independent setup, calibration, instrumentation, "
        "CSV I/O, and process cost. Stacked bars separate solve (including "
        "diagnostics), solver/preconditioner setup, problem construction, "
        "calibration, and remaining process/I/O overhead. The timing CSV also "
        "exports those mean components, which sum to the total worker time. "
        "Missing large-size OSQP evaluations remain labeled.\n\n"
        "The theory figure compares mean squared normalized inner graph residuals "
        "with verified GD/CG contraction bounds. Duplicate boundary audits replace "
        "recurrence observations; observed outer solves are averaged within each "
        "seed, then seeds receive equal weight. Only local update counts observed "
        "in all three seeds are shown. The original individual-observation audit "
        "remains in ../theory_comparison.csv. No outer KKT rate is claimed.\n\n"
        "Reproduce with `uv run python -m plots.warm_evaluation`.\n"
    )
    print(f"Four warm-started figures generated from {len(runs)} independent runs")


if __name__ == "__main__":
    main()
