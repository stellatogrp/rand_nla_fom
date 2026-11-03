"""Create auditable QP summary figures from independent CSV results and traces."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.io import identity, read_csv, write_csv

from . import style
from .accounting import main_eligible
from .corrections import clean_seconds

MODES = ("warm", "fixed", "gaussian")
METHODS = ("gd", "agd", "cg", "gmres", "nystrom_pcg_32", "nystrom_pcg_128")
BASELINES = ("pdhg", "scs", "osqp")
NAMES = dict(
    gd="GD",
    agd="AGD",
    cg="CG",
    gmres="GMRES",
    nystrom_pcg_32="Nystrom-PCG (32)",
    nystrom_pcg_128="Nystrom-PCG (128)",
    pdhg="PDHG",
    scs="SCS",
    osqp="OSQP",
)


def method(config):
    name = config["inner"]
    return f"{name}_{config['rank']}" if name == "nystrom_pcg" else name


def encoding(name):
    if name in BASELINES:
        return style.REFERENCE, style.BASELINE_MARKERS[name]
    index = METHODS.index(name)
    return style.SWEEP_COLORS[index], style.SWEEP_MARKERS[index]


def load(directory, manifest=None, phase="evaluate"):
    """Require exact identities and pair histories with their final attempt."""
    expected = (
        {r["run_id"]: json.loads(r["config"]) for r in read_csv(manifest)}
        if manifest
        else {}
    )
    runs, sources = [], []
    for path in sorted(directory.glob("*/result.csv")):
        rows = read_csv(path)
        if len(rows) != 1:
            raise ValueError(f"expected one final result: {path}")
        row = rows[0]
        config = json.loads(row["config"])
        if config.get("family") != "qp" or config.get("phase") != phase:
            continue
        if identity(config) != row["run_id"]:
            raise ValueError(f"result identity mismatch: {path}")
        if expected and expected.get(row["run_id"]) != config:
            raise ValueError(f"unexpected evaluation result: {path}")
        if not main_eligible(row) or not math.isfinite(clean_seconds(row)):
            raise ValueError(f"ineligible setup/timing: {path}")
        if row.get("trace_complete") == "False":
            raise ValueError(f"truncated inner trace: {path}")
        candidates = []
        for attempt in path.parent.glob("attempts/*/result.csv"):
            other = read_csv(attempt)[0]
            if all(
                other.get(k) == row.get(k)
                for k in (
                    "run_id",
                    "status",
                    "total_inner",
                    "stationarity",
                    "feasibility",
                )
            ):
                candidates.append(attempt.parent)
        if len(candidates) != 1:
            raise ValueError(f"cannot uniquely pair final history: {path}")
        attempt = candidates[0]
        history = read_csv(attempt / "history.csv")
        inner = read_csv(attempt / "inner_history.csv")
        if not history:
            raise ValueError(f"missing convergence history: {path}")
        for source in (path, attempt / "history.csv", attempt / "inner_history.csv"):
            if source.exists():
                sources.append(
                    dict(
                        path=str(source),
                        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    )
                )
        runs.append(
            dict(
                row=row,
                config=config,
                history=history,
                inner=inner,
                method=method(config),
                n=int(row["n"]),
                mode=config.get("inner_initialization", "warm"),
            )
        )
    if not runs or (expected and {r["row"]["run_id"] for r in runs} != set(expected)):
        raise ValueError(
            "evaluation is empty or incomplete; figures require every manifest row"
        )
    return runs, sources


def timing_parts(row):
    """Keep every charged setup/overhead second; never amortize across seeds."""
    total = clean_seconds(row)
    solve = float(row["solve_seconds"])
    setup = float(row["solver_setup_seconds"])
    other = total - solve - setup
    if min(total, solve, setup) < 0 or other < -1e-6:
        raise ValueError("invalid or overlapping time components")
    return solve, setup, max(0.0, other), total


def points(run, axis):
    rows = run["inner"] if axis == "inner" and run["inner"] else run["history"]
    values = {}
    for row in rows:
        x = float(row["inner"]) if axis == "inner" else float(row["iteration"]) + 1
        y = max(float(row["stationarity"]), float(row["feasibility"]))
        if math.isfinite(x) and math.isfinite(y):
            values[x] = max(y, 1e-16)
    # Final CSV metrics are independently certified even for recurrence traces.
    row = run["row"]
    if values:
        values[max(values)] = max(
            float(row["stationarity"]), float(row["feasibility"]), 1e-16
        )
    return np.array(sorted(values)), np.array([values[x] for x in sorted(values)])


def common_curve(series):
    """Interpolate log diagnostics only within every replicate's observed range."""
    left = max(x[0] for x, _ in series)
    right = min(x[-1] for x, _ in series)
    if right < left:
        raise ValueError("replicate histories have no common support")
    grid = np.unique(np.concatenate([x[(x >= left) & (x <= right)] for x, _ in series]))
    logs = np.array([np.interp(grid, x, np.log10(y)) for x, y in series])
    return (
        grid,
        10 ** np.median(logs, axis=0),
        10 ** np.min(logs, axis=0),
        10 ** np.max(logs, axis=0),
    )


def grouping(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[run["n"], run["mode"], run["method"]].append(run)
    return groups


def convergence(runs, output, axis):
    import matplotlib.pyplot as plt

    sizes = sorted({r["n"] for r in runs})
    groups = grouping(runs)
    fig, axes = plt.subplots(3, len(sizes), figsize=(12, 8.4), squeeze=False)
    handles = {}
    for i, mode in enumerate(MODES):
        for j, n in enumerate(sizes):
            ax = axes[i, j]
            for name in METHODS + (("pdhg",) if axis == "inner" else ()):
                records = groups.get(
                    (n, "warm" if name in BASELINES else mode, name), []
                )
                if not records:
                    continue
                series = [points(r, axis) for r in records]
                grid, median, low, high = common_curve(series)
                color, marker = encoding(name)
                (line,) = ax.plot(
                    grid,
                    median,
                    color=color,
                    marker=marker,
                    markevery=max(1, len(grid) // 7),
                    markersize=3,
                    label=NAMES[name],
                )
                handles[name] = line
                ax.fill_between(grid, low, high, color=color, alpha=0.10, linewidth=0)
                for x, y in series:
                    ax.plot(
                        x[-1],
                        y[-1],
                        marker=marker,
                        color=color,
                        alpha=0.4,
                        markersize=3,
                    )
            ax.axhline(1e-6, color=style.MUTED, linestyle=":", linewidth=1)
            ax.set_yscale("log")
            ax.set_title(f"n = {n:,} | {mode}")
            ax.set_xlabel(
                "Cumulative inner updates"
                if axis == "inner"
                else "Outer solves completed"
            )
            if j == 0:
                ax.set_ylabel("max(stationarity, feasibility)")
    fig.legend(
        list(handles.values()), [NAMES[k] for k in handles], loc="lower center", ncol=4
    )
    subtitle = (
        "Inner updates are solver-specific; PDHG uses native iterations."
        if axis == "inner"
        else "Splitting outer solves; native baselines have no matching outer loop."
    )
    fig.suptitle(
        "QP convergence: "
        + ("inner iterations" if axis == "inner" else "outer iterations"),
        y=0.995,
    )
    fig.text(
        0.5,
        0.065,
        subtitle + " Median and range across three held-out seeds.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.095, 1, 0.97))
    style.save(
        fig,
        output
        / ("02_qp_inner_convergence" if axis == "inner" else "01_qp_outer_convergence"),
    )


def wall_time(runs, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    sizes = sorted({r["n"] for r in runs})
    groups = grouping(runs)
    fig, axes = plt.subplots(3, len(sizes), figsize=(12, 9), squeeze=False)
    order = METHODS + BASELINES
    for i, mode in enumerate(MODES):
        for j, n in enumerate(sizes):
            ax = axes[i, j]
            for x, name in enumerate(order):
                records = groups.get(
                    (n, "warm" if name in BASELINES else mode, name), []
                )
                good = [r for r in records if r["row"].get("converged") == "True"]
                color, marker = encoding(name)
                if not good:
                    label = "not evaluated" if not records else "censored"
                    ax.text(
                        x,
                        0.03,
                        label,
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        transform=ax.get_xaxis_transform(),
                    )
                    continue
                good.sort(key=lambda r: clean_seconds(r["row"]))
                representative = good[len(good) // 2]
                solve, setup, other, total = timing_parts(representative["row"])
                bottom = 0.0
                for height, hatch, alpha in (
                    (solve, "", 0.9),
                    (setup, "///", 0.55),
                    (other, "..", 0.25),
                ):
                    ax.bar(
                        x,
                        height,
                        bottom=bottom,
                        width=0.72,
                        color=color,
                        edgecolor="black",
                        linewidth=0.35,
                        hatch=hatch,
                        alpha=alpha,
                    )
                    bottom += height
                times = [clean_seconds(r["row"]) for r in good]
                ax.errorbar(
                    x,
                    total,
                    yerr=[[total - min(times)], [max(times) - total]],
                    color="black",
                    marker=marker,
                    markersize=3,
                    capsize=2,
                    linewidth=0.7,
                )
            ax.set_yscale("log")
            ax.set_xticks(
                range(len(order)),
                [NAMES[k].replace("Nystrom-PCG", "Nys-PCG") for k in order],
                rotation=60,
                ha="right",
            )
            ax.set_title(f"n = {n:,} | {mode}")
            if j == 0:
                ax.set_ylabel("Independent worker wall time (s)")
    fig.legend(
        handles=[
            Patch(facecolor="gray", label="Solve"),
            Patch(facecolor="lightgray", hatch="///", label="Solver setup"),
            Patch(
                facecolor="white",
                hatch="..",
                label="Build, calibration, and process overhead",
            ),
        ],
        loc="lower center",
        ncol=3,
    )
    fig.suptitle("QP wall-clock time, including all setup and instrumentation", y=0.995)
    fig.text(
        0.5,
        0.045,
        "Bars: median-time run; whiskers: seed range. "
        "Baselines repeat across initialization rows for reference.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    style.save(fig, output / "03_qp_wall_clock")


def contraction_rows(runs):
    """Use verified GD/CG constants; do not invent Nystrom or GMRES bounds."""
    rows, excluded = [], []
    for run in runs:
        row, config = run["row"], run["config"]
        if config["inner"] not in ("gd", "cg") or not run["inner"]:
            excluded.append(
                dict(
                    run_id=row["run_id"],
                    reason="no verified recorded contraction model/inner trace",
                )
            )
            continue
        alpha, c0 = float(row["alpha"]), float(row["c0"])
        if not 0 < alpha < 1 or c0 < 1:
            raise ValueError("invalid contraction constants")
        for event in run["inner"]:
            initial = float(event["initial_graph_residual"])
            if initial <= 1e-300:
                continue
            measured = (float(event["graph_residual"]) / initial) ** 2
            step = int(event["local_inner"])
            bound = c0 * alpha**step
            rows.append(
                dict(
                    run_id=row["run_id"],
                    n=run["n"],
                    method=run["method"],
                    mode=run["mode"],
                    outer=int(event["iteration"]),
                    local_inner=step,
                    alpha=alpha,
                    c0=c0,
                    measured_squared_ratio=measured,
                    deterministic_upper_bound=bound,
                    gradient_source=event["gradient_source"],
                    outer_boundary=event["outer_boundary"],
                    bound_ratio=measured / bound if bound > 0 else float("nan"),
                    exceeds_bound_with_roundoff_slack=measured
                    > bound * (1 + 1e-8) + 1e-24,
                )
            )
    return rows, excluded


def theory_figure(rows, output):
    import matplotlib.pyplot as plt

    if not rows:
        return False
    sizes = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(2, len(sizes), figsize=(12, 5.8), squeeze=False)
    for i, name in enumerate(("gd", "cg")):
        for j, n in enumerate(sizes):
            ax = axes[i, j]
            selected = [r for r in rows if r["method"] == name and r["n"] == n]
            for k, mode in enumerate(MODES):
                maxima = defaultdict(list)
                for row in selected:
                    if row["mode"] == mode:
                        maxima[row["local_inner"]].append(row["measured_squared_ratio"])
                steps = sorted(maxima)
                ax.plot(
                    steps,
                    [max(maxima[s]) for s in steps],
                    color=style.SWEEP_COLORS[k],
                    marker=style.SWEEP_MARKERS[k],
                    markersize=3,
                    label=f"Observed max: {mode}",
                )
            constants = {(r["alpha"], r["c0"]) for r in selected}
            if len(constants) != 1:
                raise ValueError("theory panel requires matched contraction parameters")
            alpha, c0 = next(iter(constants))
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
                ax.set_ylabel(r"$\|e_\ell\|^2 / \|e_0\|^2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(
        "Measured inner convergence versus verified contraction bounds", y=0.995
    )
    fig.text(
        0.5,
        0.065,
        "Maximum over observed outer solves and seeds; "
        "CG uses recurrence diagnostics plus true boundary audits.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    style.save(fig, output / "04_qp_theoretical_bounds")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("main_figures"))
    args = parser.parse_args()
    if args.manifest and "figure_campaign" in args.manifest.parts:
        from figure_campaign.runner import verified

        verified(args.manifest)
    runs, sources = load(args.results_dir, args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    style.apply()
    summary = []
    for run in runs:
        row = run["row"]
        solve, setup, other, total = timing_parts(row)
        summary.append(
            dict(
                run_id=row["run_id"],
                n=run["n"],
                method=run["method"],
                mode=run["mode"],
                seed=run["config"]["seed"],
                status=row["status"],
                solve_seconds=solve,
                solver_setup_seconds=setup,
                build_calibration_overhead_seconds=other,
                worker_wall_seconds=total,
                outer_solves=len(run["history"]) if run["method"] in METHODS else "",
                inner_updates=row["total_inner"],
                config=json.dumps(run["config"], sort_keys=True),
            )
        )
    write_csv(args.output / "runs.csv", summary)
    write_csv(args.output / "input_hashes.csv", sources)
    convergence(runs, args.output, "outer")
    convergence(runs, args.output, "inner")
    wall_time(runs, args.output)
    theory, excluded = contraction_rows(runs)
    if theory:
        write_csv(args.output / "theory_comparison.csv", theory)
    if excluded:
        write_csv(args.output / "theory_exclusions.csv", excluded)
    plotted = theory_figure(theory, args.output)
    violations = sum(r["exceeds_bound_with_roundoff_slack"] for r in theory)
    (args.output / "README.md").write_text(
        "# QP main figures\n\n"
        f"Source: `{args.results_dir}`. {len(runs)} independent evaluation runs.\n\n"
        "01: outer convergence; 02: cumulative inner convergence; 03: wall-clock "
        "time including setup, build, calibration, trace verification/I/O, and process "
        "overhead. Lines show the median and full seed range on common observed "
        "support, using log-linear interpolation between saved diagnostics. End "
        "markers show individual terminal observations; there is no extrapolation. "
        "The first saved outer point follows the first solve; "
        "it is not the zero initial iterate. "
        "Inner updates are solver-specific, not equivalent operator work. "
        "OSQP/SCS are timed but omitted from iterative convergence plots because "
        "their native histories expose only tolerance-ladder endpoints. PDHG is "
        "shown on the inner-axis plot in native iterations. Baselines repeat across "
        "initialization panels only for reference; runs.csv counts each once.\n\n"
        "04: the squared inner graph residual relative to its initial value, "
        "compared with verified deterministic GD/CG contraction bounds. Maxima "
        "are over observed outer solves/seeds at each local update. CG intermediate "
        "values come from residual recurrence; outer boundaries are independently "
        "audited with the true gradient. The CSV labels both sources. No finite-run "
        "outer KKT or total-work theorem is claimed by this inner-contraction figure. "
        "Unsupported Nystrom/AGD/GMRES bounds are explicitly excluded.\n\n"
        f"Theory figure available: {plotted}. Observations exceeding the bound "
        f"with numerical slack: {violations}. Inspect theory_comparison.csv; "
        "exceedances are retained, never hidden or clipped.\n\n"
        "Parameters, identities, statuses, and input hashes are exported as CSV. "
        "Fresh v8 runs hold the warm-selected parameters fixed across initialization "
        "modes. Older v7 plots use independently selected parameters by mode and "
        "therefore are not a controlled initialization ablation. All timing includes "
        "instrumentation; traced timing should not be pooled with untraced releases. "
        "Kaczmarz screening did not converge and is excluded from evaluation.\n"
    )
    print(
        f"{len(runs)} runs; {len(theory)} bound comparisons; {violations} exceedances"
    )


if __name__ == "__main__":
    main()
