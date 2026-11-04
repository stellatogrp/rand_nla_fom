"""Plot separate QP sigma and stride sweeps using seed means without error bars."""

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
from .main_summary import METHODS, MODES, NAMES, encoding, method

METRICS = {
    "wall_clock": ("worker_wall_seconds", "Wall time to convergence (s)"),
    "inner_updates": ("inner_updates", "Inner updates to convergence"),
    "outer_solves": ("outer_solves", "Outer solves to convergence"),
}


def load(directory, manifest):
    """Use complete, independently measured tuning arms without pooling settings."""
    expected = [
        r
        for r in read_csv(manifest)
        if json.loads(r["config"]).get("sweep") in ("sigma", "stride")
    ]
    if not expected:
        raise ValueError("manifest contains no sigma/stride sweep arms")
    records, sources, seen = [], [], set()
    signatures = defaultdict(set)
    for task in expected:
        config = json.loads(task["config"])
        key = task["run_id"]
        if key in seen or identity(config) != key:
            raise ValueError("duplicate or invalid manifest identity")
        seen.add(key)
        path = directory / key / "result.csv"
        rows = read_csv(path)
        if len(rows) != 1 or json.loads(rows[0]["config"]) != config:
            raise ValueError(f"missing or mismatched sweep result: {path}")
        row = rows[0]
        if row.get("run_id") != key or not main_eligible(row):
            raise ValueError(f"ineligible result identity or setup: {path}")
        if config.get("family") != "qp" or config.get("phase") != "tune":
            raise ValueError("expected QP tuning data")
        sweep = config["sweep"]
        parameter = "sigma" if sweep == "sigma" else "stride"
        if config["ratio"] != 10 or (
            config["stride"] != 1 if sweep == "sigma" else config["sigma"] != 0.1
        ):
            raise ValueError("sweep does not hold its reference parameters fixed")
        requested = config[parameter]
        value = float(row[parameter])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid resolved sweep value")
        seconds = clean_seconds(row)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("sweep result lacks an honest independent worker time")
        name = method(config)
        mode = config.get("inner_initialization", "warm")
        signature = {
            k: v
            for k, v in config.items()
            if k not in (parameter, "seed", "initialization_seed")
        }
        signatures[sweep, int(row["n"]), name, mode].add(
            json.dumps(signature, sort_keys=True)
        )
        records.append(
            dict(
                run_id=key,
                sweep=sweep,
                n=int(row["n"]),
                method=name,
                initialization=mode,
                seed=config["seed"],
                requested_value=str(requested),
                resolved_value=value,
                choice_kind="model" if requested == "optimal" else "grid",
                status=row["status"],
                converged=row.get("converged") == "True",
                worker_wall_seconds=seconds,
                inner_updates=float(row["total_inner"]),
                outer_solves=int(row["outer"]) + 1,
                sigma=float(row["sigma"]),
                stride=int(row["stride"]),
                penalty=float(row["c"]),
                ratio=config["ratio"],
                config=json.dumps(config, sort_keys=True),
            )
        )
        sources.append(
            dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
    if any(len(values) != 1 for values in signatures.values()):
        raise ValueError("a sweep changes additional configuration parameters")
    sources.append(
        dict(
            path=str(manifest), sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
        )
    )
    return records, sources


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        key = tuple(
            row[k]
            for k in ("sweep", "n", "method", "initialization", "requested_value")
        )
        groups[key].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        seeds = [r["seed"] for r in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError("duplicate seed within a sweep setting")
        if set(seeds) != {0, 1, 2}:
            raise ValueError("each sweep setting must contain seeds 0, 1, and 2")
        entry = dict(
            zip(
                ("sweep", "n", "method", "initialization", "requested_value"),
                key,
                strict=True,
            )
        )
        entry.update(
            runs=len(rows),
            converged_runs=sum(r["converged"] for r in rows),
            choice_kind=rows[0]["choice_kind"],
            resolved_values=json.dumps([r["resolved_value"] for r in rows]),
            statuses=json.dumps([r["status"] for r in rows]),
        )
        for field in ("resolved_value", *(m[0] for m in METRICS.values())):
            values = [r[field] for r in rows]
            # A censored duration is not a time-to-convergence observation.
            # Preserve the outcomes and mark the entire point as censored.
            valid = field == "resolved_value" or all(r["converged"] for r in rows)
            entry[f"{field}_mean"] = float(np.mean(values)) if valid else float("nan")
        result.append(entry)
    return result


def figure(summary, output, sweep, metric):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected = [r for r in summary if r["sweep"] == sweep]
    sizes = sorted({r["n"] for r in selected})
    field, ylabel = METRICS[metric]
    fig, axes = plt.subplots(3, len(sizes), figsize=(12, 8.5), squeeze=False)
    handles = {}
    for i, mode in enumerate(MODES):
        for j, n in enumerate(sizes):
            ax = axes[i, j]
            for name in METHODS:
                rows = [
                    r
                    for r in selected
                    if r["n"] == n
                    and r["initialization"] == mode
                    and r["method"] == name
                ]
                numeric = sorted(
                    (r for r in rows if r["choice_kind"] == "grid"),
                    key=lambda r: r["resolved_value_mean"],
                )
                color, marker = encoding(name)
                x = [r["resolved_value_mean"] for r in numeric]
                y = [r[f"{field}_mean"] for r in numeric]
                (line,) = ax.plot(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    markersize=4,
                    linewidth=1.1,
                )
                handles[name] = line
                for row in rows:
                    if row["choice_kind"] != "model":
                        continue
                    xx, yy = row["resolved_value_mean"], row[f"{field}_mean"]
                    ax.plot(
                        xx,
                        yy,
                        color=color,
                        marker="P",
                        markeredgecolor="black",
                        markeredgewidth=0.65,
                        markersize=7,
                        linestyle="none",
                        zorder=5,
                    )
                for row in rows:
                    if row["converged_runs"] < row["runs"]:
                        ax.text(
                            row["resolved_value_mean"],
                            0.02,
                            f"{row['converged_runs']}/{row['runs']} converged",
                            transform=ax.get_xaxis_transform(),
                            rotation=90,
                            ha="center",
                            va="bottom",
                            fontsize=6,
                        )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ticks = [0.01, 0.05, 0.2, 0.5, 1] if sweep == "sigma" else [1, 2, 5, 10, 20]
            ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
            ax.set_title(f"n = {n:,} | {mode}")
            ax.set_xlabel(
                r"Inexactness $\sigma$" if sweep == "sigma" else r"Stride $s$"
            )
            if j == 0:
                ax.set_ylabel(ylabel)
    legend_handles = [handles[m] for m in METHODS if m in handles]
    legend_labels = [NAMES[m] for m in METHODS if m in handles]
    legend_handles.append(
        Line2D([], [], marker="P", color="black", linestyle="none", markersize=6)
    )
    legend_labels.append("Model-prescribed setting (GD/CG)")
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=4)
    fixed = r"$s=1$" if sweep == "sigma" else r"$\sigma=0.1$"
    fig.suptitle(
        (r"Effect of $\sigma$" if sweep == "sigma" else r"Effect of stride $s$")
        + f": {ylabel.lower()} | {fixed}, "
        + r"$L/c=10$",
        y=0.995,
    )
    footnote = "Points: arithmetic mean of seeds 0, 1, 2. "
    footnote += (
        "All setup/calibration/process costs included."
        if metric == "wall_clock"
        else "Inner updates are solver-specific counts."
    )
    fig.text(0.5, 0.065, footnote, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    style.save(fig, output / f"{sweep}_{metric}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/qp_tune_v7_20261031")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("campaign_v7/manifests/qp_tune.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("main_figures/sweeps_v7"))
    args = parser.parse_args()
    records, sources = load(args.results_dir, args.manifest)
    summary = summarize(records)
    style.apply()
    for sweep in ("sigma", "stride"):
        for metric in METRICS:
            figure(summary, args.output, sweep, metric)
    write_csv(args.output / "runs.csv", records)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "input_hashes.csv", sources)
    (args.output / "README.md").write_text(
        "# Separate QP sigma and stride sweeps\n\n"
        f"Source: `{args.results_dir}`; {len(records)} runs and {len(summary)} "
        "seed groups. All result identities are checked against the sweep manifest. "
        "Each setting has seeds 0, 1, and 2; initialization modes and sizes remain "
        "separate. Numerical updates use the sealed accounting_v7 release.\n\n"
        "`sigma_wall_clock` and `stride_wall_clock` are the primary figures. "
        "Companion figures show inner updates and outer solves. Each is saved as "
        "PNG and PDF. Points are arithmetic means across the three seeds, "
        "without error bars. The sigma sweep fixes s=1; the stride "
        "sweep fixes sigma=0.1. Both fix L/c=10 (c=0.1 for these QPs).\n\n"
        "Wall time is the measured independent worker time, including problem "
        "generation, preconditioner setup, calibration, and process overhead. "
        "There is no cross-run setup amortization. These untraced sweep timings "
        "are not pooled with the traced fresh_v8 evaluation. Inner updates are "
        "solver-specific counts, not equivalent operator work.\n\n"
        "Filled-plus markers show the separately measured GD/CG model choices. "
        "The x coordinate uses the mean resolved parameter, not the string "
        "'optimal'. If calibration selects different strides across seeds, "
        "this mean can be fractional. These points remain separate from explicit "
        "grid points even when they resolve to the same value. No unsupported "
        "model choice is invented for other methods.\n\n"
        "Censored outcomes remain in the CSVs, with convergence counts labeled "
        "in plots; they are not presented as successful times to convergence. "
        "Baseline and near-exact arms are outside these one-parameter sweeps. "
        "Supporting CSVs record configurations, statuses, statistics, and hashes.\n"
    )
    print(f"{len(records)} sweep runs; {len(summary)} seed groups; six PNG/PDF figures")


if __name__ == "__main__":
    main()
