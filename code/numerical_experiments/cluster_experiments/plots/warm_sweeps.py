"""Combine the warm-started sigma and stride wall-time sweeps in one figure."""

import argparse
from pathlib import Path

from experiments.io import write_csv

from . import style
from .main_summary import METHODS, NAMES, encoding
from .sweep_effects import load, summarize


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
    summary = [r for r in summarize(records) if r["initialization"] == "warm"]
    style.apply()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sizes = sorted({r["n"] for r in summary})
    fig, axes = plt.subplots(
        2, len(sizes), figsize=(12, 5.9), sharey="col", squeeze=False
    )
    handles = {}
    for i, sweep in enumerate(("sigma", "stride")):
        for j, n in enumerate(sizes):
            ax = axes[i, j]
            for name in METHODS:
                rows = [
                    r
                    for r in summary
                    if r["sweep"] == sweep and r["n"] == n and r["method"] == name
                ]
                numeric = sorted(
                    (r for r in rows if r["choice_kind"] == "grid"),
                    key=lambda r: r["resolved_value_mean"],
                )
                color, marker = encoding(name)
                (line,) = ax.plot(
                    [r["resolved_value_mean"] for r in numeric],
                    [r["worker_wall_seconds_mean"] for r in numeric],
                    color=color,
                    marker=marker,
                    markersize=4,
                    linewidth=1.2,
                )
                handles[name] = line
                for row in rows:
                    if row["choice_kind"] == "model":
                        ax.plot(
                            row["resolved_value_mean"],
                            row["worker_wall_seconds_mean"],
                            color=color,
                            marker="P",
                            markeredgecolor="black",
                            markeredgewidth=0.65,
                            markersize=7,
                            linestyle="none",
                            zorder=5,
                        )
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
            fixed = r"$s=1$" if sweep == "sigma" else r"$\sigma=0.1$"
            ax.set_title(f"n = {n:,} | {fixed}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ticks = [0.01, 0.05, 0.2, 0.5, 1] if sweep == "sigma" else [1, 2, 5, 10, 20]
            ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
            ax.set_xlabel(
                r"Inexactness $\sigma$" if sweep == "sigma" else r"Stride $s$"
            )
            if j == 0:
                ax.set_ylabel("Mean wall time to convergence (s)")
    legend = [handles[name] for name in METHODS]
    labels = [NAMES[name] for name in METHODS]
    legend.append(Line2D([], [], marker="P", color="black", linestyle="none"))
    labels.append("Model-prescribed setting (GD/CG)")
    fig.legend(legend, labels, loc="lower center", ncol=4)
    fig.suptitle(r"Warm-started QP sweeps | $L/c=10$", y=0.995)
    fig.text(
        0.5,
        0.10,
        "Arithmetic means across seeds 0, 1, 2; all setup and process costs included.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.145, 1, 0.97))
    style.save(fig, args.output / "warm_start_sweeps")
    write_csv(args.output / "warm_start_summary.csv", summary)
    write_csv(args.output / "warm_start_input_hashes.csv", sources)
    print(f"Warm-started figure saved: {len(summary)} three-seed settings")


if __name__ == "__main__":
    main()
