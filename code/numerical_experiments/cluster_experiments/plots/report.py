"""CSV-only charts with the existing Okabe-Ito colors and marker scheme."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.io import identity, read_csv, write_csv

from . import style
from .accounting import finite_time, main_eligible
from .corrections import clean_seconds, corrected
from .initialization import comparisons

BASELINES = ("osqp", "scs", "pdhg")


def finite_median(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    parser.add_argument(
        "--include-pilots",
        action="store_true",
        help="show explicitly separated incremental pilot timings",
    )
    args = parser.parse_args()
    style.apply()
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    # Tuning-phase baselines carry sweep="baseline"; draw them on every sweep
    # figure of the same instance instead of on a figure of their own.
    references = defaultdict(list)
    all_rows = []
    excluded = []
    for path in sorted(args.results_dir.glob("*/result.csv")):
        for row in read_csv(path):
            if not main_eligible(row) and not args.include_pilots:
                excluded.append(row)
                continue
            config = json.loads(row["config"])
            row.update({k: v for k, v in config.items() if k not in row})
            all_rows.append(row)
            group = (
                config["family"],
                config["dataset"],
                config["orientation"],
                config.get("side", ""),
                config.get("sweep", ""),
                identity(config["matrix_source"])[:8]
                if "matrix_source" in config
                else "",
                config.get("implementation", "legacy"),
                config.get("blas_threads", 1),
                row.get("setup_policy", "independent"),
                row.get("timing_scope", "legacy_setup_plus_solve"),
            )
            if config.get("sweep") == "baseline":
                references[group[:4] + group[5:]].append(row)
            else:
                groups[group].append(row)
    for instance, rows in references.items():
        if not any(group[:4] + group[5:] == instance for group in groups):
            groups[instance[:4] + ("baseline",) + instance[4:]] = rows
    if not all_rows:
        parser.error(
            "no eligible main result CSVs; pilot plots require --include-pilots"
        )
    if excluded:
        write_csv(args.output / "excluded_accounting.csv", excluded)
    write_csv(args.output / "all_runs.csv", all_rows)
    pairs = comparisons(all_rows)
    if pairs:
        write_csv(args.output / "initialization_comparison.csv", pairs)
    summary = []
    for group, rows in groups.items():
        series = defaultdict(list)
        if group[4] != "baseline":
            rows = rows + references.get(group[:4] + group[5:], [])
        for row in rows:
            cfg = json.loads(row["config"])
            label = cfg["inner"]
            for key in ["rank", "block_size", "sigma", "stride", "ratio", "gd_factor"]:
                if key in cfg:
                    label += f" {key}={cfg[key]}"
            if cfg["inner"] not in ("osqp", "scs", "pdhg"):
                mode = cfg.get("inner_initialization", "warm")
                label += f" init={mode}"
                if mode == "gaussian":
                    label += f" scale={cfg.get('gaussian_scale', 1.0)}"
                elif mode == "fixed":
                    label += f" value={cfg.get('fixed_initial_value', 0.0)}"
            series[label].append(row)
        # Paginate so the six validated sweep encodings are never exhausted;
        # baselines use the reference color and repeat on every page.
        ordered = sorted(
            item for item in series.items() if item[0].split()[0] not in BASELINES
        )
        baselines = sorted(
            item for item in series.items() if item[0].split()[0] in BASELINES
        )
        for page in range(0, max(len(ordered), 1), 6):
            chunk = ordered[page : page + 6] + baselines
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
            for i, (label, runs) in enumerate(chunk):
                converged = [r for r in runs if r.get("converged") == "True"]
                method = label.split()[0]
                if method in BASELINES:
                    color, marker = style.REFERENCE, style.BASELINE_MARKERS[method]
                elif "optimal" in label:
                    color, marker = style.ACCENT, style.ACCENT_MARKER
                else:
                    color, marker = style.SWEEP_COLORS[i], style.SWEEP_MARKERS[i]
                times = [clean_seconds(r) for r in converged if finite_time(r)]
                fixes = [corrected(r) for r in runs]
                if times:
                    quantiles = np.percentile(times, [25, 50, 75])
                    axes[0].errorbar(
                        quantiles[1],
                        i,
                        xerr=[
                            [quantiles[1] - quantiles[0]],
                            [quantiles[2] - quantiles[1]],
                        ],
                        fmt=marker,
                        color=color,
                    )
                else:
                    axes[0].text(
                        0.02,
                        i,
                        "no eligible timing",
                        va="center",
                        fontsize=7,
                        transform=axes[0].get_yaxis_transform(),
                    )
                residuals = [
                    max(
                        float(r.get("stationarity", "nan")),
                        float(r.get("feasibility", "nan")),
                    )
                    for r in runs
                ]
                residuals = [v for v in residuals if np.isfinite(v)]
                if residuals:
                    axes[1].plot(
                        max(np.median(residuals), 1e-16), i, marker=marker, color=color
                    )
                summary.append(
                    dict(
                        family=group[0],
                        dataset=group[1],
                        orientation=group[2],
                        side=group[3],
                        sweep=group[4],
                        configuration=label,
                        implementation=group[-4],
                        blas_threads=group[-3],
                        setup_policy=group[-2],
                        timing_scope=group[-1],
                        runs=len(runs),
                        converged=len(converged),
                        median_clean_seconds=np.median(times)
                        if times
                        else float("nan"),
                        median_cpu_seconds=finite_median(
                            [f["cpu_seconds"] for f in fixes]
                        ),
                        primary_work_unit=fixes[0]["primary_work_unit"],
                        primary_work_scope=(
                            "problem_operators_after_build_including_calibration"
                            if any(
                                f["primary_work_evidence"] == "measured" for f in fixes
                            )
                            else "unknown"
                        ),
                        measured_work_runs=sum(
                            f["primary_work_evidence"] == "measured" for f in fixes
                        ),
                        median_measured_primary_work_to_convergence=finite_median(
                            [
                                f["primary_work_value"]
                                for f, r in zip(fixes, runs, strict=True)
                                if r.get("converged") == "True"
                            ]
                        ),
                        hessian_work_role="diagnostic",
                        median_hessian_products_corrected=finite_median(
                            [f["hessian_products_corrected"] for f in fixes]
                        ),
                        median_design_forward_products=finite_median(
                            [f["design_forward_products"] for f in fixes]
                        ),
                        median_design_transpose_products=finite_median(
                            [f["design_transpose_products"] for f in fixes]
                        ),
                        work_count_bases=json.dumps(
                            sorted({f["work_count_basis"] for f in fixes})
                        ),
                        solve_phase_bases=json.dumps(
                            sorted({f["solve_phase_basis"] for f in fixes})
                        ),
                        timing_bases=json.dumps(
                            sorted({f["timing_basis"] for f in fixes})
                        ),
                        node_families=json.dumps(
                            sorted({f["node_family"] for f in fixes})
                        ),
                        model_label=";".join(
                            sorted({f["model_label"] for f in fixes} - {""})
                        ),
                    )
                )
            labels = [
                f"{label} [{sum(r.get('converged') == 'True' for r in runs)}"
                f"/{len(runs)}]"
                for label, runs in chunk
            ]
            axes[0].set_yticks(range(len(chunk)), labels)
            axes[1].set_yticks(range(len(chunk)), [""] * len(chunk))
            axes[0].set_xlabel(
                "Incremental pilot time (s); shared setup recorded separately"
                if group[-2] == "shared_pilot"
                else "Clean single-attempt time including setup (s), converged runs"
            )
            axes[1].set_xlabel("Final common KKT residual (all runs)")
            axes[0].set_xscale("log")  # selection uses geometric means
            axes[1].set_xscale("log")
            fig.suptitle(" / ".join(str(v) for v in group if v != ""))
            stem = "_".join(str(v) for v in group) + f"_{page // 6}"
            style.save(fig, args.output / stem)
    write_csv(args.output / "summary.csv", summary)


if __name__ == "__main__":
    main()
