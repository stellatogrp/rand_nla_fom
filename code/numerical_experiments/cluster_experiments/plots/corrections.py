"""Retroactive corrections for timing and work counts in recorded result CSVs.

These read existing results only; no experiment is rerun.

- Clean timing: accounting_v3 added the wall time of earlier attempts to
  ``total_seconds`` and turned it into NaN after any hard kill, although every
  v3 attempt restarts from scratch. The final attempt is the clean measurement.
  Older releases resumed from checkpoints; resumed runs have no clean time.
- Work: measured operator totals retain their recorded unit and scope.
  Historical counters may be partial estimates; recurrence-check work and
  PDHG diagnostic schedules cannot be inferred from iteration counts alone.
  Both work columns carry basis labels that must accompany comparisons.
- Hardware: the node family, from the hostname when no CPU model was recorded.
- Model labels: Nystrom-PCG "optimal" parameters before accounting_v4 came
  from a bound weaker than plain CG's, so they are not model optima.
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.io import read_csv, write_csv

from .work import primary_work


def number(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def timing(row):
    """Clean wall time of one from-scratch attempt and how it was obtained."""
    for name, basis in [
        ("clean_attempt_seconds", "final_attempt"),
        ("attempt_worker_wall_seconds", "final_attempt"),
    ]:
        if math.isfinite(number(row, name)):
            return number(row, name), basis
    parts = [number(row, k) for k in ("solve_seconds", "generation_seconds")]
    process = number(row, "process_seconds")
    if all(math.isfinite(v) for v in [*parts, process]):
        if sum(parts) > process + 1.0:
            return float("nan"), "resumed_from_checkpoint"
        return number(row, "total_seconds"), "single_attempt"
    return number(row, "total_seconds"), "unverified_total"


def clean_seconds(row):
    return timing(row)[0]


MEASURED_SCOPE = "problem_operators_after_build_including_calibration"


def measured(row):
    return (
        str(row.get("work_count_complete")) == "True"
        and row.get("work_count_scope") == MEASURED_SCOPE
        and math.isfinite(number(row, "hessian_products_total"))
    )


def scoped(basis, config):
    """Flag counts whose Hessian unit does not cover the method's work.

    LASSO PDHG multiplies by A and A^T separately, never by A^T A, so its
    measured Hessian count holds only metric evaluations. Compare LASSO work
    across methods with ``design_forward_products``/``design_transpose_products``;
    an ADMM Hessian product is one of each.
    """
    if config["inner"] == "pdhg" and config["family"] == "lasso":
        return basis + "_pdhg_lasso_metrics_only_compare_design_products"
    return basis


def solve_phase_products(row, config):
    """Measured solve work, or explicitly partial historical counters.

    Historical counters omit calibration and may omit final diagnostics.
    Recurrence checks cannot be reconstructed by adding the number of checks.
    """
    if config["inner"] in ("osqp", "scs"):
        return float("nan"), "not_exposed_by_backend"
    if config["family"] == "logistic":
        return float("nan"), "use_gradient_and_design_product_counts"
    if measured(row):
        calibration = number(row, "calibration_hessian_products")
        if config["inner"] == "pdhg":
            calibration = 0.0  # PDHG has no calibration stage.
        if math.isfinite(calibration):
            return (
                number(row, "hessian_products_total") - calibration,
                scoped("measured_solve_phase", config),
            )
        return float("nan"), "missing_calibration_count"
    if config["inner"] == "pdhg":
        return float("nan"), "historical_pdhg_diagnostics_not_recorded"
    if row.get("check_products") not in (None, "") and math.isfinite(
        number(row, "hessian_products_total")
    ):
        return number(
            row, "hessian_products_total"
        ), "partial_recorded_work_excludes_final_diagnostics"
    if config.get("check_source", "recompute") != "recompute":
        return float("nan"), "historical_recurrence_products_not_recorded"
    counted, checks = number(row, "inner_hessian_products"), number(row, "checks")
    if not (math.isfinite(counted) and math.isfinite(checks)):
        return float("nan"), "missing_counts"
    return counted + checks, "estimated_inner_plus_checks_excludes_final_diagnostics"


def hessian_products(row, config):
    """Measured total if available; otherwise an explicitly partial estimate.

    Accounting_v5 and later measure calibration work. Earlier releases yield the
    solve-phase reconstruction, labeled as such; never compare the two scopes
    without subtracting calibration (``solve_phase_products`` does this).
    """
    if config["inner"] not in ("osqp", "scs") and config["family"] != "logistic":
        if measured(row):
            return (
                number(row, "hessian_products_total"),
                scoped("measured_after_build_including_calibration", config),
            )
    return solve_phase_products(row, config)


def node_family(row):
    if row.get("cpu_model"):
        return row["cpu_model"]
    match = re.match(r"^([a-z]+-[a-z]+)", row.get("hostname", ""))
    return f"{match.group(1)} (hostname)" if match else "unknown"


def model_label(row, config):
    kind = row.get("recommendation_kind", "")
    optimal = "optimal" in (str(config.get("sigma")), str(config.get("stride")))
    if (
        optimal
        and config["inner"] == "nystrom_pcg"
        and config.get("implementation")
        not in ("accounting_v4", "accounting_v5", "accounting_v6", "accounting_v7")
    ):
        return "loose_bound_not_model_optimal"
    return kind


def corrected(row):
    config = json.loads(row["config"])
    seconds, basis = timing(row)
    products, product_basis = hessian_products(row, config)
    solve_phase, solve_phase_basis = solve_phase_products(row, config)
    work = primary_work(row, config)
    return dict(
        primary_work_value=work["value"],
        primary_work_unit=work["unit"],
        primary_work_scope=work["scope"],
        primary_work_evidence=work["evidence"],
        run_id=row.get("run_id", ""),
        family=config["family"],
        dataset=config.get("dataset", ""),
        side=config.get("side", ""),
        inner=config["inner"],
        rank=config.get("rank", ""),
        block_size=config.get("block_size", ""),
        inner_initialization=config.get("inner_initialization", ""),
        sweep=config.get("sweep", ""),
        seed=config.get("seed", ""),
        implementation=config.get("implementation", "legacy"),
        blas_threads=config.get("blas_threads", 1),
        status=row.get("status", ""),
        converged=row.get("converged", ""),
        reported_total_seconds=number(row, "total_seconds"),
        clean_seconds=seconds,
        timing_basis=basis,
        cpu_seconds=number(row, "cpu_seconds"),
        reported_hessian_products_total=number(row, "hessian_products_total"),
        reported_inner_hessian_products=number(row, "inner_hessian_products"),
        hessian_products_corrected=products,
        work_count_basis=product_basis,
        # Historical values may be partial; consult solve_phase_basis.
        hessian_products_solve_phase=solve_phase,
        solve_phase_basis=solve_phase_basis,
        # Same unit for every LASSO/logistic method, PDHG included (v5+ only).
        design_forward_products=number(row, "design_forward_products"),
        design_transpose_products=number(row, "design_transpose_products"),
        node_family=node_family(row),
        model_label=model_label(row, config),
    )


def summarize(rows):
    """Hardware mix per instance, so cross-method timings can be stratified."""
    mix = defaultdict(Counter)
    for row in rows:
        mix[(row["family"], row["dataset"], row["side"])][row["node_family"]] += 1
    return [
        dict(
            family=key[0],
            dataset=key[1],
            side=key[2],
            node_families=json.dumps(dict(counts), sort_keys=True),
            mixed_hardware=len(counts) > 1,
        )
        for key, counts in sorted(mix.items(), key=lambda item: str(item[0]))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        corrected(row)
        for directory in args.results_dir
        for path in sorted(directory.glob("*/result.csv"))
        for row in read_csv(path)
        if "config" in row
    ]
    if not rows:
        parser.error("no result CSVs found")
    write_csv(args.output / "corrected_runs.csv", rows)
    write_csv(args.output / "hardware_mix.csv", summarize(rows))
    bases = Counter(row["timing_basis"] for row in rows)
    gd = [
        row["hessian_products_solve_phase"] / row["reported_inner_hessian_products"]
        for row in rows
        if row["inner"] == "gd"
        and row["reported_inner_hessian_products"] > 0
        and math.isfinite(row["hessian_products_solve_phase"])
    ]
    print(f"{len(rows)} runs; timing basis {dict(bases)}")
    if gd:
        print(f"GD work undercount: median factor {np.median(gd):.1f}")


if __name__ == "__main__":
    main()
