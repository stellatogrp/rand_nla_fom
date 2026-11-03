"""Export separate measured-work and historical-estimate publication tables."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.io import identity, read_csv, write_csv

from .accounting import main_eligible
from .corrections import clean_seconds, corrected
from .work import primary_work


def tables(rows, include_pilots=False):
    partitions = dict(measured=[], historical_estimates=[], unknown=[], excluded=[])
    seen = set()
    for row in rows:
        config = json.loads(row["config"])
        # Include input location in callers' diagnostics, but never pool duplicate runs.
        key = row.get("run_id") or identity(config)
        if key in seen:
            raise ValueError(f"duplicate run identity across input directories: {key}")
        seen.add(key)
        if not main_eligible(row) or (
            not include_pilots
            and config.get("phase") in ("pilot", "smoke", "profile", "screen")
        ):
            partitions["excluded"].append(
                dict(run_id=key, reason="pilot_or_ineligible_setup")
            )
            continue
        work = primary_work(row, config)
        fix = corrected(row)
        entry = dict(
            run_id=key,
            config=json.dumps(config, sort_keys=True),
            family=config["family"],
            dataset=config.get("dataset", ""),
            orientation=config.get("orientation", ""),
            inner=config["inner"],
            implementation=config.get("implementation", "legacy"),
            phase=config.get("phase", ""),
            hardware=fix["node_family"],
            status=row.get("status", ""),
            converged=str(row.get("converged")) == "True",
            work_value=work["value"],
            work_unit=work["unit"],
            work_scope=work["scope"],
            work_evidence=work["evidence"],
            clean_seconds=clean_seconds(row),
            hessian_products_diagnostic=fix["hessian_products_corrected"],
            hessian_diagnostic_basis=fix["work_count_basis"],
            design_forward_products=fix["design_forward_products"],
            design_transpose_products=fix["design_transpose_products"],
        )
        if work["evidence"] == "measured":
            partitions["measured"].append(entry)
        elif config["family"] == "qp" and math.isfinite(
            fix["hessian_products_solve_phase"]
        ):
            # Historical estimates have a different scope and their own table.
            entry.update(
                work_value=fix["hessian_products_solve_phase"],
                work_scope=fix["solve_phase_basis"],
                work_evidence="historical_estimate",
            )
            partitions["historical_estimates"].append(entry)
        else:
            partitions["unknown"].append(entry)
    return partitions


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        config = json.loads(row["config"])
        config.pop("seed", None)
        # Other parameters, hardware policy, evidence, units, and scope stay distinct.
        key = (
            json.dumps(config, sort_keys=True),
            row["work_evidence"],
            row["work_unit"],
            row["work_scope"],
            row["hardware"],
        )
        groups[key].append(row)
    summaries = []
    for (config, evidence, unit, scope, hardware), group in groups.items():
        converged = [r for r in group if r["converged"]]
        values = [r["work_value"] for r in converged if math.isfinite(r["work_value"])]
        times = [
            r["clean_seconds"] for r in converged if math.isfinite(r["clean_seconds"])
        ]
        summaries.append(
            dict(
                config=config,
                evidence=evidence,
                work_unit=unit,
                work_scope=scope,
                hardware=hardware,
                runs=len(group),
                converged_runs=len(converged),
                measured_or_estimated_work_runs=len(values),
                median_work_to_convergence=float(np.median(values))
                if values
                else float("nan"),
                median_clean_seconds=float(np.median(times)) if times else float("nan"),
            )
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-pilots", action="store_true")
    args = parser.parse_args()
    rows = [
        row
        for d in args.results_dir
        for p in sorted(d.glob("*/result.csv"))
        for row in read_csv(p)
    ]
    if not rows:
        parser.error("no result CSVs found")
    partitions = tables(rows, args.include_pilots)
    for name, entries in partitions.items():
        # Write even empty partitions, so a rerun cannot leave stale table contents.
        write_csv(args.output / f"{name}_work.csv", entries)
        if name != "excluded":
            write_csv(args.output / f"{name}_summary.csv", summarize(entries))
    print({name: len(entries) for name, entries in partitions.items()})


if __name__ == "__main__":
    main()
