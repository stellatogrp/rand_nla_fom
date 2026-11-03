"""Audit bounded pilot results without modifying original result CSVs."""

import argparse
import json
import math
from pathlib import Path

from releases.accounting_v6.experiments.io import read_csv, write_csv
from releases.accounting_v6.experiments.outcomes import is_method_outcome


def audit(manifest, results, task_id=None):
    checks = []
    rows = read_csv(manifest)
    for task in rows if task_id is None else [rows[task_id]]:
        config = json.loads(task["config"])
        records = read_csv(results / task["run_id"] / "result.csv")
        failures = []
        row = records[0] if records else {}
        if not records:
            failures.append("missing result")
        elif not is_method_outcome(row):
            failures.append("unresolved/infrastructure result")
        if records:

            def number(name, record=row):
                return float(record.get(name, "nan") or "nan")

            if (
                row.get("run_id") != task["run_id"]
                or json.loads(row["config"]) != config
            ):
                failures.append("configuration identity mismatch")
            if row.get("setup_policy") != "independent":
                failures.append("setup must be independently charged")
            if row.get("setup_accounting_complete") != "True":
                failures.append("incomplete setup accounting")
            for name in [
                "generation_seconds",
                "solver_setup_seconds",
                "solve_seconds",
                "total_seconds",
                "component_seconds",
            ]:
                if not math.isfinite(number(name)) or number(name) < 0:
                    failures.append(f"invalid {name}")
            if number("component_seconds") > number("total_seconds") + 1:
                failures.append("components exceed measured worker time")
            if row.get("work_count_complete") != "True":
                failures.append("missing measured work scope")
            if row.get("converged") == "True" and any(
                not math.isfinite(number(key)) or number(key) > config["tolerance"]
                for key in ("stationarity", "feasibility")
            ):
                failures.append("false convergence certificate")
            if (
                config["inner"] != "pdhg"
                and number("work_units") > config["max_inner"] + 1e-6
            ):
                failures.append("inner work cap exceeded")
            if config["family"] == "qp":
                if number("L") / number("mu") < 1e4:
                    failures.append("QP condition number below required threshold")
                if config["inner"] != "pdhg":
                    remainder = (
                        number("hessian_products_total")
                        - number("calibration_hessian_products")
                        - number("inner_hessian_products")
                        - number("check_products")
                    )
                    if remainder not in (0, 1):
                        failures.append("inconsistent QP operator count decomposition")
            else:
                for name in ["design_forward_products", "design_transpose_products"]:
                    if not math.isfinite(number(name)) or number(name) <= 0:
                        failures.append(f"missing {name}")
            if config["inner"] == "block_kaczmarz" and row.get("converged") == "True":
                failures.append(
                    "capped Kaczmarz case unexpectedly converged; investigate"
                )
        checks.append(
            dict(
                task_id=task["task_id"],
                run_id=task["run_id"],
                family=config["family"],
                inner=config["inner"],
                orientation=config["orientation"],
                status=row.get("status", "missing"),
                passed=not failures,
                failures="; ".join(failures),
            )
        )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = audit(args.manifest, args.results_dir)
    write_csv(args.output, checks)
    print(f"{sum(c['passed'] for c in checks)}/{len(checks)} pilot audits passed")
    if not all(c["passed"] for c in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
