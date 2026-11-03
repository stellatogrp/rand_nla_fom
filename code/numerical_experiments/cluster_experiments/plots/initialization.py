"""Pair initialization arms only when all other settings actually match."""

import json
from collections import defaultdict

from experiments.initialization import BASELINES
from experiments.io import identity

from .accounting import finite_time, main_eligible
from .corrections import clean_seconds


def measure(row, name):
    if name == "clean_seconds":
        return clean_seconds(row)
    return float(row.get(name, "nan"))


def comparisons(rows):
    groups = defaultdict(dict)
    for row in rows:
        if not main_eligible(row) or not finite_time(row):
            continue
        config = json.loads(row["config"])
        if config["inner"] in BASELINES:
            continue
        config = dict(config)
        mode = config.pop("inner_initialization", "warm")
        # Runtime-resolved optimal recommendations can differ between arms.
        # Do not call those paired unless their actual parameters also match.
        config["resolved_sigma"] = str(row.get("sigma", config.get("sigma", "")))
        config["resolved_stride"] = str(row.get("stride", config.get("stride", "")))
        groups[identity(config)][mode] = row
    output = []
    for pair_id, arms in sorted(groups.items()):
        warm = arms.get("warm")
        for mode in ("fixed", "gaussian"):
            if mode not in arms:
                continue
            row = arms[mode]
            valid = (
                warm is not None
                and warm.get("converged") == "True"
                and row.get("converged") == "True"
            )
            result = dict(
                pair_id=pair_id,
                inner_initialization=mode,
                run_id=row.get("run_id", ""),
                warm_run_id=warm.get("run_id", "") if warm else "",
                status=row.get("status", ""),
                warm_status=warm.get("status", "") if warm else "missing",
                both_converged=valid,
                config=row["config"],
            )
            for name in (
                "clean_seconds",
                "total_seconds",
                "solve_seconds",
                "total_inner",
                "checks",
            ):
                numerator = measure(row, name)
                denominator = measure(warm, name) if warm else float("nan")
                result[f"{name}_ratio_to_warm"] = (
                    numerator / denominator
                    if valid and denominator > 0
                    else float("nan")
                )
            output.append(result)
    return output
