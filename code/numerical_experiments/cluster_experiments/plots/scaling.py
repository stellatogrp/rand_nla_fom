"""Summarize measured kernel scaling; recommend cores only with good utilization."""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.io import read_csv, write_csv


def summarize(rows, min_speedup=1.2, min_efficiency=0.9, min_repeats=2):
    groups = defaultdict(lambda: defaultdict(list))
    keys = ["kind", "dataset", "family", "storage", "operation", "inner", "cpu_model"]
    for row in rows:
        if row.get("benchmark_status") != "completed":
            continue
        if row.get("operation") == "solve" and row.get("converged") != "True":
            continue
        key = tuple(row.get(k, "") for k in keys)
        groups[key][int(row["threads"])].append(row)
    output = []
    for key, candidates in groups.items():
        usable = {n: rs for n, rs in candidates.items() if len(rs) >= min_repeats}
        if 1 not in usable:
            continue
        reference = statistics.median(float(r["seconds"]) for r in usable[1])
        measured = {}
        for count, rs in usable.items():
            seconds = statistics.median(float(r["seconds"]) for r in rs)
            efficiency = statistics.median(float(r["cpu_efficiency"]) for r in rs)
            measured[count] = dict(
                seconds=seconds, speedup=reference / seconds, cpu_efficiency=efficiency
            )
        eligible = [
            n
            for n, m in measured.items()
            if n == 1
            or (m["speedup"] >= min_speedup and m["cpu_efficiency"] >= min_efficiency)
        ]
        chosen = min(eligible, key=lambda n: measured[n]["seconds"])
        for count, metrics in measured.items():
            output.append(
                dict(zip(keys, key, strict=True))
                | dict(
                    threads=count,
                    recommended_threads=chosen,
                    decision_status=(
                        "awaiting_multicore_measurements"
                        if len(measured) == 1
                        else "measured_candidates"
                    ),
                    recommendation_scope=(
                        "kernel_only"
                        if dict(zip(keys, key, strict=True))["operation"] == "gradient"
                        else "solve"
                    ),
                    cpu_seconds_per_repeat=count
                    * metrics["seconds"]
                    * metrics["cpu_efficiency"],
                    **metrics,
                )
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        row
        for path in args.results_dir.glob("**/*.csv")
        if path.resolve() != args.output.resolve()
        for row in read_csv(path)
    ]
    result = summarize(rows)
    if not result:
        parser.error(
            "no matched completed repeated benchmarks with a one-core reference"
        )
    write_csv(args.output, result)
    for row in result:
        print(
            f"{row['dataset']} {row['storage']}: {row['threads']} cores, "
            f"speedup={row['speedup']:.2f}, efficiency={row['cpu_efficiency']:.1%}; "
            f"recommend {row['recommended_threads']}"
        )


if __name__ == "__main__":
    main()
