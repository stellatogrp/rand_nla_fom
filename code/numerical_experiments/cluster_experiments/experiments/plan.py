"""Split QP tuning into a fast lane and a bounded Kaczmarz screening study."""

import argparse
import json
from pathlib import Path

from .io import identity, read_csv, write_csv
from .limits import budgets


def split(configs, screen_seconds=120.0):
    fast, screen = [], []
    for original in configs:
        config = dict(original, implementation="speed_v2", blas_threads=1)
        if "kaczmarz" not in config["inner"]:
            fast.append(config)
        elif (
            config["side"] == 40
            and config["seed"] == 0
            and config.get("sigma") == 0.2
            and config.get("stride") == 1
            and config.get("sweep") == "sigma"
        ):
            screen.append(
                dict(
                    config,
                    phase="screen",
                    sweep="kaczmarz_screen",
                    time_limit=screen_seconds,
                    inner_time_limit=screen_seconds,
                    max_inner=100_000,
                    max_inner_per_outer=100_000,
                )
            )
    return fast, screen


def emit(path, configs):
    rows, seen = [], set()
    for config in configs:
        budgets(config)
        key = identity(config)
        if key in seen:
            continue
        rows.append(
            dict(
                task_id=len(rows), run_id=key, config=json.dumps(config, sort_keys=True)
            )
        )
        seen.add(key)
    if not rows:
        raise ValueError("no eligible configurations; no manifest emitted")
    write_csv(path, rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--screen-seconds", type=float, default=120.0)
    parser.add_argument(
        "--expand-from",
        type=Path,
        help="expand only block/method pairs with a converged screen arm",
    )
    args = parser.parse_args()
    configs = [json.loads(row["config"]) for row in read_csv(args.from_manifest)]
    fast, screen = split(configs, args.screen_seconds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.expand_from:
        expected = {identity(c) for c in screen}
        results = {}
        for path in args.expand_from.glob("*/result.csv"):
            for row in read_csv(path):
                if row.get("run_id") in expected:
                    results[row["run_id"]] = row
        if expected - results.keys():
            parser.error("Kaczmarz screening is incomplete")
        eligible = {
            (c["inner"], c["block_size"])
            for c in screen
            if results[identity(c)].get("converged") == "True"
        }
        selected = [
            dict(c, implementation="speed_v2", blas_threads=1)
            for c in configs
            if (c["inner"], c.get("block_size")) in eligible
        ]
        if not selected:
            print("No screening arm converged; retain the censored screening results.")
            return
        rows = emit(args.output_dir / "kaczmarz_expanded.csv", selected)
        print(f"{len(rows)} expanded tasks; pairs={sorted(eligible)}")
        return
    for name, subset in [("fast", fast), ("kaczmarz_screen", screen)]:
        rows = emit(args.output_dir / f"{name}.csv", subset)
        print(f"{name}: {len(rows)} tasks")


if __name__ == "__main__":
    main()
