"""Split QP tuning into a fast lane and a bounded Kaczmarz screening study.

accounting_v5 budgets count work: a block update is block/n of a Hessian
product. The speed_v2 screen counted steps, so one-row blocks stopped after
about 1.6 passes over the rows (status max_inner), not on time.
"""

import argparse
import json
from pathlib import Path

from .io import identity, read_csv, write_csv
from .limits import budgets
from .manifest import MODEL_SOLVERS, strides
from .outcomes import is_method_outcome

RELEASE_FIELDS = dict(
    implementation="accounting_v5",
    blas_threads=1,
    check_source="recurrence",
    budget_unit="work",
)


def upgrade(configs):
    """Migrate legacy grids without carrying unsupported model prescriptions.

    Regenerate each Kaczmarz stride sweep from its instance/block size. Other
    sweep arms, initialization modes, seeds, and resource limits are retained.
    """
    seen = set()
    for original in configs:
        if original["inner"] not in MODEL_SOLVERS and "optimal" in (
            original.get("sigma"),
            original.get("stride"),
        ):
            continue
        config = dict(original, **RELEASE_FIELDS)
        grid = (
            strides(config, config)
            if "kaczmarz" in config["inner"]
            and config.get("sweep") == "stride"
            and config.get("stride") != "optimal"
            else [config.get("stride")]
        )
        for stride in grid:
            candidate = dict(config)
            if stride is not None:
                candidate["stride"] = stride
            key = identity(candidate)
            if key not in seen:
                seen.add(key)
                yield candidate


def split(configs, screen_seconds=120.0):
    fast, screen = [], []
    for config in upgrade(configs):
        if "kaczmarz" not in config["inner"]:
            fast.append(config)
        elif (
            config["side"] == 40
            and config["seed"] == 0
            and config.get("sigma") == 0.2
            and config.get("stride") == 1
            and config.get("sweep") == "sigma"
        ):
            for stride in strides(config, config):
                screen.append(
                    dict(
                        config,
                        stride=stride,
                        phase="screen",
                        sweep="kaczmarz_screen",
                        time_limit=screen_seconds,
                        inner_time_limit=screen_seconds,
                        max_inner=100_000,
                        max_inner_per_outer=100_000,
                    )
                )
    return fast, screen


def expand(configs, screen, results):
    """Expand successful method/block pairs using the same upgraded grid."""
    expected = {identity(c) for c in screen}
    if expected - results.keys():
        raise ValueError("Kaczmarz screening is incomplete")
    # A crashed or interrupted arm is missing evidence, not nonconvergence.
    if any(not is_method_outcome(results[key]) for key in expected):
        raise ValueError("Kaczmarz screening has infrastructure events; rerun them")
    eligible = {
        (c["inner"], c["block_size"])
        for c in screen
        if str(results[identity(c)].get("converged")) == "True"
    }
    return [
        c for c in upgrade(configs) if (c["inner"], c.get("block_size")) in eligible
    ], eligible


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
        try:
            selected, eligible = expand(configs, screen, results)
        except ValueError as error:
            parser.error(str(error))
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
