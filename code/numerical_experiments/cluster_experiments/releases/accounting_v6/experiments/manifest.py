"""Generate staged CSV manifests; never launch the full grid automatically."""

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .initialization import BASELINES, MODES
from .initialization import settings as initialization_settings
from .initialization import variants as initialization_variants
from .io import identity, read_csv, write_csv
from .limits import budgets
from .matrices import read_specs
from .outcomes import blocked, is_method_outcome

# Nystrom-PCG has no verified contraction bound in v4 (see tuning.constants).
MODEL_SOLVERS = {"gd", "cg", "block_kaczmarz"}
SELECTION_SEEDS = [0, 1, 2]
EVALUATION_SEEDS = [3, 4, 5]
BUDGET_KEYS = [
    "time_limit",
    "max_inner",
    "max_outer",
    "max_inner_per_outer",
    "inner_time_limit",
    "tolerance",
    "check_source",
    "budget_unit",
    "baseline_tolerance_rungs",
]


def instances(smoke=False, pilot=False, seeds=(0, 1, 2), datasets=None):
    catalog = datasets
    for seed in seeds:
        for side in [4] if smoke else [40] if pilot else [40, 64, 100]:
            yield dict(
                family="qp",
                dataset="synthetic",
                orientation="native",
                side=side,
                seed=seed,
                kappa=1e4,
            )
        datasets = (
            list(catalog)
            if catalog is not None
            else (
                ["smoke"] if smoke else ["GL7d12"] if pilot else ["GL7d16", "lp_osa_30"]
            )
        )
        for family, dataset, orientation in itertools.product(
            ["lasso", "logistic"], datasets, ["native", "transpose"]
        ):
            if catalog is not None and orientation not in catalog[dataset].get(
                "orientations", ["native", "transpose"]
            ):
                continue
            instance = dict(
                family=family, dataset=dataset, orientation=orientation, seed=seed
            )
            if catalog is not None:
                instance["matrix_source"] = catalog[dataset]
            yield instance


def variants(family, smoke=False):
    if family == "logistic":
        yield dict(inner="gd")
        return
    for inner in (
        ["gd", "cg", "nystrom_pcg"]
        if family == "lasso"
        else [
            "gd",
            "agd",
            "cg",
            "gmres",
            "nystrom_pcg",
            "block_kaczmarz",
            "cyclic_kaczmarz",
        ]
    ):
        if inner == "nystrom_pcg":
            for rank in [4] if smoke else [32, 128]:
                yield dict(inner=inner, rank=rank)
        elif "kaczmarz" in inner:
            for block in [1, 4] if smoke else [1, 8, 32, 128, 512]:
                yield dict(inner=inner, block_size=block)
        else:
            yield dict(inner=inner)


def strides(instance, variant, smoke=False):
    """Fixed strides, except Kaczmarz: fractions of one pass over the blocks."""
    if smoke:
        return [1, 2]
    if "kaczmarz" not in variant["inner"] or "side" not in instance:
        return [1, 2, 5, 10, 20]
    blocks = math.ceil(instance["side"] ** 3 / variant["block_size"])
    return sorted(
        {1} | {max(1, round(f * blocks)) for f in (0.01, 0.05, 0.1, 0.5, 1.0)}
    )


def tuning_grid(instance, smoke=False):
    for variant in variants(instance["family"], smoke):
        base = dict(**instance, **variant, ratio=10.0)
        for sigma in [0.2] if smoke else [0.01, 0.05, 0.2, 0.5, 0.9, 1.0]:
            yield dict(base, sigma=sigma, stride=1, sweep="sigma")
        for stride in strides(instance, variant, smoke):
            yield dict(base, sigma=0.1, stride=stride, sweep="stride")
        if variant["inner"] in MODEL_SOLVERS:
            yield dict(base, sigma="optimal", stride=1, sweep="sigma")
            yield dict(base, sigma=0.1, stride="optimal", sweep="stride")
        yield dict(base, sigma=0.0, stride=1, near_exact=True, sweep="near_exact")
    for baseline in ["pdhg", "osqp", "scs"]:
        yield dict(
            **instance,
            inner=baseline,
            sweep="baseline",
            baseline_step=0.1 if baseline != "pdhg" else 1.0,
        )


def group_key(config):
    return tuple(
        (
            k,
            json.dumps(config[k], sort_keys=True)
            if isinstance(config[k], dict)
            else config[k],
        )
        for k in [
            "family",
            "dataset",
            "matrix_source",
            "orientation",
            "side",
            "inner",
            "rank",
            "block_size",
            "inner_initialization",
            "gaussian_scale",
            "fixed_initial_value",
            "implementation",
            "blas_threads",
            *BUDGET_KEYS,
        ]
        if k in config
    )


def clean_seconds(row):
    """Wall time of one clean, from-scratch attempt; NaN when none exists.

    accounting_v4 records it directly. accounting_v3 attempts also restart
    from scratch, so the final attempt's wall time is clean even when an
    earlier attempt was lost. Older releases resumed from checkpoints, which
    shows up as solve plus loading time exceeding the in-process wall time.
    """
    if row.get("clean_attempt_seconds") not in (None, ""):
        return float(row["clean_attempt_seconds"])
    if row.get("attempt_worker_wall_seconds") not in (None, ""):
        return float(row["attempt_worker_wall_seconds"])
    try:
        resumed = (
            float(row["solve_seconds"]) + float(row["generation_seconds"])
            > float(row["process_seconds"]) + 1.0
        )
    except (KeyError, ValueError):
        return float("nan")
    return float("nan") if resumed else float(row.get("total_seconds", "nan"))


def winners(directories, warm_only=False, freeze_parameters=False):
    """Penalize failed/censored seeds; require every replicate to converge.

    Ranking uses clean single-attempt wall time. Infrastructure events are
    never silently treated as method failures: selection stops and lists
    the runs to rerun.
    """
    if isinstance(directories, (str, Path)):
        directories = [directories]
    groups = defaultdict(lambda: defaultdict(list))
    rerun = {}
    paths = [
        path
        for directory in directories
        for path in sorted(Path(directory).glob("*/result.csv"))
    ]
    for path in paths:
        for row in read_csv(path):
            if row.get("setup_policy", "independent") != "independent" or (
                row.get("timing_scope", "") == "incremental_pilot_after_shared_loading"
            ):
                continue
            if "config" not in row:
                continue
            config = json.loads(row["config"])
            if warm_only and (
                config["inner"] in BASELINES
                or config.get("inner_initialization", "warm") != "warm"
            ):
                continue
            if row.get("status") == "not_applicable":
                continue
            if config.get("sweep") == "near_exact":
                continue
            if not is_method_outcome(row) or (
                row.get("converged") == "True" and not math.isfinite(clean_seconds(row))
            ):
                # A converged run without a clean timing is listed as a rerun.
                rerun[row.get("run_id", str(path.parent.name))] = (
                    row if not is_method_outcome(row) else dict(status="no_clean_time")
                )
                continue
            parameter = (
                config.get("sigma", ""),
                config.get("stride", ""),
                config.get("ratio", ""),
                config.get("baseline_step", ""),
                config.get("gd_factor", 1.0),
            )
            groups[group_key(config)][parameter].append((row, config))
    if rerun:
        raise ValueError(blocked(rerun))
    selected = []
    for candidates in groups.values():
        expected_seeds = {
            cfg["seed"] for runs in candidates.values() for _, cfg in runs
        }
        eligible = []
        for runs in candidates.values():
            if {cfg["seed"] for _, cfg in runs} == expected_seeds and all(
                row.get("converged") == "True" for row, _ in runs
            ):
                cost = float(
                    np.exp(
                        np.mean(
                            [np.log(max(clean_seconds(row), 1e-12)) for row, _ in runs]
                        )
                    )
                )
                eligible.append((cost, runs))
        if not eligible:
            # Every run here ended in a method outcome (infrastructure events
            # raised above), so "never converged" is itself the result, e.g.
            # a direct factorization that exceeds the memory or time budget.
            # It must not block selection for the other solver groups.
            config = next(iter(candidates.values()))[0][1]
            print(
                "no configuration converged for every seed; reported, not "
                f"selected: {config['inner']} {dict(group_key(config))}",
                file=sys.stderr,
            )
            continue
        _, runs = min(eligible, key=lambda pair: pair[0])
        row, config = runs[0]
        config = dict(config)
        if freeze_parameters:
            config.update(sigma=float(row["sigma"]), stride=int(row["stride"]))
            config["reference_run_id"] = row.get("run_id", "")
        selected.append(config)
    if not selected:
        raise ValueError("no finished results found")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=["smoke", "pilot", "tune", "steps", "initialization", "evaluate"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-results", type=Path, nargs="+")
    parser.add_argument("--from-manifest", type=Path, nargs="+")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="default: 0 1 2 for selection phases, 3 4 5 for evaluate",
    )
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--max-inner", type=int, default=1_000_000)
    parser.add_argument("--max-outer", type=int, default=50_000)
    parser.add_argument("--max-inner-per-outer", type=int, default=100_000)
    parser.add_argument("--inner-time-limit", type=float, default=300.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--blas-threads", type=int)
    parser.add_argument("--implementation", default="accounting_v6")
    parser.add_argument(
        "--check-source",
        choices=["recompute", "recurrence"],
        default="recurrence",
        help="CG/PCG stopping test from the maintained residual (v4 default)",
    )
    parser.add_argument(
        "--budget-unit",
        choices=["steps", "work"],
        default="work",
        help="count Kaczmarz updates as block/n of a Hessian product",
    )
    parser.add_argument("--baseline-tolerance-rungs", type=int, default=3)
    parser.add_argument("--thread-policy", type=Path)
    parser.add_argument(
        "--setup-policy", choices=["independent", "shared_pilot"], default="independent"
    )
    parser.add_argument("--dense-storage", choices=["streamed_dense", "resident_dense"])
    parser.add_argument("--max-resident-gb", type=float, default=32.0)
    parser.add_argument("--families", nargs="+", choices=["qp", "lasso", "logistic"])
    parser.add_argument("--solvers", nargs="+")
    parser.add_argument(
        "--inner-initializations",
        nargs="+",
        choices=MODES,
        help="default: all three for new sweeps; inherited in steps",
    )
    parser.add_argument("--gaussian-scale", type=float)
    parser.add_argument("--fixed-initial-value", type=float)
    parser.add_argument("--initialization-seed", type=int)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset-spec",
        type=Path,
        help="JSON name-to-source mapping; embedded in every task",
    )
    source.add_argument(
        "--dataset-preset", choices=["random", "suitesparse"], default="random"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="base directory for relative input paths",
    )
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0]
    )
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = EVALUATION_SEEDS if args.phase == "evaluate" else SELECTION_SEEDS
    if args.baseline_tolerance_rungs < 1:
        parser.error("baseline-tolerance-rungs must be positive")
    thread_policy = (
        json.loads(args.thread_policy.read_text()) if args.thread_policy else None
    )
    if args.setup_policy == "shared_pilot" and args.phase not in ("pilot", "smoke"):
        parser.error("shared-data timing is restricted to pilot/smoke manifests")
    inherited = args.phase in ("steps", "initialization", "evaluate")
    if inherited and args.dataset_spec:
        parser.error("this phase inherits matrix sources; do not replace them")
    datasets = None
    if not inherited:
        if args.dataset_spec:
            datasets = read_specs(args.dataset_spec, args.data_dir)
            if args.dense_storage:
                for spec in datasets.values():
                    if spec.get("format") != "npy":
                        parser.error("dense-storage requires NPY dataset sources")
                    spec.update(
                        storage=args.dense_storage, max_resident_gb=args.max_resident_gb
                    )
        elif args.dataset_preset == "random":
            datasets = read_specs(
                Path(__file__).resolve().parent.parent / "datasets/random.json",
                args.data_dir,
            )
            if args.phase in ("pilot", "smoke"):
                rows, cols = (6000, 2000) if args.phase == "pilot" else (48, 24)
                datasets = {
                    f"{name}_{args.phase}": dict(
                        spec, rows=rows, cols=cols, entries_per_row=min(20, cols)
                    )
                    for name, spec in datasets.items()
                }
    configs = []
    if inherited:
        if not args.from_results:
            parser.error("this phase requires --from-results from a completed sweep")
        if not args.from_manifest:
            parser.error("this phase requires --from-manifest to verify completeness")
        expected = {
            row["run_id"] for path in args.from_manifest for row in read_csv(path)
        }
        observed = {
            path.parent.name
            for directory in args.from_results
            for path in directory.glob("*/result.csv")
        }
        if not expected or expected - observed:
            parser.error(
                f"tuning is incomplete: {len(expected - observed)} tasks missing"
            )
        if args.phase == "evaluate":
            used = {
                json.loads(row["config"])["seed"]
                for path in args.from_manifest
                for row in read_csv(path)
            }
            if used & set(args.seeds):
                parser.error(
                    f"evaluation seeds {sorted(used & set(args.seeds))} were "
                    "used for selection; choose held-out seeds"
                )
        try:
            selected = winners(
                args.from_results,
                warm_only=args.phase == "initialization",
                freeze_parameters=args.phase == "initialization",
            )
        except ValueError as error:
            parser.error(str(error))
        for winner in selected:
            if args.phase == "evaluate":
                for seed in args.seeds:
                    configs.append(
                        dict(winner, seed=seed, phase="evaluate", sweep="evaluate")
                    )
                continue
            if args.phase == "initialization":
                for seed in args.seeds:
                    base = dict(
                        winner,
                        seed=seed,
                        phase="initialization",
                        sweep="initialization",
                    )
                    base.pop("inner_initialization", None)
                    configs.append(base)
                continue
            if winner["inner"] in ("pdhg", "osqp", "scs"):
                for step, seed in itertools.product(
                    [0.01, 0.1, 1.0, 10.0, 100.0], args.seeds
                ):
                    configs.append(
                        dict(
                            winner,
                            baseline_step=step,
                            seed=seed,
                            sweep="steps",
                            phase="steps",
                        )
                    )
                continue
            # c=sqrt(mu*L) minimizes the worst-case reflected-prox contraction
            # for strongly convex quadratic f. It is not a wall-time oracle.
            ratios = list(args.ratios)
            if winner["family"] == "qp":
                ratios.append(float(np.sqrt(winner["kappa"])))
            for ratio, seed, gd_factor in itertools.product(
                sorted(set(ratios)),
                args.seeds,
                [0.5, 1.0] if winner["inner"] == "gd" else [1.0],
            ):
                config = dict(
                    winner,
                    ratio=ratio,
                    seed=seed,
                    gd_factor=gd_factor,
                    sweep="steps",
                    phase="steps",
                )
                configs.append(config)
                if winner["inner"] in MODEL_SOLVERS:
                    configs.append(dict(config, sigma="optimal", stride=1))
                    configs.append(dict(config, sigma=0.1, stride="optimal"))
    else:
        for instance in instances(
            args.phase == "smoke", args.phase == "pilot", args.seeds, datasets
        ):
            for config in tuning_grid(instance, args.phase == "smoke"):
                if args.phase == "pilot" and config.get("sweep") != "baseline":
                    if config["sweep"] != "sigma" or config["sigma"] != 0.2:
                        continue
                configs.append(dict(config, phase=args.phase))
    rows = []
    seen = set()
    expanded = []
    for config in configs:
        if "inner_initialization" in config:
            if (
                args.inner_initializations
                and config["inner_initialization"] not in args.inner_initializations
            ):
                continue
            expanded.append(config)
        else:
            expanded.extend(
                initialization_variants(
                    config,
                    args.inner_initializations or MODES,
                    args.gaussian_scale
                    if args.gaussian_scale is not None
                    else config.get("gaussian_scale", 1.0),
                    args.fixed_initial_value
                    if args.fixed_initial_value is not None
                    else config.get("fixed_initial_value", 0.0),
                )
            )
    for config in expanded:
        if args.families and config["family"] not in args.families:
            continue
        if args.solvers and config["inner"] not in args.solvers:
            continue
        if config["inner"] not in BASELINES:
            config["initialization_seed"] = (
                args.initialization_seed
                if args.initialization_seed is not None
                else config["seed"] + 200003
            )
            if args.gaussian_scale is not None:
                config["gaussian_scale"] = args.gaussian_scale
            if args.fixed_initial_value is not None:
                config["fixed_initial_value"] = args.fixed_initial_value
            try:
                initialization_settings(config)
            except ValueError as error:
                parser.error(str(error))
        if config["inner"] in BASELINES:
            if config["inner"] in ("osqp", "scs"):
                config["baseline_tolerance_rungs"] = args.baseline_tolerance_rungs
        else:
            config.update(check_source=args.check_source, budget_unit=args.budget_unit)
        config.update(
            implementation=args.implementation,
            setup_policy=args.setup_policy,
            blas_threads=(
                args.blas_threads
                if args.blas_threads is not None
                else config.get("blas_threads", 1)
            ),
            time_limit=args.time_limit,
            max_inner=args.max_inner,
            max_outer=args.max_outer,
            max_inner_per_outer=args.max_inner_per_outer,
            inner_time_limit=args.inner_time_limit,
            tolerance=args.tolerance,
        )
        budgets(config)
        if thread_policy:
            from .core_policy import apply

            config = apply(config, thread_policy)
        key = identity(config)
        if key not in seen:
            rows.append(
                dict(
                    task_id=len(rows),
                    run_id=key,
                    config=json.dumps(config, sort_keys=True),
                )
            )
            seen.add(key)
    write_csv(args.output, rows)
    print(f"{len(rows)} tasks -> {args.output}; array 0-{len(rows) - 1}%4")
    if args.shard_size < 1:
        parser.error(
            "shard-size must be positive and below the discovered MaxArraySize"
        )
    if len(rows) > args.shard_size:
        for part, start in enumerate(range(0, len(rows), args.shard_size)):
            subset = [
                dict(row, task_id=i)
                for i, row in enumerate(rows[start : start + args.shard_size])
            ]
            name = args.output.with_name(f"{args.output.stem}.part{part:03}.csv")
            write_csv(name, subset)
            print(f"  {name}: array 0-{len(subset) - 1}%4")


if __name__ == "__main__":
    main()
