"""Re-time every configuration of every inner solver on one node, interleaved.

    python -m bench.one_node plan --campaign studies_v2 --benchmark bench_v1 \\
        --output bench_v1/plan.json --instances 5 --near-exact qp_side100=3
    python -m bench.one_node seal bench_v1
    python -m bench.one_node run --plan bench_v1/plan.json \\
        --results-dir results/bench_v1 --problem qp_side40       # inside Slurm
    python -m bench.one_node summarize --plan bench_v1/plan.json \\
        --results-dir results/bench_v1

``plan`` fixes, for every ADMM problem of a collected campaign, the instances
to time (the first ``--instances``) and, for every inner solver, every
inexact configuration of the campaign (both sweeps), labelled ``fixed``
(``sigma = 0.1`` and ``s = 1``, the same for every solver), ``sigma_star``
(Proposition 5.1), and ``campaign_best`` (the smallest median cost in the
campaign, for reference), with the near-exact baseline on the first instances
of each problem and exact ADMM (``exact``) once per instance.  Every
configuration is timed because the campaign's timings, taken on shared nodes
under different loads, cannot be trusted to rank the near-ties among them.

``run`` times them in one process.  Instance by instance it rebuilds every
solver's setup from the campaign's stored one, read only (``L``, the ground
truth, and the paper's parameters are loaded; the exact solver and the inner
solver's own setup, such as the Nystrom sketch, are rebuilt and timed here),
then runs the instance's configurations of all the solvers in a seeded random
order, so that a change of load on the node falls on every solver alike.
Each run is appended to ``<results>/<problem>/measurements.csv`` when it
finishes, and a rerun skips the runs already there.

``summarize`` compares the solvers instance by instance, each at its fastest
configuration as timed here and all at the ``fixed`` one, against the fastest
solver on the same instance, gives the near-exact baseline's cost over the fastest
configuration's, reports the spread of the exact-ADMM times as a probe of the
node's load, and checks that every run repeats the campaign's outer and inner
iteration counts, as the computations are the same.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from releases.regression_v9.experiments.runtime import cpu_model
from studies import ENGINE
from studies import campaign as source
from studies._io import read_rows, write_rows
from studies.study import (
    PARTS,
    SETUP,
    _provenance,
    build_parser,
    prepare_instance,
    run_configuration,
    settings_from,
    study_directory,
)

ROOT = source.ROOT
SCRIPT = "slurm/bench_one_node.slurm"
MEASUREMENTS = "measurements.csv"
SUMMARY = "summary.csv"
ORDER_SEED = 20_261_108
FIELDS = (
    "problem",
    "study",
    "solver",
    "seed",
    "round",
    "position",
    "kinds",
    "method",
    "sweep",
    "sigma",
    "b",
    "started_at",
    "runtime_seconds",
    "own_setup_seconds",
    "prepare_seconds",
    "outer_iterations",
    "total_inner",
    "converged",
    "stopped_by",
    "error",
    "hostname",
    "cpu_model",
    "slurm_job_id",
)
nan = float("nan")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slug(problem: str) -> str:
    """``qp side 40`` -> ``qp_side40``; ``lasso random_normal`` unchanged but joined."""
    words = problem.split()
    if words[:2] == ["qp", "side"]:
        return f"qp_side{words[2]}"
    return "_".join(words)


def _sigma(value) -> float:
    if value is None or value == "":
        return nan
    return float(value)


def _same_sigma(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= 1e-9 * max(1.0, abs(a))


def _sigma_text(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.12g}"


# ---------------------------------------------------------------- the plan


def make_plan(
    campaign: Path,
    benchmark: str,
    near_exact: dict[str, int],
    near_exact_default: int,
    rounds: int = 1,
    instances: int | None = None,
) -> dict:
    """The configurations to time, chosen from a collected campaign's tables."""
    # Local only: the plan is made where the campaign's results are, and
    # ``plots`` is not synced to the cluster.
    from plots import solver_comparison as tables
    from studies.aggregate import config_key

    campaign = Path(campaign)
    seal_path = (ROOT / campaign) / "seal.json"
    seal = json.loads(seal_path.read_text())
    results = ROOT / seal["results_dir"]
    problems: dict[str, dict] = {}
    for entry in sorted(seal["studies"], key=lambda s: s["name"]):
        directory = results / entry["name"]
        identity = tables._identity(directory)
        if identity["algorithm"] != "admm":
            continue
        table = tables.configurations(directory)
        chosen = tables.pick(table)
        finished = [
            key
            for key, row in table.items()
            if key[0] == "inexact" and row["n_converged"] == row["n_runs"]
        ]
        ranked = sorted(finished, key=lambda key: table[key]["cost"])
        listed = read_rows(directory / "configurations.csv")
        index_of = {config_key(row): i for i, row in enumerate(listed)}
        marks = {
            "fixed": chosen.get("fixed"),
            "sigma_star": chosen.get("sigma_star"),
            "campaign_best": ranked[0] if ranked else None,
        }
        kinds: dict[tuple, list[str]] = {}
        for row in listed:
            key = config_key(row)
            if key[0] == "inexact":
                kinds[key] = [key[1]] + [m for m, k in marks.items() if k == key]

        def describe(
            key: tuple, labels: list[str], index_of=index_of, listed=listed
        ) -> dict:
            index = index_of[key]
            # The key's sigma is rounded; the plan keeps the listed value.
            sigma = float(listed[index]["sigma"])
            return dict(
                index=index,
                kinds="+".join(labels),
                method=key[0],
                sweep=key[1],
                sigma=None if math.isnan(sigma) else sigma,
                b=int(key[3]),
            )

        args = build_parser(entry["module"]).parse_args([f"@{ROOT / entry['args']}"])
        first = int(settings_from(args, entry["module"], pin=False)["seed"])
        name = tables.problem_of(identity)
        problem = problems.setdefault(
            name,
            dict(
                name=name,
                slug=slug(name),
                seeds=[
                    first + i
                    for i in range(min(int(entry["instances"]), instances or 10**9))
                ],
                studies=[],
            ),
        )
        problem["studies"].append(
            dict(
                name=entry["name"],
                module=entry["module"],
                args=entry["args"],
                solver=tables.solver_of(identity),
                configurations=[describe(k, v) for k, v in kinds.items()],
                near_exact=describe(chosen["near_exact"], ["near_exact"]),
                exact=describe(chosen["exact"], ["exact"]),
            )
        )
    for problem in problems.values():
        solvers = [s["solver"] for s in problem["studies"]]
        problem["exact_study"] = problem["studies"][
            solvers.index("cg") if "cg" in solvers else 0
        ]["name"]
        count = near_exact.get(problem["slug"], near_exact_default)
        problem["near_exact_seeds"] = problem["seeds"][:count]
    unknown = set(near_exact) - {p["slug"] for p in problems.values()}
    if unknown:
        raise SystemExit(f"no problem named {sorted(unknown)} in {campaign}")
    return dict(
        benchmark=benchmark,
        source_campaign=str(campaign),
        source_seal_sha256=_sha256(seal_path),
        source_results=seal["results_dir"],
        order_seed=ORDER_SEED,
        rounds=int(rounds),
        problems=sorted(problems.values(), key=lambda p: p["slug"]),
    )


# ---------------------------------------------------------------- the seal


def _sealed_files(plan_path: Path, plan: dict) -> list[Path]:
    return sorted(
        [
            *(ROOT / "bench").glob("*.py"),
            *(ROOT / "studies").glob("*.py"),
            ROOT / SCRIPT,
            Path(plan_path).resolve(),
            (ROOT / plan["source_campaign"]) / "seal.json",
        ]
    )


def seal(directory: Path, *, force: bool = False) -> Path:
    """Freeze the plan, the sources it runs, and the campaign it reads."""
    directory = Path(directory)
    plan_path = directory / "plan.json"
    plan = json.loads(plan_path.read_text())
    seal_path = directory / "seal.json"
    if seal_path.exists() and not force:
        raise SystemExit(
            f"{seal_path} exists: a sealed benchmark is frozen once launched; "
            "use a new benchmark name, or --force before any launch"
        )
    record = dict(
        benchmark=plan["benchmark"],
        engine=ENGINE,
        sources={
            source._relative(path): _sha256(path)
            for path in _sealed_files(plan_path, plan)
        },
    )
    seal_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return seal_path


def verify(plan_path: Path) -> tuple[dict, dict]:
    """Check the benchmark's seal and the source campaign's; return the plan."""
    plan_path = Path(plan_path)
    seal_path = plan_path.parent / "seal.json"
    if not seal_path.exists():
        raise SystemExit(f"{seal_path} is missing: seal the benchmark first")
    record = json.loads(seal_path.read_text())
    plan = json.loads(plan_path.read_text())
    changed = [
        name
        for name, digest in record["sources"].items()
        if not (ROOT / name).exists() or _sha256(ROOT / name) != digest
    ]
    if changed:
        raise SystemExit(
            f"benchmark {record['benchmark']} is sealed and these changed since: "
            f"{', '.join(changed)}; make a new benchmark instead"
        )
    campaign_seal = json.loads(
        ((ROOT / plan["source_campaign"]) / "seal.json").read_text()
    )
    lane = next(iter(campaign_seal["lanes"].values()))
    source.verify(ROOT / lane["calibrate"])
    return plan, record


# ---------------------------------------------------------------- the runs


@dataclass(frozen=True)
class Task:
    study: str
    solver: str
    index: int
    kinds: str
    method: str
    sweep: str
    sigma: float
    b: int

    def key(self, round_: int, seed: int) -> tuple:
        return (
            int(round_),
            int(seed),
            self.study,
            self.method,
            self.sweep,
            _sigma_text(self.sigma),
            int(self.b),
        )


def _task(study: dict, configuration: dict) -> Task:
    return Task(
        study=study["name"],
        solver=study["solver"],
        index=int(configuration["index"]),
        kinds=configuration["kinds"],
        method=configuration["method"],
        sweep=configuration["sweep"],
        sigma=_sigma(configuration["sigma"]),
        b=int(configuration["b"]),
    )


def tasks_for(problem: dict, seed: int, near_exact_seeds) -> list[Task]:
    """Every run of one instance, in plan order (before shuffling)."""
    tasks = []
    for study in problem["studies"]:
        tasks += [_task(study, c) for c in study["configurations"]]
        if seed in near_exact_seeds:
            tasks.append(_task(study, study["near_exact"]))
    exact = next(s for s in problem["studies"] if s["name"] == problem["exact_study"])
    tasks.append(_task(exact, exact["exact"]))
    return tasks


def order(plan: dict, problem: dict, round_: int, seed: int, count: int) -> list[int]:
    """The seeded order of an instance's runs, the same on every resumption."""
    label = zlib.crc32(problem["slug"].encode())
    stream = np.random.default_rng([plan["order_seed"], int(round_), int(seed), label])
    return [int(i) for i in stream.permutation(count)]


def _done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    done = set()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if any(row.get(field) in (None, "") for field in ("round", "seed", "b")):
                continue  # a line cut short when a job was stopped
            done.add(
                (
                    int(row["round"]),
                    int(row["seed"]),
                    row["study"],
                    row["method"],
                    row["sweep"],
                    _sigma_text(_sigma(row["sigma"])),
                    int(row["b"]),
                )
            )
    return done


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def _check(config: dict, task: Task) -> None:
    if (
        str(config["method"]) != task.method
        or str(config["sweep"]) != task.sweep
        or int(config["b"]) != task.b
        or not _same_sigma(float(config["sigma"]), task.sigma)
    ):
        raise SystemExit(
            f"{task.study}: configuration {task.index} is {config}, not the planned "
            f"{task}; the campaign and the plan disagree"
        )


def _settings(study: dict) -> dict:
    args = build_parser(study["module"]).parse_args([f"@{ROOT / study['args']}"])
    return settings_from(args, study["module"])


def run(
    plan_path: Path,
    results_dir: Path,
    problems: list[str],
    *,
    instances: int | None = None,
    near_exact_instances: int | None = None,
) -> int:
    """Time the planned runs of ``problems``; the number of runs timed now."""
    plan, record = verify(plan_path)
    by_slug = {p["slug"]: p for p in plan["problems"]}
    unknown = set(problems) - set(by_slug)
    if unknown:
        raise SystemExit(f"no problem {sorted(unknown)} in {plan_path}")
    results_dir = Path(results_dir)
    host, processor = platform.node(), cpu_model()
    job = os.environ.get("SLURM_JOB_ID", "local")
    timed = 0
    for name in problems:
        problem = by_slug[name]
        settings = {s["name"]: _settings(s) for s in problem["studies"]}
        parts = {
            study: ROOT / study_directory(values) / PARTS
            for study, values in settings.items()
        }
        seeds = problem["seeds"][:instances] if instances else problem["seeds"]
        near = problem["near_exact_seeds"]
        if near_exact_instances is not None:
            near = near[:near_exact_instances]
        near = set(near)
        output = results_dir / problem["slug"] / MEASUREMENTS
        output.parent.mkdir(parents=True, exist_ok=True)
        provenance = dict(_provenance(None), benchmark_seal=record["sources"])
        (output.parent / f"provenance-{job}.json").write_text(
            json.dumps(provenance, indent=1, sort_keys=True) + "\n"
        )
        done = _done(output)
        print(
            f"[{plan['benchmark']}] {name} on {host} ({processor}): "
            f"{len(seeds)} instances, near-exact on {sorted(near)}",
            flush=True,
        )
        for round_ in range(int(plan["rounds"])):
            for seed in seeds:
                tasks = tasks_for(problem, seed, near)
                sequence = order(plan, problem, round_, seed, len(tasks))
                pending = [
                    (position, tasks[i])
                    for position, i in enumerate(sequence)
                    if tasks[i].key(round_, seed) not in done
                ]
                if not pending:
                    continue
                needed = sorted({task.study for _, task in pending})
                setup_order = np.random.default_rng(
                    [plan["order_seed"], round_, seed, 1]
                ).permutation(len(needed))
                setups = {}
                for i in setup_order:
                    study = needed[int(i)]
                    part = parts[study] / str(seed)
                    if not (part / SETUP / "truth.npy").exists():
                        raise SystemExit(
                            f"{part / SETUP} is missing: the benchmark reads the "
                            "campaign's stored setups and never writes them"
                        )
                    module = next(
                        s["module"] for s in problem["studies"] if s["name"] == study
                    )
                    started = time.perf_counter()
                    setup = prepare_instance(module, seed, settings[study], part)
                    prepared = time.perf_counter() - started
                    own = setup.factory.setup_seconds if setup.factory else 0.0
                    setups[study] = (setup, prepared, own)
                for position, task in pending:
                    setup, prepared, own = setups[task.study]
                    _check(setup.configurations[task.index], task)
                    if task.method == "exact":
                        started = time.perf_counter()
                        values = settings[task.study]
                        direct = int(values.get("exact_direct_max", 5_000))
                        setup.instance.exact_solver(setup.c, direct)
                        own = time.perf_counter() - started
                    started_at = time.time()
                    row, _ = run_configuration(setup, settings[task.study], task.index)
                    _append(
                        output,
                        dict(
                            problem=problem["name"],
                            study=task.study,
                            solver=task.solver,
                            seed=seed,
                            round=round_,
                            position=position,
                            kinds=task.kinds,
                            method=task.method,
                            sweep=task.sweep,
                            sigma=_sigma_text(task.sigma),
                            b=task.b,
                            started_at=f"{started_at:.3f}",
                            runtime_seconds=row.get("runtime_seconds", nan),
                            own_setup_seconds=own,
                            prepare_seconds=prepared,
                            outer_iterations=row.get("outer_iterations", ""),
                            total_inner=row.get("total_inner", ""),
                            converged=row.get("converged", False),
                            stopped_by=row.get("stopped_by", ""),
                            error=row.get("error", ""),
                            hostname=host,
                            cpu_model=processor,
                            slurm_job_id=job,
                        ),
                    )
                    timed += 1
                    print(
                        f"  seed {seed} #{position:02d} {task.solver:20s} "
                        f"{task.kinds:24s} "
                        f"{float(row.get('runtime_seconds', nan)):8.2f} s "
                        f"{row.get('stopped_by', '')}",
                        flush=True,
                    )
                del setups
                gc.collect()
    return timed


# ---------------------------------------------------------------- the summary


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    finite = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if finite.size == 0:
        return nan, nan, nan
    return (
        float(np.percentile(finite, 10)),
        float(np.median(finite)),
        float(np.percentile(finite, 90)),
    )


def summarize(plan_path: Path, results_dir: Path) -> tuple[list[dict], str]:
    """Per configuration medians, the paired comparison, and the checks."""
    plan = json.loads(Path(plan_path).read_text())
    results_dir = Path(results_dir)
    table, lines = [], []
    for problem in plan["problems"]:
        path = results_dir / problem["slug"] / MEASUREMENTS
        if not path.exists():
            continue
        runs = [r for r in read_rows(path) if r.get("stopped_by") not in (None, "")]
        reference = {}
        for study in problem["studies"]:
            directory = ROOT / plan["source_results"] / study["name"]
            for r in read_rows(directory / "results.csv"):
                key = (
                    study["name"],
                    int(r["seed"]),
                    str(r["method"]),
                    str(r["sweep"]),
                    _sigma_text(_sigma(r["sigma"])),
                    int(r["b"]),
                )
                reference[key] = (int(r["outer_iterations"]), int(r["total_inner"]))
        groups: dict[tuple, list[dict]] = defaultdict(list)
        same = different = unmatched = 0
        for r in runs:
            sigma = _sigma_text(_sigma(r["sigma"]))
            key = (r["study"], str(r["method"]), str(r["sweep"]), sigma, int(r["b"]))
            groups[key].append(r)
            counts = reference.get((r["study"], int(r["seed"]), *key[1:]))
            if counts is None:
                unmatched += 1
            elif counts == (int(r["outer_iterations"]), int(r["total_inner"])):
                same += 1
            else:
                different += 1
        cost = {
            key: {
                int(r["seed"]): float(r["runtime_seconds"])
                + float(r["own_setup_seconds"])
                for r in rows
            }
            for key, rows in groups.items()
        }

        def median_cost(key: tuple, cost=cost) -> float:
            return float(np.median(list(cost[key].values())))

        best: dict[str, tuple] = {}
        for key, rows in groups.items():
            if key[1] != "inexact" or not all(bool(r["converged"]) for r in rows):
                continue
            solver = rows[0]["solver"]
            if solver not in best or median_cost(key) < median_cost(best[solver]):
                best[solver] = key
        fixed = {
            rows[0]["solver"]: key
            for key, rows in groups.items()
            if key[1] == "inexact" and "fixed" in str(rows[0]["kinds"]).split("+")
        }
        exact = [key for key in groups if key[1] == "exact"]
        contenders, at_fixed = dict(best), dict(fixed)
        if exact:
            contenders["exact ADMM"] = at_fixed["exact ADMM"] = exact[0]
        seeds = sorted({int(r["seed"]) for r in runs})

        def paired(chosen: dict[str, tuple], cost=cost, seeds=seeds) -> dict:
            """Per solver, its cost over the fastest one's, instance by instance."""
            ratios: dict[str, list[float]] = defaultdict(list)
            for seed in seeds:
                costs = {
                    solver: cost[key][seed]
                    for solver, key in chosen.items()
                    if seed in cost[key]
                }
                if len(costs) < len(chosen):
                    continue
                fastest = min(costs.values())
                for solver, value in costs.items():
                    ratios[solver].append(value / fastest)
            return ratios

        ratios, ratios_fixed = paired(contenders), paired(at_fixed)
        speedups: dict[str, list[float]] = defaultdict(list)
        for key, rows in groups.items():
            if key[1] != "near_exact":
                continue
            solver = rows[0]["solver"]
            if solver not in best:
                continue
            for seed, value in cost[key].items():
                if seed in cost[best[solver]]:
                    speedups[solver].append(value / cost[best[solver]][seed])
        for key, rows in sorted(groups.items()):
            solver = rows[0]["solver"]
            runtime = [float(r["runtime_seconds"]) for r in rows]
            own = [float(r["own_setup_seconds"]) for r in rows]
            p10, median, p90 = _quantiles(list(cost[key].values()))
            is_best = best.get(solver) == key
            label = "exact ADMM" if key[1] == "exact" else solver
            compared = is_best or key[1] == "exact"
            ratio = _quantiles(ratios[label]) if compared else (nan,) * 3
            is_fixed = fixed.get(solver) == key or key[1] == "exact"
            ratio_fixed = _quantiles(ratios_fixed[label]) if is_fixed else (nan,) * 3
            speedup = _quantiles(speedups[solver]) if is_best else (nan,) * 3
            table.append(
                dict(
                    problem=problem["name"],
                    solver=label,
                    study=key[0],
                    kinds=rows[0]["kinds"],
                    method=key[1],
                    sweep=key[2],
                    sigma=key[3],
                    b=key[4],
                    n=len(rows),
                    n_converged=sum(bool(r["converged"]) for r in rows),
                    runtime_median=float(np.median(runtime)),
                    own_setup_median=float(np.median(own)),
                    cost_p10=p10,
                    cost_median=median,
                    cost_p90=p90,
                    fastest_here=is_best,
                    ratio_to_fastest_median=ratio[1],
                    ratio_to_fastest_p90=ratio[2],
                    near_exact_over_fastest_median=speedup[1],
                    ratio_at_fixed_median=ratio_fixed[1],
                    hosts=";".join(sorted({str(r["hostname"]) for r in rows})),
                )
            )
        probe = [float(r["runtime_seconds"]) for r in runs if r["method"] == "exact"]
        lines.append(
            f"== {problem['name']}: {len(runs)} runs on "
            f"{', '.join(sorted({str(r['hostname']) for r in runs}))}; iteration "
            f"counts equal to the campaign's in {same}, different in {different}, "
            f"unmatched {unmatched}"
        )
        if probe:
            lines.append(
                f"   exact-ADMM probe: {min(probe):.2f}-{max(probe):.2f} s "
                f"(median {float(np.median(probe)):.2f} s, max/min "
                f"{max(probe) / min(probe):.2f}) over {len(probe)} instances"
            )
        for title, chosen, shares in (
            ("each solver at its fastest configuration here", contenders, ratios),
            ("every solver at sigma 0.1, s 1", at_fixed, ratios_fixed),
        ):
            lines.append(f"   {title}:")
            ranked = sorted(
                chosen,
                key=lambda s, shares=shares: float(np.median(shares[s] or [math.inf])),
            )
            for solver in ranked:
                key = chosen[solver]
                p10, median, p90 = _quantiles(list(cost[key].values()))
                r10, r50, r90 = _quantiles(shares[solver])
                s50 = nan
                if chosen is contenders and solver in speedups:
                    s50 = _quantiles(speedups[solver])[1]
                config = "" if key[1] == "exact" else f"sigma {key[3]}, s {key[4]}"
                lines.append(
                    f"     {solver:20s} {config:22s} cost {median:8.2f} s "
                    f"[{p10:.2f}, {p90:.2f}]  x fastest {r50:5.2f} "
                    f"[{r10:.2f}, {r90:.2f}]"
                    + (f"  near-exact/this {s50:6.1f}" if not math.isnan(s50) else "")
                )
    return table, "\n".join(lines)


# ---------------------------------------------------------------- the commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("plan", help="choose the configurations to time")
    making.add_argument("--campaign", type=Path, required=True)
    making.add_argument("--benchmark", required=True)
    making.add_argument("--output", type=Path, required=True)
    making.add_argument(
        "--near-exact",
        action="append",
        default=[],
        metavar="PROBLEM=K",
        help="time the near-exact baseline on the first K instances of PROBLEM",
    )
    making.add_argument("--near-exact-default", type=int, default=5)
    making.add_argument("--rounds", type=int, default=1)
    making.add_argument(
        "--instances", type=int, help="the first N instances of each problem"
    )
    sealing = commands.add_parser("seal", help="freeze a benchmark before launch")
    sealing.add_argument("directory", type=Path)
    sealing.add_argument("--force", action="store_true")
    running = commands.add_parser("run", help="time the planned runs (inside Slurm)")
    running.add_argument("--plan", type=Path, required=True)
    running.add_argument("--results-dir", type=Path, required=True)
    running.add_argument("--problem", action="append", required=True)
    running.add_argument("--instances", type=int, help="the first N instances only")
    running.add_argument(
        "--near-exact-instances", type=int, help="at most K near-exact instances"
    )
    running.add_argument(
        "--allow-local",
        action="store_true",
        help="time outside a Slurm allocation (workstations, never login nodes)",
    )
    summing = commands.add_parser("summarize", help="compare the timed runs")
    summing.add_argument("--plan", type=Path, required=True)
    summing.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        near = {}
        for item in args.near_exact:
            name, _, count = item.partition("=")
            near[name] = int(count)
        plan = make_plan(
            args.campaign,
            args.benchmark,
            near,
            args.near_exact_default,
            args.rounds,
            args.instances,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=1) + "\n")
        print(args.output)
    elif args.command == "seal":
        print(seal(args.directory, force=args.force))
    elif args.command == "run":
        if not (os.environ.get("SLURM_JOB_ID") or args.allow_local):
            raise SystemExit(
                "the benchmark times inside a Slurm allocation; pass --allow-local "
                "on a workstation (never on a login node)"
            )
        count = run(
            args.plan,
            args.results_dir,
            args.problem,
            instances=args.instances,
            near_exact_instances=args.near_exact_instances,
        )
        print(f"timed {count} runs", flush=True)
    else:
        table, report = summarize(args.plan, args.results_dir)
        if not table:
            raise SystemExit(f"{args.results_dir} holds no measurements")
        write_rows(Path(args.results_dir) / SUMMARY, table)
        (Path(args.results_dir) / "report.txt").write_text(report + "\n")
        print(report)
        print(Path(args.results_dir) / SUMMARY)


if __name__ == "__main__":
    sys.exit(main())
