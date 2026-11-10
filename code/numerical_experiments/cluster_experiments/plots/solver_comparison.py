"""Compare the inner solvers of each ADMM problem from the collected study tables.

    python -m plots.solver_comparison --results-dir results/studies_v2

For every problem (the QP at each side, the LASSO on each design) and every
inner solver of the original-format studies (``studies``), one row per
configuration kind, each with medians over the instances of

``runtime``
    the algorithm's wall-clock time (``runtime_seconds``);
``own_setup``
    the setup that belongs to the solver: the inner solver's
    (``inner_setup_seconds``, the Nystrom sketch for instance; zero for
    GMRES) or, for exact ADMM, its factorization (``factor_seconds``);
``cost``
    the two per run, the measure the rows are compared and ranked by;
``total``
    the run's ``setup_seconds`` plus its runtime, which adds the projector
    and the Lanczos estimate of ``L``: the same computation for every solver
    of a problem, so it is reported but does not tell solvers apart;

and the total inner and outer iterations.  The kinds:

``fixed``
    ``sigma = 0.1`` and ``s = 1``, the first point of the stride sweep, the
    same configuration for every solver;
``sigma_star``
    ``sigma*`` of Proposition 5.1, for the solvers with contraction constants;
``best``
    the configuration with the smallest median cost among those in which
    every instance converged, chosen per solver (so optimistic for all of
    them alike);
``near_exact`` and ``exact``
    the baselines, for scale.

``speedup`` is the near-exact baseline's median cost over the row's.  Each
configuration runs as one Slurm task on a shared node, and wall-clock time
depends on the node: Della's ``icelake`` nodes come in two processor models,
and the load on a node changes timings as much.  So ``run_hosts`` counts the
host and processor of the row's runs, and ``setup_host`` names those of the
study's calibrate task, which timed the setups; compare the times of rows on
different hosts with that in mind.  Writes ``solver_comparison.csv`` in the
results directory and prints it by problem.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from studies._io import read_rows, write_rows
from studies.aggregate import config_key

OUTPUT = "solver_comparison.csv"
KINDS = ("fixed", "sigma_star", "best", "near_exact", "exact")
MEDIANS = (
    "runtime",
    "own_setup",
    "cost",
    "total",
    "total_inner",
    "outer",
)
nan = float("nan")


def _identity(directory: Path) -> dict:
    return json.loads((directory / "settings.json").read_text())["identity"]


def problem_of(identity: dict) -> str:
    if identity["family"] == "qp":
        return f"qp side {identity['side']}"
    return f"{identity['family']} {identity['dataset']}"


def solver_of(identity: dict) -> str:
    inner = identity["inner"]
    return f"{inner} rank {identity['rank']}" if inner == "nystrom_pcg" else inner


def short_cpu(model: str) -> str:
    """``Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz`` -> ``Platinum 8380``."""
    words = model.replace("(R)", "").split()
    for family in ("Platinum", "Gold", "Silver"):
        if family in words:
            position = words.index(family)
            return " ".join(words[position : position + 2])
    return model


def host_label(provenance: dict) -> str:
    """``h12n14 Platinum 8380`` from a provenance record (host and processor)."""
    model = short_cpu(provenance.get("cpu_model", "unknown"))
    host = provenance.get("hostname", "").removeprefix("della-")
    return f"{host} {model}" if host else model


def run_hosts(directory: Path, key: tuple) -> str:
    configurations = read_rows(directory / "configurations.csv")
    index = next(i for i, c in enumerate(configurations) if config_key(c) == key)
    hosts: Counter = Counter()
    for part in sorted((directory / "parts").iterdir(), key=lambda p: p.name):
        path = part / "runs" / f"{index:03d}.provenance.json"
        if path.exists():
            hosts[host_label(json.loads(path.read_text()))] += 1
    return "; ".join(f"{host} x{count}" for host, count in hosts.most_common())


def setup_host(directory: Path) -> str:
    path = directory / "calibration.json"
    return host_label(json.loads(path.read_text())) if path.exists() else "unknown"


def _median(values: list[float]) -> float:
    finite = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.median(finite)) if finite else nan


def configurations(directory: Path) -> dict[tuple, dict]:
    """Per configuration of a study: its counts and medians over instances."""
    instances = {int(p["seed"]): p for p in read_rows(directory / "parameters.csv")}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in read_rows(directory / "results.csv"):
        groups[config_key(row)].append(row)
    table = {}
    for key, rows in groups.items():
        own = [
            float(
                instances[int(r["seed"])][
                    "factor_seconds" if key[0] == "exact" else "inner_setup_seconds"
                ]
            )
            for r in rows
        ]
        runtime = [float(r["runtime_seconds"]) for r in rows]
        setup = [float(r["setup_seconds"]) for r in rows]
        table[key] = dict(
            method=key[0],
            sweep=key[1],
            sigma=key[2],
            b=key[3],
            sigma_is_optimal=any(bool(r.get("sigma_is_optimal")) for r in rows),
            n_runs=len(rows),
            n_converged=sum(bool(r.get("converged")) for r in rows),
            runtime=_median(runtime),
            own_setup=_median(own),
            cost=_median([t + s for t, s in zip(runtime, own, strict=True)]),
            total=_median([s + t for s, t in zip(setup, runtime, strict=True)]),
            total_inner=_median([r["total_inner"] for r in rows]),
            outer=_median([r["outer_iterations"] for r in rows]),
        )
    return table


def pick(table: dict[tuple, dict]) -> dict[str, tuple]:
    """The configuration of every kind that the study has."""
    chosen: dict[str, tuple] = {}
    for key, entry in table.items():
        method, sweep, sigma, b = key
        if method in ("exact", "near_exact"):
            chosen[method] = key
        elif method == "inexact" and sweep == "b" and b == 1:
            chosen["fixed"] = key
        elif method == "inexact" and sweep == "sigma" and entry["sigma_is_optimal"]:
            chosen["sigma_star"] = key
    finished = [
        key
        for key, entry in table.items()
        if key[0] == "inexact" and entry["n_converged"] == entry["n_runs"]
    ]
    if finished:
        chosen["best"] = min(finished, key=lambda k: table[k]["cost"])
    return chosen


def compare(results_dir: Path) -> list[dict]:
    rows = []
    for directory in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if not (directory / "results.csv").exists():
            continue
        identity = _identity(directory)
        if identity["algorithm"] != "admm":
            continue
        table = configurations(directory)
        chosen = pick(table)
        near = table.get(chosen.get("near_exact"), {}).get("cost", nan)
        for kind in KINDS:
            if kind not in chosen:
                continue
            entry = table[chosen[kind]]
            rows.append(
                dict(
                    problem=problem_of(identity),
                    solver=solver_of(identity),
                    study=directory.name,
                    kind=kind,
                    **{k: entry[k] for k in ("sweep", "sigma", "b")},
                    n_runs=entry["n_runs"],
                    n_converged=entry["n_converged"],
                    **{f"{k}_median": entry[k] for k in MEDIANS},
                    speedup=near / entry["cost"] if entry["cost"] > 0 else nan,
                    run_hosts=run_hosts(directory, chosen[kind]),
                    setup_host=setup_host(directory),
                )
            )
    return rows


def _label(row: dict) -> str:
    if row["kind"] in ("exact", "near_exact"):
        return row["kind"]
    return f"{row['kind']} (sigma {row['sigma']:.3g}, s {row['b']})"


def report(rows: list[dict]) -> str:
    lines = []
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[row["problem"]].append(row)
    for problem, group in by_problem.items():
        lines.append(f"== {problem}")
        lines.append(
            f"{'solver':21s} {'configuration':33s} {'conv':>7s} {'cost s':>8s} "
            f"{'algo s':>8s} {'own su':>7s} {'total s':>8s} {'inner':>8s} "
            f"{'outer':>6s} {'speedup':>7s}  runs on / setup on"
        )
        best = {r["solver"]: r["cost_median"] for r in group if r["kind"] == "best"}
        for row in sorted(
            group,
            key=lambda r: (best.get(r["solver"], math.inf), KINDS.index(r["kind"])),
        ):
            lines.append(
                f"{row['solver']:21s} {_label(row):33s} "
                f"{row['n_converged']:>3d}/{row['n_runs']:<3d} "
                f"{row['cost_median']:8.1f} {row['runtime_median']:8.1f} "
                f"{row['own_setup_median']:7.1f} {row['total_median']:8.1f} "
                f"{row['total_inner_median']:8.0f} {row['outer_median']:6.0f} "
                f"{row['speedup']:7.2f}  {row['run_hosts']} / {row['setup_host']}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results/studies_v2"))
    parser.add_argument("--output", type=Path, help=f"default: <results-dir>/{OUTPUT}")
    args = parser.parse_args()
    rows = compare(args.results_dir)
    if not rows:
        raise SystemExit(f"{args.results_dir} has no collected ADMM study")
    output = args.output or args.results_dir / OUTPUT
    write_rows(output, rows)
    print(report(rows))
    print(output)


if __name__ == "__main__":
    main()
