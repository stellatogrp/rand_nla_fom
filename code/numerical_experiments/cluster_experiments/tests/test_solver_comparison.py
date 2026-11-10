"""The cross-solver comparison: what it measures and which configurations it picks."""

import csv
import json

from plots import solver_comparison

CONFIG_FIELDS = ["method", "sweep", "sigma", "b", "sigma_is_optimal"]
# method, sweep, sigma, b, sigma*?, runtimes of the two instances, converged?
RUNS = [
    ("exact", "reference", "nan", 0, False, (2.0, 2.2), True),
    ("near_exact", "reference", "nan", 1, False, (40.0, 44.0), True),
    ("inexact", "b", "0.1", 1, False, (10.0, 12.0), True),
    ("inexact", "b", "0.1", 5, False, (1.0, 1.0), False),  # fastest, not converged
    ("inexact", "sigma", "0.5", 1, False, (8.0, 9.0), True),
    ("inexact", "sigma", "0.07", 1, True, (11.0, 11.0), True),
]
INNER_SETUP = {0: 3.0, 1: 5.0}
COMMON_SETUP = 100.0  # projector and Lanczos: shared by every solver


def write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def study(tmp_path, cpu="Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz"):
    directory = tmp_path / "admm_qp_side4_gmres"
    directory.mkdir()
    identity = dict(family="qp", side=4, inner="gmres", rank=64, algorithm="admm")
    (directory / "settings.json").write_text(json.dumps(dict(identity=identity)))
    machine = dict(cpu_model=cpu, hostname="della-h12n14")
    (directory / "calibration.json").write_text(json.dumps(machine))
    write(
        directory / "configurations.csv",
        [dict(zip(CONFIG_FIELDS, run[:5], strict=True)) for run in RUNS],
    )
    write(
        directory / "parameters.csv",
        [
            dict(seed=seed, inner_setup_seconds=setup, factor_seconds=0.5)
            for seed, setup in INNER_SETUP.items()
        ],
    )
    results = []
    for index, (method, sweep, sigma, b, optimal, times, converged) in enumerate(RUNS):
        for seed, runtime in zip(INNER_SETUP, times, strict=True):
            results.append(
                dict(
                    seed=seed,
                    method=method,
                    sweep=sweep,
                    sigma=sigma,
                    b=b,
                    sigma_is_optimal=optimal,
                    converged=converged,
                    runtime_seconds=runtime,
                    setup_seconds=COMMON_SETUP,
                    total_inner=10 * b,
                    outer_iterations=7,
                )
            )
            runs = directory / "parts" / str(seed) / "runs"
            runs.mkdir(parents=True, exist_ok=True)
            (runs / f"{index:03d}.provenance.json").write_text(json.dumps(machine))
    write(directory / "results.csv", results)
    return directory


def test_kinds_are_picked_by_their_rules(tmp_path):
    table = solver_comparison.configurations(study(tmp_path))
    chosen = solver_comparison.pick(table)
    assert chosen["exact"][0] == "exact" and chosen["near_exact"][0] == "near_exact"
    assert chosen["fixed"] == ("inexact", "b", 0.1, 1)
    assert chosen["sigma_star"] == ("inexact", "sigma", 0.07, 1)
    # The fastest configuration did not converge everywhere, so it is not "best".
    assert chosen["best"] == ("inexact", "sigma", 0.5, 1)


def test_cost_is_runtime_plus_the_solvers_own_setup(tmp_path):
    table = solver_comparison.configurations(study(tmp_path))
    best = table[("inexact", "sigma", 0.5, 1)]
    assert best["cost"] == (8.0 + 3.0 + 9.0 + 5.0) / 2
    assert best["total"] == COMMON_SETUP + (8.0 + 9.0) / 2
    # The baselines' sigma is nan, which matches no key by equality.
    exact = next(e for k, e in table.items() if k[0] == "exact")
    assert exact["own_setup"] == 0.5  # the factorization, not the inner setup
    assert exact["n_converged"] == exact["n_runs"] == 2


def test_rows_report_speedup_and_hosts(tmp_path):
    study(tmp_path)
    rows = solver_comparison.compare(tmp_path)
    by_kind = {row["kind"]: row for row in rows}
    assert set(by_kind) == set(solver_comparison.KINDS)
    near = by_kind["near_exact"]["cost_median"]
    assert by_kind["best"]["speedup"] == near / by_kind["best"]["cost_median"]
    assert by_kind["best"]["run_hosts"] == "h12n14 Platinum 8380 x2"
    assert by_kind["best"]["setup_host"] == "h12n14 Platinum 8380"
    assert solver_comparison.host_label({}) == "unknown"
    assert all(row["problem"] == "qp side 4" for row in rows)
    assert "fixed (sigma 0.1, s 1)" in solver_comparison.report(rows)
