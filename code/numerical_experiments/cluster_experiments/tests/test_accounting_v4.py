import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from releases.accounting_v4.experiments import baselines, manifest, solvers
from releases.accounting_v4.experiments.inner import Inner
from releases.accounting_v4.experiments.io import identity, read_csv, write_csv
from releases.accounting_v4.experiments.problems import make_qp
from releases.accounting_v4.experiments.run import execute
from releases.accounting_v4.experiments.tuning import resolve

RELEASE = Path(__file__).resolve().parents[1] / "releases/accounting_v4"


def small_config(**changes):
    return dict(
        dict(
            family="qp",
            dataset="synthetic",
            side=4,
            seed=0,
            inner="cg",
            ratio=10.0,
            sigma=0.2,
            stride=1,
            phase="smoke",
            time_limit=5.0,
            implementation="accounting_v4",
            max_inner=10000,
            max_outer=10000,
        ),
        **changes,
    )


def run_parent(tmp_path, config, *extra):
    manifest_path = tmp_path / "manifest.csv"
    run_id = identity(config)
    write_csv(manifest_path, [dict(run_id=run_id, config=json.dumps(config))])
    env = dict(os.environ, SLURM_CPUS_PER_TASK="1", OPENBLAS_NUM_THREADS="1")
    env.pop("SLURM_JOB_ID", None)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.run",
            "--manifest",
            str(manifest_path),
            "--task-id",
            "0",
            "--results-dir",
            str(tmp_path / "results"),
            "--data-dir",
            str(tmp_path),
            "--memory-gb",
            "1",
            *extra,
        ],
        cwd=RELEASE,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return tmp_path / "results" / run_id


def test_forced_rerun_is_measured_alone(tmp_path):
    output = run_parent(tmp_path, small_config())
    first = read_csv(output / "result.csv")[0]
    output = run_parent(tmp_path, small_config(), "--force")
    second = read_csv(output / "result.csv")[0]
    assert second["attempts"] == "1"
    assert float(second["total_seconds"]) == pytest.approx(
        float(second["clean_attempt_seconds"])
    )
    # accounting_v3 reported roughly the sum of both runs here.
    assert float(second["total_seconds"]) < 1.5 * float(first["total_seconds"]) + 1
    assert list((output / "superseded").glob("*/attempts/attempts.json"))


def test_prior_hard_kill_does_not_poison_the_clean_timing(tmp_path):
    config = small_config()
    output = tmp_path / "results" / identity(config)
    (output / "attempts").mkdir(parents=True)
    killed = [dict(id="1", status="running", elapsed_seconds=0.0, job_id="x")]
    (output / "attempts" / "attempts.json").write_text(json.dumps(killed))
    row = read_csv(run_parent(tmp_path, config) / "result.csv")[0]
    assert row["primary_eligible"] == "True"
    assert math.isfinite(float(row["total_seconds"]))
    assert row["prior_unclean_attempts"] == "1"
    assert row["campaign_accounting_complete"] == "False"
    assert math.isnan(float(row["all_attempts_seconds"]))


def test_every_check_gradient_is_counted(tmp_path):
    row = execute(small_config(inner="gd"), tmp_path, tmp_path)
    assert row["converged"]
    # At stride 1 every check gradient is reused by the next GD step, so the
    # inner counter alone misses almost all products.
    assert row["check_products"] == row["checks"]
    assert (
        row["hessian_products_total"]
        == row["inner_hessian_products"] + row["check_products"]
    )


def test_recurrence_gradient_matches_recomputed_gradient():
    p = make_qp(0, 4)
    c = p.L / 10
    inner = Inner(p, c, "cg", np.random.default_rng(0))
    w = np.random.default_rng(1).normal(size=p.n)
    inner.start(np.zeros(p.n), c * w - p.q)
    for _ in range(5):
        inner.step()
    recurrence = c * (w - inner.x) - inner.r
    np.testing.assert_allclose(recurrence, p.gradient(inner.x), atol=1e-10)


@pytest.mark.parametrize("method", ["cg", "nystrom_pcg"])
def test_recurrence_check_converges_and_saves_products(tmp_path, method):
    base = small_config(inner=method, rank=4)
    exact = execute(dict(base, check_source="recompute"), tmp_path / "a", tmp_path)
    cheap = execute(dict(base, check_source="recurrence"), tmp_path / "b", tmp_path)
    assert exact["converged"] and cheap["converged"]
    assert cheap["check_source"] == "recurrence"
    assert cheap["check_products"] < exact["check_products"]
    assert max(cheap["stationarity"], cheap["feasibility"]) <= 1e-6


def test_work_budget_counts_kaczmarz_rows_not_steps():
    p = make_qp(0, 4)
    config = small_config(
        inner="block_kaczmarz", block_size=1, max_inner=2, budget_unit="work"
    )
    work = solvers.solve(p, config)[0]
    steps = solvers.solve(p, dict(config, budget_unit="steps"))[0]
    assert steps["total_inner"] <= 2
    # Two Hessian products of work equal two passes over the 64 rows.
    assert work["total_inner"] > steps["total_inner"]
    assert work["work_units"] <= 2 + 1e-9


def test_nystrom_has_no_verified_model_recommendation():
    p = make_qp(0, 4)
    with pytest.raises(ValueError, match="no verified"):
        resolve(p, small_config(inner="nystrom_pcg", rank=4, sigma="optimal"))


def test_kaczmarz_strides_scale_with_the_number_of_blocks():
    instance = dict(family="qp", side=40)
    small = manifest.strides(instance, dict(inner="block_kaczmarz", block_size=1))
    assert max(small) == 40**3
    assert manifest.strides(instance, dict(inner="cg")) == [1, 2, 5, 10, 20]


def write_result(root, name, config, **row):
    write_csv(
        root / name / "result.csv",
        [dict(config=json.dumps(config), run_id=name, **row)],
    )


def test_selection_uses_clean_attempts_from_older_releases(tmp_path):
    fast = small_config(sigma=0.5)
    slow = small_config(sigma=0.2)
    # accounting_v3 marked this run NaN after an earlier hard kill, although
    # its final attempt restarted from scratch and is a clean measurement.
    write_result(
        tmp_path,
        "a",
        fast,
        status="converged",
        converged=True,
        total_seconds="nan",
        attempt_worker_wall_seconds=1.0,
        setup_accounting_complete=False,
    )
    write_result(
        tmp_path,
        "b",
        slow,
        status="converged",
        converged=True,
        total_seconds=5.0,
        attempt_worker_wall_seconds=5.0,
    )
    (winner,) = manifest.winners(tmp_path)
    assert winner["sigma"] == 0.5


def test_infrastructure_events_require_reruns_instead_of_losing(tmp_path):
    write_result(
        tmp_path,
        "a",
        small_config(sigma=0.5),
        status="interrupted",
        converged=False,
        total_seconds=1.0,
    )
    write_result(
        tmp_path,
        "b",
        small_config(sigma=0.2),
        status="converged",
        converged=True,
        total_seconds=5.0,
    )
    with pytest.raises(ValueError, match="infrastructure"):
        manifest.winners(tmp_path)


def test_resumed_legacy_runs_have_no_clean_time():
    resumed = dict(
        total_seconds=10,
        solve_seconds=9,
        generation_seconds=1,
        process_seconds=4,
    )
    assert math.isnan(manifest.clean_seconds(resumed))
    assert manifest.clean_seconds(dict(resumed, process_seconds=10)) == 10


def test_budgets_separate_selection_groups():
    short = manifest.group_key(small_config(time_limit=120.0))
    full = manifest.group_key(small_config(time_limit=1800.0))
    assert short != full


def test_evaluate_phase_uses_held_out_seeds(tmp_path, monkeypatch):
    config = small_config(sigma=0.2)
    write_csv(
        tmp_path / "tune.csv",
        [dict(task_id=0, run_id="a", config=json.dumps(config))],
    )
    write_result(
        tmp_path / "results",
        "a",
        config,
        status="converged",
        converged=True,
        total_seconds=1.0,
        clean_attempt_seconds=1.0,
    )
    common = [
        "manifest",
        "evaluate",
        "--from-results",
        str(tmp_path / "results"),
        "--from-manifest",
        str(tmp_path / "tune.csv"),
        "--output",
        str(tmp_path / "evaluate.csv"),
    ]
    monkeypatch.setattr(sys, "argv", [*common, "--seeds", "0"])
    with pytest.raises(SystemExit):
        manifest.main()
    monkeypatch.setattr(sys, "argv", common)
    manifest.main()
    seeds = sorted(
        json.loads(row["config"])["seed"] for row in read_csv(tmp_path / "evaluate.csv")
    )
    assert seeds == [3, 3, 3, 4, 4, 4, 5, 5, 5]  # three initialization arms


def test_baseline_ladder_tightens_from_the_requested_factor():
    assert baselines.ladder(dict(), 1e-6) == pytest.approx([1e-7, 1e-8, 1e-9])
    assert baselines.ladder(dict(baseline_tolerance_rungs=1), 1e-6) == pytest.approx(
        [1e-7]
    )


@pytest.mark.parametrize("method", ["osqp", "scs"])
def test_external_baselines_record_their_tolerance_ladder(method):
    p = make_qp(0, 4)
    row, history = baselines.external(p, small_config(inner=method))
    used = json.loads(row["native_tolerance_ladder"])
    assert 1 <= len(used) <= 3
    assert len(history) == len(used)
    assert row["converged"] == (max(row["stationarity"], row["feasibility"]) <= 1e-6)


def test_interruption_during_build_is_recorded(tmp_path, monkeypatch):
    from releases.accounting_v4.experiments import run

    original = run.build

    def interrupted_build(*args):
        problem = original(*args)
        solvers.INTERRUPTED = True
        return problem

    monkeypatch.setattr(run, "build", interrupted_build)
    row = execute(small_config(), tmp_path, tmp_path)
    assert row["status"] == "interrupted"
    assert row["interrupted_stage"] == "build"
    solvers.INTERRUPTED = False


def test_a_solver_that_never_converges_does_not_block_selection(tmp_path):
    write_result(
        tmp_path,
        "a",
        small_config(inner="osqp", sweep="baseline"),
        status="worker_timeout",
        converged=False,
    )
    write_result(
        tmp_path,
        "b",
        small_config(),
        status="converged",
        converged=True,
        clean_attempt_seconds=1.0,
    )
    (winner,) = manifest.winners(tmp_path)
    assert winner["inner"] == "cg"
