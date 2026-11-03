import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from releases.accounting_v6.experiments import baselines, manifest, solvers
from releases.accounting_v6.experiments.inner import Inner
from releases.accounting_v6.experiments.io import identity, read_csv, write_csv
from releases.accounting_v6.experiments.problems import make_qp
from releases.accounting_v6.experiments.run import execute
from releases.accounting_v6.experiments.tuning import resolve

RELEASE = Path(__file__).resolve().parents[1] / "releases/accounting_v6"


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
            implementation="accounting_v6",
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
        == row["inner_hessian_products"]
        + row["check_products"]
        + row["calibration_hessian_products"]
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
    from releases.accounting_v6.experiments import run

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


def test_legacy_planner_filters_unsupported_model_parameters():
    from releases.accounting_v6.experiments import plan

    configs = [
        small_config(inner=inner, sigma=sigma, stride=stride)
        for inner in ("nystrom_pcg", "cg", "agd")
        for sigma, stride in ((0.2, 1), ("optimal", 1), (0.1, "optimal"))
    ]
    fast, screen = plan.split(configs)
    assert not screen
    assert len(fast) == 5  # CG model arms survive; Nyström/AGD use grids only.
    assert all(c["implementation"] == "accounting_v6" for c in fast)
    assert all(
        c["inner"] in manifest.MODEL_SOLVERS
        or "optimal" not in (c["sigma"], c["stride"])
        for c in fast
    )


def test_screen_and_expansion_both_use_block_scaled_strides():
    from releases.accounting_v6.experiments import plan

    configs = [
        small_config(
            side=side,
            seed=seed,
            inner=inner,
            block_size=8,
            inner_initialization=mode,
            sweep=sweep,
            sigma=sigma,
            stride=stride,
        )
        for side in (40, 64)
        for seed in (0, 1)
        for inner in ("block_kaczmarz", "cyclic_kaczmarz")
        for mode in ("warm", "fixed", "gaussian")
        for sweep, sigma, stride in (
            ("sigma", 0.2, 1),
            ("stride", 0.1, 1),
            ("stride", 0.1, 2),
            ("stride", 0.1, 20),
        )
    ]
    _, screen = plan.split(configs)
    expected = manifest.strides(
        dict(side=40), dict(inner="block_kaczmarz", block_size=8)
    )
    assert len(screen) == 6 * len(expected)
    assert {c["stride"] for c in screen} == set(expected)
    results = {
        identity(c): dict(status="time_limit", converged="False") for c in screen
    }
    # Success only at a large stride must be sufficient for pair expansion.
    success = next(
        c
        for c in screen
        if c["inner"] == "block_kaczmarz" and c["stride"] == max(expected)
    )
    results[identity(success)] = dict(status="converged", converged="True")
    selected, eligible = plan.expand(configs, screen, results)
    assert eligible == {("block_kaczmarz", 8)}
    for side in (40, 64):
        assert {
            c["stride"]
            for c in selected
            if c["side"] == side and c["sweep"] == "stride"
        } == set(
            manifest.strides(
                dict(side=side), dict(inner="block_kaczmarz", block_size=8)
            )
        )
    assert {c["seed"] for c in selected} == {0, 1}
    assert {c["inner_initialization"] for c in selected} == {
        "warm",
        "fixed",
        "gaussian",
    }
    assert len({identity(c) for c in selected}) == len(selected)
    results.pop(identity(success))
    with pytest.raises(ValueError, match="incomplete"):
        plan.expand(configs, screen, results)


@pytest.mark.parametrize("status", ["error", "infrastructure_error", "worker_failed"])
def test_generic_errors_cannot_silently_lose_parameter_selection(tmp_path, status):
    write_result(
        tmp_path,
        "missing",
        small_config(sigma=0.5),
        status=status,
        converged=False,
        error="FileNotFoundError: missing data",
    )
    write_result(
        tmp_path,
        "valid",
        small_config(sigma=0.2),
        status="converged",
        converged=True,
        clean_attempt_seconds=1.0,
    )
    with pytest.raises(ValueError, match="infrastructure.*missing"):
        manifest.winners(tmp_path)


@pytest.mark.parametrize("method", ["cg", "nystrom_pcg", "gd", "block_kaczmarz"])
@pytest.mark.parametrize("check_source", ["recurrence", "recompute"])
def test_capped_solve_reports_every_actual_qp_product(
    monkeypatch, method, check_source
):
    p = make_qp(0, 4)
    actual = 0
    original = p.hess

    def counted(x):
        nonlocal actual
        actual += 1 if x.ndim == 1 else x.shape[1]
        return original(x)

    monkeypatch.setattr(p, "hess", counted)
    row, _ = solvers.solve(
        p,
        small_config(
            inner=method,
            rank=4,
            block_size=1,
            max_inner=2,
            check_source=check_source,
        ),
    )
    assert not row["converged"]
    assert row["hessian_products_total"] == actual
    if method in ("cg", "nystrom_pcg") and check_source == "recurrence":
        assert row["certificate_products"] >= 1


@pytest.mark.parametrize(
    "method", ["cg", "gd", "nystrom_pcg", "block_kaczmarz", "pdhg"]
)
def test_full_worker_counts_calibration_and_setup_once(tmp_path, monkeypatch, method):
    from releases.accounting_v6.experiments import run

    p = make_qp(0, 4)
    actual = 0
    original = p.hess

    def counted(x):
        nonlocal actual
        actual += 1 if x.ndim == 1 else x.shape[1]
        return original(x)

    monkeypatch.setattr(p, "hess", counted)
    monkeypatch.setattr(run, "build", lambda *args: p)
    row = run.execute(
        small_config(
            inner=method,
            rank=4,
            block_size=1,
            max_inner=2,
            check_source="recurrence",
        ),
        tmp_path,
        tmp_path,
    )
    assert row["hessian_products_total"] == actual
    assert row["work_count_complete"]
    if method in ("cg", "gd", "block_kaczmarz"):
        assert row["calibration_hessian_products"] > 0
    if method == "nystrom_pcg":
        assert row["setup_hessian_products"] == 4
        assert row["calibration_hessian_products"] == 0


@pytest.mark.parametrize("max_inner", [1, 9, 10, 11, 25])
def test_pdhg_counts_actual_terminal_metrics(tmp_path, max_inner):
    row = execute(
        small_config(inner="pdhg", max_inner=max_inner, tolerance=1e-15),
        tmp_path,
        tmp_path,
    )
    # Diagnostics at k=0,10,... and the last iteration, plus a final certificate.
    checks = len(set(range(0, max_inner, 10)) | {max_inner - 1})
    assert row["hessian_products_total"] == max_inner + checks + 1


@pytest.mark.parametrize("family", ["lasso", "logistic"])
def test_regression_reports_design_passes_and_distinguishes_units(
    tmp_path, monkeypatch, family
):
    import scipy.sparse as sp

    from releases.accounting_v6.experiments import run
    from releases.accounting_v6.experiments.problems import make_regression

    p = make_regression(family, sp.eye(8, format="csr"), seed=0)
    p.provenance = json.dumps(dict(matrix={"source": "test"}))
    monkeypatch.setattr(run, "build", lambda *args: p)
    row = run.execute(
        small_config(family=family, inner="gd", max_inner=2), tmp_path, tmp_path
    )
    assert row["design_forward_products"] > 0
    assert row["design_transpose_products"] > 0
    if family == "logistic":
        assert math.isnan(row["hessian_products_total"])
        assert row["gradient_evaluations"] > 0
        assert row["hessian_product_unit"] == "not_applicable"


@pytest.mark.parametrize("family", ["lasso", "logistic"])
def test_recorded_design_products_match_actual_sparse_multiplications(
    tmp_path, monkeypatch, family
):
    import scipy.sparse as sp

    from releases.accounting_v6.experiments import run
    from releases.accounting_v6.experiments.problems import make_regression

    actual = dict(forward=0, transpose=0)

    class TrackedDesign:
        def __init__(self, matrix, direction="forward"):
            self.matrix, self.direction = matrix, direction
            self.shape = matrix.shape

        @property
        def T(self):
            direction = "transpose" if self.direction == "forward" else "forward"
            return TrackedDesign(self.matrix.T, direction)

        def __matmul__(self, vector):
            actual[self.direction] += 1 if vector.ndim == 1 else vector.shape[1]
            return self.matrix @ vector

    p = make_regression(family, sp.eye(8, format="csr"), seed=0)
    p.A = TrackedDesign(p.A)
    p.provenance = json.dumps(dict(matrix={"source": "test"}))
    monkeypatch.setattr(run, "build", lambda *args: p)
    row = run.execute(
        small_config(family=family, inner="gd", max_inner=2), tmp_path, tmp_path
    )
    assert row["design_forward_products"] == actual["forward"]
    assert row["design_transpose_products"] == actual["transpose"]


def test_native_backend_work_is_explicitly_unknown(tmp_path):
    row = execute(small_config(inner="osqp"), tmp_path, tmp_path)
    assert not row["work_count_complete"]
    assert math.isnan(row["hessian_products_total"])


def test_worker_labels_unexpected_exception_as_infrastructure(tmp_path, monkeypatch):
    from releases.accounting_v6.experiments import run

    config = small_config()
    path = tmp_path / "tasks.csv"
    write_csv(path, [dict(run_id=identity(config), config=json.dumps(config))])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--worker",
            "--manifest",
            str(path),
            "--task-id",
            "0",
            "--results-dir",
            str(tmp_path / "results"),
            "--memory-gb",
            "0",
        ],
    )

    def missing(*args):
        raise FileNotFoundError("missing matrix")

    monkeypatch.setattr(run, "execute", missing)
    with pytest.raises(FileNotFoundError):
        run.main()
    row = read_csv(tmp_path / "results" / identity(config) / "result.csv")[0]
    assert row["status"] == "infrastructure_error"
    assert row["error_type"] == "FileNotFoundError"


@pytest.mark.parametrize(
    "name, status",
    [
        ("FileNotFoundError", "infrastructure_error"),
        ("ConnectionResetError", "infrastructure_error"),
        ("EOFError", "infrastructure_error"),
        ("MemoryError", "resource_limit"),
        ("OSQPException", "unresolved_error"),
        ("ArithmeticError", "unresolved_error"),
        ("ValueError", "unresolved_error"),
    ],
)
def test_exceptions_are_classified_by_type(name, status):
    from releases.accounting_v6.experiments.outcomes import error_status

    assert error_status(name) == status


def test_worker_labels_a_solver_exception_as_a_method_outcome(tmp_path, monkeypatch):
    from releases.accounting_v6.experiments import run

    config = small_config()
    path = tmp_path / "tasks.csv"
    write_csv(path, [dict(run_id=identity(config), config=json.dumps(config))])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--worker",
            "--manifest",
            str(path),
            "--task-id",
            "0",
            "--results-dir",
            str(tmp_path / "results"),
            "--memory-gb",
            "0",
        ],
    )

    def curvature(*args):
        raise ArithmeticError("CG lost positive curvature")

    monkeypatch.setattr(run, "execute", curvature)
    with pytest.raises(ArithmeticError):
        run.main()
    row = read_csv(tmp_path / "results" / identity(config) / "result.csv")[0]
    assert row["status"] == "solver_error"
    assert row["error_type"] == "ArithmeticError"


@pytest.mark.parametrize(
    "row",
    [
        # speed_v2 fast lane: OSQP reported OSQP_NONCVX_ERROR (code 4) at side 100.
        dict(status="error", error="OSQPException: 4"),
        dict(status="solver_error", error_type="SolverFailure"),
        dict(status="resource_limit", error_type="MemoryError"),
    ],
)
def test_deterministic_solver_failures_do_not_block_selection(tmp_path, row):
    write_result(
        tmp_path,
        "failed",
        small_config(inner="osqp", sweep="baseline"),
        converged=False,
        **row,
    )
    write_result(
        tmp_path,
        "valid",
        small_config(),
        status="converged",
        converged=True,
        clean_attempt_seconds=1.0,
    )
    (winner,) = manifest.winners(tmp_path)
    assert winner["inner"] == "cg"


def test_legacy_error_without_a_type_still_requires_a_rerun():
    from releases.accounting_v6.experiments.outcomes import is_method_outcome

    assert not is_method_outcome(dict(status="error"))
    assert not is_method_outcome(dict(status="error", error="OSError: disk quota"))
    assert not is_method_outcome(
        dict(status="error", error="ValueError: bad curvature")
    )


@pytest.mark.parametrize("status", ["error", "solver_error", "infrastructure_error"])
@pytest.mark.parametrize(
    "name",
    ["ValueError", "KeyError", "ImportError", "JSONDecodeError", "UnknownException"],
)
def test_unrecognized_errors_require_investigation_before_selection(
    tmp_path, status, name
):
    write_result(
        tmp_path,
        "bad",
        small_config(sigma=0.5),
        status=status,
        converged=False,
        error_type=name,
        error=f"{name}: label checksum mismatch",
    )
    write_result(
        tmp_path,
        "good",
        small_config(sigma=0.2),
        status="converged",
        converged=True,
        clean_attempt_seconds=1.0,
    )
    # An infrastructure_error recorded by an earlier v6 build means rerun;
    # unrecognized exceptions recorded as error/solver_error need investigation.
    expected = (
        r"infrastructure events.*\['bad'\]"
        if status == "infrastructure_error"
        else r"investigate unresolved errors, then rerun: \['bad'\]"
    )
    with pytest.raises(ValueError, match=expected):
        manifest.winners(tmp_path)


def test_only_specific_backend_errors_are_method_failures():
    from releases.accounting_v6.experiments.outcomes import (
        error_status,
        is_method_outcome,
    )

    assert error_status("OSQPException", "4") == "solver_error"
    assert error_status("OSQPException", "2") == "unresolved_error"
    assert error_status("OSQPException", "5") == "resource_limit"
    assert (
        error_status("ArithmeticError", "CG lost positive curvature") == "solver_error"
    )
    assert (
        error_status("ArithmeticError", "unexpected arithmetic error")
        == "unresolved_error"
    )
    assert is_method_outcome(dict(status="solver_error", error="OSQPException: 4"))
    assert not is_method_outcome(dict(status="solver_error"))


@pytest.mark.parametrize("max_inner", [1, 9, 10, 11, 25])
def test_pdhg_correction_uses_actual_measured_work(tmp_path, max_inner):
    from plots.corrections import hessian_products, solve_phase_products

    config = small_config(inner="pdhg", max_inner=max_inner, tolerance=1e-15)
    row = execute(config, tmp_path, tmp_path)
    expected = row["hessian_products_total"]
    assert hessian_products(row, config)[0] == expected
    assert solve_phase_products(row, config)[0] == expected
    if max_inner == 9:
        assert expected == 12


def test_kaczmarz_expansion_rejects_previously_misclassified_errors():
    from releases.accounting_v6.experiments import plan

    configs = [
        small_config(side=40, inner="block_kaczmarz", block_size=1, sweep="sigma")
    ]
    _, screen = plan.split(configs)
    results = {identity(c): dict(status="time_limit", converged=False) for c in screen}
    results[identity(screen[0])] = dict(
        status="solver_error", error="ValueError: bad data"
    )
    with pytest.raises(ValueError, match="investigate unresolved errors"):
        plan.expand(configs, screen, results)


def test_osqp_memory_allocation_failure_does_not_block_selection(tmp_path):
    write_result(
        tmp_path,
        "osqp",
        small_config(inner="osqp", sweep="baseline"),
        status="error",
        converged=False,
        error="OSQPException: 5",
    )
    write_result(
        tmp_path,
        "valid",
        small_config(),
        status="converged",
        converged=True,
        clean_attempt_seconds=1.0,
    )
    (winner,) = manifest.winners(tmp_path)
    assert winner["inner"] == "cg"


def test_blocked_selection_separates_reruns_from_investigations(tmp_path):
    rows = dict(
        lost=dict(status="interrupted"),
        bug=dict(status="error", error="KeyError: 'sigma'"),
        clean=dict(status="no_clean_time"),
    )
    for name, row in rows.items():
        write_result(tmp_path, name, small_config(sigma=len(name) / 10), **row)
    write_result(
        tmp_path,
        "valid",
        small_config(sigma=0.9),
        status="converged",
        converged=True,
        clean_attempt_seconds=1.0,
    )
    with pytest.raises(ValueError) as error:
        manifest.winners(tmp_path)
    message = str(error.value)
    assert "3 runs cannot be ranked yet" in message
    assert (
        "infrastructure events (experiments.run --force): ['clean', 'lost']" in message
    )
    assert "investigate unresolved errors, then rerun: ['bug']" in message
