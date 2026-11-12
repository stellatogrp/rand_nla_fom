"""The cluster problems run in the format of the original studies (``studies``)."""

import csv
import json
import math
import shutil

import numpy as np
import pytest

from plots import studies as figures
from releases.regression_v9.experiments import theory
from releases.regression_v9.experiments.inner import Inner
from studies import study
from studies._io import read_rows
from studies.algorithms import Initialization, inexact_admm
from studies.inner import (
    CALIBRATION_DEPTH,
    EngineInner,
    Gmres,
    InnerFactory,
    calibrate,
    contraction,
    recommendations,
    stride_scan_limit,
)
from studies.problems import Instance

# The columns of the original studies' tables (numerical_experiments.study).
PARAMETERS = [
    "study",
    "seed",
    "n",
    "m",
    "ratio",
    "c",
    "gamma",
    "delta",
    "gd_step",
    "smoothness",
    "alpha",
    "c0",
    "generation_seconds",
    "truth_seconds",
    "truth_status",
    "truth_tolerance",
    "truth_value",
    "truth_stationarity",
    "truth_feasibility",
    "truth_planted_error",
    "smoothness_seconds",
    "projector_seconds",
    "factor_seconds",
    "parameter_seconds",
    "sigma_sweep_b",
    "b_sweep_sigma",
    "sigma_star",
    "sigma_star_expected",
    "sigma_upper_bound",
    "sigma_star_b1",
    "sigma_star_b2",
    "sigma_star_b5",
    "sigma_star_b10",
    "sigma_star_b20",
    "calibration_seconds",
    "c_in_seconds",
    "c_test_seconds",
    "b_star",
    "b_best",
    "b_star_exact",
    "b_star_oracle",
    "r0_norm",
    "r0_seconds",
    "dist0",
    "k_delta_plus_log10",
]
RESULTS = [
    "study",
    "seed",
    "method",
    "sweep",
    "sigma",
    "b",
    "sigma_is_optimal",
    "b_is_optimal",
    "b_exact_is_optimal",
    "in_grid",
    "converged",
    "stopped_by",
    "k_delta",
    "outer_iterations",
    "total_inner",
    "total_checks",
    "gradient_evaluations",
    "max_inner_per_outer",
    "mean_inner_per_outer",
    "runtime_seconds",
    "final_suboptimality",
    "final_suboptimality_z",
    "final_feasibility",
    "final_z_norm",
    "final_dist",
    "sum_gamma_z_sq",
    "thm31_bound",
    "thm31_ratio",
    "lemma32_min_ratio",
    "lemma32_checked",
    "beta0",
    "thm43_bound",
    "thm43_ratio",
    "thm42_bound",
    "thm42_ratio",
    "per_step_violations",
    "per_step_max_ratio",
    "per_step_violations_exact",
    "kappa_empirical",
    "rate_bound",
    "rate_observed_max",
    "rate_observed_median",
    "setup_seconds",
    "parameter_seconds",
    "calibration_seconds",
    "truth_seconds",
    "error",
]
HISTORIES = [
    "seed",
    "method",
    "sweep",
    "sigma",
    "b",
    "iteration",
    "suboptimality",
    "suboptimality_z",
    "feasibility",
    "z_norm",
    "inner",
    "cumulative_inner",
    "checks",
    "elapsed",
    "dist",
    "accepted_by",
    "residual_exact",
    "bound_surrogate",
    "bound_exact",
]
# Columns that do not depend on timing, compared between two computations.
DETERMINISTIC = [
    "method",
    "sweep",
    "sigma",
    "b",
    "converged",
    "stopped_by",
    "k_delta",
    "total_inner",
    "total_checks",
    "gradient_evaluations",
    "final_z_norm",
    "final_dist",
    "thm43_ratio",
]
SIZES = {
    "admm_qp": ["--side", "4"],
    "admm_lasso": ["--rows", "60", "--cols", "30"],
    "ppm_logistic": ["--rows", "60", "--cols", "30"],
}
SPEC = dict(
    kind="random_sparse", rows=60, cols=30, entries_per_row=6, distribution="normal"
)


def arguments(name, directory, *extra):
    return study.build_parser(name).parse_args(
        [
            "--instances",
            "2",
            "--workers",
            "1",
            "--calibration-repetitions",
            "5",
            "--results-dir",
            str(directory),
            "--allow-local",
            *SIZES[name],
            *extra,
        ]
    )


def qp(side=4, seed=0):
    return Instance("qp", seed, dict(side=side, kappa=1e4))


def regression(family, rows=60, cols=30, seed=0):
    spec = dict(SPEC, rows=rows, cols=cols, seed=11)
    settings = dict(dataset="tiny", orientation="native", matrix_source=spec)
    return Instance(family, seed, settings)


def header(path):
    with open(path, newline="") as stream:
        return next(csv.reader(stream))


def deterministic(rows):
    return [{k: row[k] for k in DETERMINISTIC if k in row} for row in rows]


def same(first, second):
    """Equal rows, with nan equal to nan."""
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        for key in a:
            x, y = a[key], b[key]
            if isinstance(x, float) and math.isnan(x):
                assert isinstance(y, float) and math.isnan(y), key
            else:
                assert x == y, (key, x, y)


@pytest.fixture(scope="module")
def qp_study(tmp_path_factory):
    directory = tmp_path_factory.mktemp("qp")
    return study.run_study("admm_qp", arguments("admm_qp", directory))


def test_fourier_solver_inverts_the_shifted_qp_hessian():
    rng = np.random.default_rng(0)
    for side in (4, 6):
        instance = qp(side)
        rhs = rng.standard_normal(instance.n)
        x = instance.exact_solver(0.3).solve(rhs)
        residual = instance.problem.Q @ x + 0.3 * x - rhs
        assert np.linalg.norm(residual) <= 1e-12 * np.linalg.norm(rhs)


@pytest.mark.parametrize("rows,cols", [(60, 30), (30, 60)])
@pytest.mark.parametrize("direct", [5000, 0])
def test_split_lasso_solvers_invert_the_shifted_hessian(rows, cols, direct):
    instance = regression("lasso", rows, cols)
    solver = instance.exact_solver(0.7, direct_max=direct)
    assert solver.kind == (
        "cg" if direct == 0 else "cholesky" if cols <= rows else "cholesky_woodbury"
    )
    rhs = np.random.default_rng(1).standard_normal(instance.n)
    x = solver.solve(rhs)
    residual = instance.problem.hess(x) + 0.7 * x - rhs
    assert np.linalg.norm(residual) <= 1e-11 * np.linalg.norm(rhs)


def test_qp_reference_is_the_planted_point():
    truth = qp(6, seed=3).ground_truth({})
    assert truth.status == "optimal"
    assert truth.planted_error < 1e-9
    assert truth.stationarity < 1e-10
    assert truth.feasibility == 0.0


@pytest.mark.parametrize("family", ["lasso", "logistic"])
def test_regression_references_agree_with_clarabel(family):
    instance = regression(family)
    smoothness = instance.smoothness()
    iterative = instance.ground_truth({}, smoothness=smoothness)
    clarabel = instance.ground_truth(dict(truth_solver="clarabel"))
    assert iterative.status == "optimal" and clarabel.status == "optimal"
    assert iterative.stationarity < 1e-10
    assert abs(iterative.value - clarabel.value) <= 1e-8 * (1 + abs(clarabel.value))
    assert np.linalg.norm(iterative.x - clarabel.x) < 1e-4


def test_gmres_meets_its_contraction_and_grows_its_basis():
    # Side 10: enough distinct eigenvalues that no breakdown comes before 20
    # steps, so the basis grows past its 16 preallocated vectors.
    instance = qp(10)
    c = 0.1
    alpha, c0 = theory.gmres_constants(1.0 / c)
    Q = instance.problem.Q

    def apply(v):
        return Q @ v + c * v

    rng = np.random.default_rng(0)
    rhs, x0 = rng.standard_normal(instance.n), rng.standard_normal(instance.n)
    r0 = rhs - apply(x0)
    solver = Gmres(apply, instance.n)
    solver.start(x0, rhs, r0)
    previous = np.linalg.norm(r0)
    for steps in range(1, 21):
        assert not solver.full
        solver.step()
        residual = np.linalg.norm(rhs - apply(solver.solution()))
        # Monotone, and within the Chebyshev bound ||r_l||^2 <= C_0 alpha^l ||r_0||^2.
        assert residual <= previous * (1 + 1e-10) + 1e-13
        assert residual**2 <= c0 * alpha**steps * np.linalg.norm(r0) ** 2 + 1e-24
        previous = residual
    assert len(solver.V) > 16
    assert previous < 1e-4 * np.linalg.norm(r0)
    assert solver.products == 20
    # A cycle at the exact solution, and restarts, are harmless.
    exact = instance.exact_solver(c).solve(rhs)
    solver.start(exact, rhs, np.zeros(instance.n))
    assert solver.full
    np.testing.assert_array_equal(solver.solution(), exact)
    short = Gmres(apply, instance.n, restart=3)
    short.start(x0, rhs, r0)
    for _ in range(3):
        short.step()
    assert short.full


@pytest.mark.parametrize(
    "method", ["gd", "agd", "cg", "nystrom_pcg", "block_kaczmarz", "cyclic_kaczmarz"]
)
def test_engine_solvers_started_from_the_residual_take_the_engine_steps(method):
    instance = qp(6)
    p, c = instance.problem, 0.1
    rng = np.random.default_rng(2)
    rhs, x0 = rng.standard_normal(instance.n), rng.standard_normal(instance.n)

    def engine():
        return Inner(p, c, method, np.random.default_rng(5), 8, 16)

    own = engine()
    own.start(x0, rhs)
    adapted = EngineInner(engine())
    adapted.start(x0, rhs, rhs - (p.Q @ x0 + c * x0))
    for _ in range(5):
        own.step()
        adapted.step()
    np.testing.assert_allclose(adapted.solution(), own.x, rtol=1e-10, atol=1e-12)


def test_factory_runs_start_from_the_same_state():
    instance = qp(6)
    settings = dict(inner="block_kaczmarz", block_size=8)
    factory = InnerFactory(instance, 0.1, settings)
    rhs = np.ones(instance.n)
    solutions = []
    for _ in range(2):
        solver = factory.new()
        solver.start(np.zeros(instance.n), rhs, rhs.copy())
        for _ in range(7):
            solver.step()
        solutions.append(solver.solution())
    np.testing.assert_array_equal(solutions[0], solutions[1])


def test_tables_have_the_original_columns_and_the_theorems_hold(qp_study):
    assert header(qp_study / "parameters.csv")[: len(PARAMETERS)] == PARAMETERS
    assert header(qp_study / "results.csv") == RESULTS
    assert header(qp_study / "histories.csv") == HISTORIES
    assert (qp_study / "curves.csv").exists()
    results = read_rows(qp_study / "results.csv")
    assert {r["method"] for r in results} == {"exact", "near_exact", "inexact"}
    assert any(r["sigma_is_optimal"] for r in results)
    summary = read_rows(qp_study / "summary.csv")
    for row in summary:
        if row["method"] == "inexact" and row["sigma"] < 1:
            assert row["n_converged"] == row["n_runs"] == 2
            assert row["thm31_ratio_max"] <= 1.0
            assert row["lemma32_min_ratio_min"] >= 1.0 - 1e-9
            assert row["thm43_ratio_max"] <= 1.0
            assert row["thm42_ratio_max"] <= 1.0
            assert row["per_step_violations_total"] == 0
            assert row["per_step_violations_exact_total"] == 0
            assert row["rate_observed_max_max"] <= row["rate_bound_median"]
    parameters = read_rows(qp_study / "parameters.csv")
    assert all(p["truth_planted_error"] < 1e-9 for p in parameters)
    assert all(p["exact_solver"] == "fft" for p in parameters)
    assert all(math.isclose(p["alpha"], 0.28802, rel_tol=1e-4) for p in parameters)


@pytest.mark.parametrize(
    "name,extra",
    [("admm_lasso", ["--inner", "cg"]), ("ppm_logistic", [])],
)
def test_regression_studies_produce_the_original_tables(tmp_path, name, extra):
    args = arguments(
        name, tmp_path, "--sigmas", "0.05", "0.9", "--bs", "1", "3", *extra
    )
    directory = study.run_study(name, args)
    assert header(directory / "results.csv") == RESULTS
    assert header(directory / "histories.csv") == HISTORIES
    results = read_rows(directory / "results.csv")
    assert all(r["converged"] for r in results)
    inexact = [r for r in results if r["method"] == "inexact"]
    assert all(
        r["thm43_ratio"] <= 1.0 and r["per_step_violations"] == 0 for r in inexact
    )
    curves = read_rows(directory / "curves.csv")
    baseline = "gd" if name == "ppm_logistic" else "exact"
    assert any(r["method"] == baseline for r in results)
    if name == "ppm_logistic":
        assert any(r["method"] == "gd" and r["axis"] == "inner" for r in curves)
        assert all(math.isnan(r["final_feasibility"]) for r in results)


def test_staged_workflow_reproduces_the_local_pool(tmp_path, qp_study):
    args = arguments("admm_qp", tmp_path)
    settings = study.settings_from(args, "admm_qp")
    directory = study.study_directory(settings)
    directory.mkdir(parents=True)
    # The same calibration gives the same recommended strides, hence the same runs.
    shutil.copy(qp_study / "calibration.json", directory / "calibration.json")
    for index in (1, 0):
        args = arguments(
            "admm_qp", tmp_path, "--stage", "task", "--task-index", str(index)
        )
        study.run_task("admm_qp", args)
    study.run_collect("admm_qp", arguments("admm_qp", tmp_path, "--stage", "collect"))
    pooled = read_rows(qp_study / "results.csv")
    staged = read_rows(directory / "results.csv")
    same(deterministic(pooled), deterministic(staged))
    assert [p["seed"] for p in read_rows(directory / "parameters.csv")] == [0, 1]


def test_an_interrupted_task_resumes_with_the_next_configuration(tmp_path):
    def task():
        args = arguments(
            "admm_qp",
            tmp_path,
            "--sigmas",
            "0.5",
            "--bs",
            "1",
            "--stage",
            "task",
            "--task-index",
            "0",
        )
        study.run_task("admm_qp", args)

    task()
    part = next((tmp_path).glob("*/parts/0"))
    runs = sorted((part / "runs").glob("*.results.csv"))
    finished = [read_rows(path)[0] for path in runs]
    parameters = (part / "setup/parameters.csv").read_text()
    for path in runs[-2:]:
        path.unlink()
    task()
    resumed = [
        read_rows(path)[0] for path in sorted((part / "runs").glob("*.results.csv"))
    ]
    same(deterministic(finished), deterministic(resumed))
    assert (part / "setup/parameters.csv").read_text() == parameters


def test_separate_runs_reproduce_the_pool_and_report_what_is_missing(
    tmp_path, qp_study
):
    def args(*extra):
        return arguments("admm_qp", tmp_path, *extra)

    settings = study.settings_from(args(), "admm_qp")
    directory = study.study_directory(settings)
    directory.mkdir(parents=True)
    shutil.copy(qp_study / "calibration.json", directory / "calibration.json")
    study.run_calibration("admm_qp", args("--stage", "calibrate"))
    planned = read_rows(directory / "configurations.csv")
    same(planned, read_rows(qp_study / "parts/0/setup/configurations.csv"))
    # Every run its own task, the last instance's in reverse order; one index
    # past the list is a no-op, as manifests reserve room for it.
    for seed in (0, 1):
        order = range(len(planned)) if seed == 0 else reversed(range(len(planned)))
        for index in order:
            study.run_run(
                "admm_qp",
                args(
                    "--stage",
                    "run",
                    "--task-index",
                    str(seed),
                    "--configuration",
                    str(index),
                ),
            )
    beyond = args("--stage", "run", "--task-index", "0", "--configuration", "99")
    study.run_run("admm_qp", beyond)
    assert not (directory / "parts/0/runs/099.results.csv").exists()
    study.run_collect("admm_qp", args("--stage", "collect"))
    same(
        deterministic(read_rows(qp_study / "results.csv")),
        deterministic(read_rows(directory / "results.csv")),
    )
    assert not (directory / "missing.csv").exists()
    # A run that has not finished delays nothing: the others are collected.
    (directory / "parts/1/runs/003.results.csv").unlink()
    with pytest.raises(SystemExit, match="unfinished"):
        study.run_collect("admm_qp", args("--stage", "collect"))
    study.run_collect("admm_qp", args("--stage", "collect", "--allow-partial"))
    missing = read_rows(directory / "missing.csv")
    assert [(m["seed"], m["configuration"]) for m in missing] == [(1, 3)]
    assert len(read_rows(directory / "results.csv")) == 2 * len(planned) - 1


def test_concurrent_setups_of_an_instance_agree_on_the_stored_one(tmp_path):
    settings = study.settings_from(arguments("admm_qp", tmp_path), "admm_qp")
    settings.update(c_in=1e-6, c_test=1e-6, calibration_seconds=0.0)
    part = tmp_path / "part"
    first = study.prepare_instance("admm_qp", 0, settings, part)
    stored = (part / "setup/parameters.csv").read_text()
    # A task that computed its own setup while the first one was storing:
    # the store fails, and it adopts the stored parameters.
    assert not study._store_setup(
        part, dict(first.parameters, truth_seconds=-1.0), first.truth, []
    )
    second = study.prepare_instance("admm_qp", 0, settings, part)
    assert (part / "setup/parameters.csv").read_text() == stored
    assert second.parameters["truth_seconds"] == first.parameters["truth_seconds"]
    assert [study._signature(c) for c in second.configurations] == [
        study._signature(c) for c in first.configurations
    ]
    assert not list(part.glob(".setup.*"))


def test_a_study_directory_refuses_other_settings(tmp_path):
    study.run_calibration(
        "admm_qp", arguments("admm_qp", tmp_path, "--stage", "calibrate")
    )
    with pytest.raises(SystemExit, match="other settings"):
        study.run_task(
            "admm_qp",
            arguments(
                "admm_qp",
                tmp_path,
                "--delta",
                "1e-8",
                "--stage",
                "task",
                "--task-index",
                "0",
            ),
        )


def test_hard_stops_and_the_near_exact_baseline_without_time_limit(tmp_path):
    args = arguments(
        "admm_qp", tmp_path, "--sigmas", "0.5", "--bs", "1", "--time-limit", "0"
    )
    settings = study.settings_from(args, "admm_qp")
    settings.update(c_in=1e-6, c_test=1e-6, calibration_seconds=0.0)
    rows = study.run_instance(("admm_qp", 0, settings))["results"]
    by_method = {r["method"]: r for r in rows if r["sweep"] != "b"}
    assert by_method["near_exact"]["stopped_by"] == "converged"
    settings.update(near_exact_time_limit=0.0)
    capped = study.run_instance(("admm_qp", 0, settings))["results"]
    near = [r for r in capped if r["method"] == "near_exact"]
    assert near[0]["stopped_by"] == "time_limit"
    assert by_method["exact"]["stopped_by"] == "converged"
    inexact = [r for r in rows if r["method"] == "inexact"]
    assert inexact and all(r["stopped_by"] == "time_limit" for r in inexact)

    instance = qp(4)
    instance.use_smoothness(instance.smoothness())
    factory = InnerFactory(instance, 0.1, dict(inner="gmres"))
    trace = inexact_admm(instance, 0.1, factory.new(), 1.0, 2, 1e-9, max_total_inner=5)
    assert trace.stopped_by == "max_total_inner"
    assert trace.accepted_by[-1] == "max_total_inner"
    assert 5 <= sum(trace.inner) < 5 + 2


@pytest.mark.parametrize("mode", ["fixed", "gaussian"])
def test_inner_initializations_reset_every_start(mode):
    instance = qp(4)
    instance.use_smoothness(instance.smoothness())
    factory = InnerFactory(instance, 0.1, dict(inner="cg"))
    starts = Initialization(dict(inner_initialization=mode, gaussian_scale=1.0), 0)
    warm = inexact_admm(instance, 0.1, factory.new(), 0.5, 1, 1e-9)
    reset = inexact_admm(
        instance, 0.1, factory.new(), 0.5, 1, 1e-9, initialization=starts
    )
    assert warm.converged and reset.converged
    assert warm.e0_norm[1:] != reset.e0_norm[1:]


def test_solvers_without_constants_skip_the_recommended_runs(tmp_path):
    args = arguments(
        "admm_qp", tmp_path, "--inner", "agd", "--sigmas", "0.5", "--bs", "1", "2"
    )
    directory = study.run_study("admm_qp", args)
    parameters = read_rows(directory / "parameters.csv")
    assert all(p["contraction"] == "none" for p in parameters)
    assert all(math.isnan(p["sigma_star"]) for p in parameters)
    rows = read_rows(directory / "results.csv")
    assert not any(r["sigma_is_optimal"] for r in rows)
    assert {r["b"] for r in rows if r["sweep"] == "b"} == {1, 2}
    inexact = [r for r in rows if r["method"] == "inexact"]
    assert all(math.isnan(r["thm43_ratio"]) for r in inexact)
    assert all(r["thm31_ratio"] <= 1.0 for r in inexact)
    # The figures mark no recommended value and draw no contraction bounds.
    loaded = figures.load(directory.name, directory.parent)
    assert loaded.recommended_b() == (None, None)
    assert figures._recommendation_handles(loaded, "b") == ([], [])
    assert figures._recommendation_handles(loaded, "sigma") == ([], [])
    written = figures.make(directory.name, directory.parent)
    assert len(written) == 2 * (len(figures.FIGURES) - len(figures.BOUND_FIGURES))


def test_block_kaczmarz_uses_its_expected_bound_and_an_interior_stride():
    instance = qp(40)
    instance.use_smoothness(1.0)
    settings = dict(
        inner="block_kaczmarz", block_size=32, sigma_sweep_b=1, b_sweep_sigma=0.1
    )
    alpha, c0, kind = contraction(instance, 0.1, 10.0, settings)
    assert kind == "expected" and 1 - 1e-5 < alpha < 1
    recommended = recommendations(alpha, c0, kind, settings, 1e-4, 1e-3)
    assert recommended["sigma_star"] == theory.optimal_sigma(
        alpha, c0, 1, deterministic=False
    )
    # Remark 4.4's minimizer lies beyond the original's scan to 1000; it must
    # be found, not reported as the scan's end point.
    b_exact = recommended["b_star_exact"]
    assert 1000 < b_exact < stride_scan_limit(recommended["b_star"])

    def cost(b):
        return theory.b_cost_per_outer(b, 1e-3, 1e-4, 0.1, c0, alpha, relaxed=False)

    assert cost(b_exact) <= min(cost(b_exact - 1), cost(b_exact + 1))


def test_an_expected_bound_leaves_the_deterministic_checks_empty(tmp_path):
    args = arguments(
        "admm_qp",
        tmp_path,
        "--inner",
        "block_kaczmarz",
        "--block-size",
        "8",
        "--sigmas",
        "0.5",
        "--bs",
        "1",
        "--near-exact-tolerance",
        "1e-6",
        "--max-total-inner",
        "20000",
    )
    settings = study.settings_from(args, "admm_qp")
    settings.update(c_in=1e-6, c_test=1e-6, calibration_seconds=0.0)
    out = study.run_instance(("admm_qp", 0, settings))
    assert out["parameters"]["contraction"] == "expected"
    inexact = [r for r in out["results"] if r["method"] == "inexact"]
    assert inexact
    for row in inexact:
        assert math.isnan(row["thm43_ratio"]) and math.isnan(row["per_step_violations"])
        assert math.isfinite(row["thm42_ratio"])
    assert all(row["bound_surrogate"] == "nan" for row in out["histories"])


def test_the_ppm_constants_follow_the_inner_step(tmp_path):
    instance = regression("logistic")
    instance.use_smoothness(instance.smoothness())
    for factor, root in ((1.0, 10 / 12), (0.5, 11 / 12)):
        alpha, c0, kind = contraction(instance, math.nan, 10.0, dict(gd_factor=factor))
        assert kind == "deterministic" and c0 == 121.0
        assert math.isclose(alpha, root**2)
    assert contraction(instance, math.nan, 10.0, dict(gd_factor=2.0))[2] == "none"
    args = arguments(
        "ppm_logistic", tmp_path, "--gd-factor", "0.5", "--sigmas", "0.05", "0.5"
    )
    settings = study.settings_from(args, "ppm_logistic")
    settings.update(c_in=1e-6, c_test=1e-6, calibration_seconds=0.0)
    rows = study.run_instance(("ppm_logistic", 0, settings))["results"]
    inexact = [r for r in rows if r["method"] == "inexact"]
    assert inexact and all(r["converged"] for r in inexact)
    assert all(r["per_step_violations"] == 0 for r in inexact)
    assert all(r["thm43_ratio"] <= 1.0 for r in inexact)


def test_a_dead_worker_restarts_the_pool_and_every_instance_finishes(tmp_path):
    args = arguments(
        "admm_qp",
        tmp_path,
        "--instances",
        "3",
        "--workers",
        "2",
        "--sigmas",
        "0.5",
        "--bs",
        "1",
        "--no-aggregate",
    )
    args.crash_seed = 1
    directory = study.run_study("admm_qp", args)
    seeds = sorted(int(r["seed"]) for r in read_rows(directory / "parameters.csv"))
    assert seeds == [0, 1, 2]


def test_computing_stages_need_an_allocation_or_allow_local(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("sys.argv", ["run_admm_qp", "--side", "4"])
    with pytest.raises(SystemExit):
        study.main("admm_qp")


def test_the_original_figures_are_drawn(qp_study):
    written = figures.make(qp_study.name, qp_study.parent)
    assert len(written) == 2 * len(figures.FIGURES)
    assert all(path.exists() for path in written)


def test_collecting_needs_neither_the_data_nor_a_new_study(tmp_path):
    import json

    import scipy.sparse as sp

    data = tmp_path / "data"
    data.mkdir()
    design = sp.random(50, 20, density=0.3, random_state=4, format="csr")
    sp.save_npz(data / "design.npz", design)
    spec = tmp_path / "sources.json"
    spec.write_text(json.dumps(dict(local=dict(kind="local", path="design.npz"))))
    common = [
        "--dataset-spec",
        str(spec),
        "--dataset",
        "local",
        "--data-dir",
        str(data),
        "--instances",
        "1",
        "--sigmas",
        "0.5",
        "--bs",
        "1",
        "--calibration-repetitions",
        "3",
        "--results-dir",
        str(tmp_path / "results"),
        "--allow-local",
    ]
    parser = study.build_parser("admm_lasso")
    task = parser.parse_args([*common, "--stage", "task", "--task-index", "0"])
    directory = study.run_task("admm_lasso", task)
    stored = json.loads((directory / "settings.json").read_text())
    assert "sha256" in stored["identity"]["matrix_source"]
    (data / "design.npz").unlink()
    collecting = parser.parse_args([*common, "--stage", "collect"])
    study.run_collect("admm_lasso", collecting)
    assert header(directory / "results.csv") == RESULTS
    # A lost run file makes its instance unfinished rather than silently short.
    next((directory / "parts/0/runs").glob("*.results.csv")).unlink()
    with pytest.raises(SystemExit, match="unfinished"):
        study.run_collect("admm_lasso", collecting)
    other = parser.parse_args([*common, "--stage", "collect", "--name", "absent"])
    with pytest.raises(SystemExit, match="holds no study"):
        study.run_collect("admm_lasso", other)


def test_the_walltime_figures_charge_s_star_its_timings_and_sigma_star_nothing(
    tmp_path,
):
    """sigma* needs only alpha and C_0; s* is charged the timings it needs."""

    def row(sigma_star=False, b_best=0.0, b_exact=0.0):
        return dict(
            sigma_is_optimal=sigma_star,
            b_optimal_fraction=b_best,
            b_exact_optimal_fraction=b_exact,
        )

    assert not figures._charged_for_s_star(row(sigma_star=True))
    assert figures._charged_for_s_star(row(b_best=1.0))
    assert figures._charged_for_s_star(row(b_exact=1.0))
    assert not figures._charged_for_s_star(row())

    class Recorded:
        directory = tmp_path / "recorded"

    Recorded.directory.mkdir()
    (Recorded.directory / "calibration.json").write_text(
        json.dumps(dict(c_in=2e-3, c_test=1e-3, measurement_seconds=0.25))
    )
    assert figures._measurement_seconds(Recorded) == 0.25

    # Before the calibration recorded it: the repetitions of the campaign task.
    args = tmp_path / "study.args"
    args.write_text("--side 4\n--calibration-repetitions 7\n")
    manifest = tmp_path / "calibrate.csv"
    manifest.write_text(f"task,stage,module,args\n0,calibrate,admm_qp,{args}\n")

    class Legacy:
        directory = tmp_path / "legacy"

    Legacy.directory.mkdir()
    command = ["campaign.py", "task", "--manifest", str(manifest), "--task-id", "0"]
    (Legacy.directory / "calibration.json").write_text(
        json.dumps(dict(c_in=2e-3, c_test=1e-3, command=command))
    )
    assert figures._measurement_seconds(Legacy) == pytest.approx((7 + 3) * 3e-3)


class _CountingSolver:
    """A stand-in inner solver that only counts what the calibration asks of it."""

    full = False

    def __init__(self):
        self.steps = self.starts = 0

    def start(self, x, rhs, w):
        self.starts += 1

    def step(self):
        self.steps += 1

    def solution(self):
        return np.zeros(3)


class _CountingFactory:
    def __init__(self, method):
        self.method, self.solvers = method, []

    def new(self, restart=None):
        self.solvers.append(_CountingSolver())
        return self.solvers[-1]


class _Problem:
    n, algorithm = 3, "admm"

    def gradient(self, x):
        return x

    def project(self, x):
        return x


@pytest.mark.parametrize("method", ["cg", "gd", "agd", "nystrom_pcg", "block_kaczmarz"])
def test_calibration_times_every_step_of_solvers_whose_steps_cost_alike(method):
    factory = _CountingFactory(method)
    c_in, c_test, measured = calibrate(_Problem(), 1.0, 1.0, factory, repetitions=20)
    (solver,) = factory.solvers
    # 23 samples (3 of warm-up), every one a timed step: no untimed steps.
    assert solver.steps == 23 and solver.starts == math.ceil(23 / CALIBRATION_DEPTH)
    assert c_in >= 0 and c_test >= 0 and measured > 0


def test_calibration_times_gmres_at_its_krylov_depth():
    factory = _CountingFactory("gmres")
    calibrate(_Problem(), 1.0, 1.0, factory, repetitions=20)
    (solver,) = factory.solvers
    # One timed step at dimension CALIBRATION_DEPTH per sample, as originally.
    assert solver.steps == 23 * CALIBRATION_DEPTH and solver.starts == 23


def test_the_study_calibrates_on_its_first_instance_and_records_the_measurements(
    tmp_path,
):
    directory = study.run_calibration(
        "admm_qp", arguments("admm_qp", tmp_path, "--stage", "calibrate")
    )
    record = study._calibration(directory)
    assert record["repetitions"] == 5
    assert 0 < record["measurement_seconds"] <= record["calibration_seconds"]
    # The first instance was set up with these very costs: nothing was rebuilt.
    first = read_rows(directory / "parts" / "0" / "setup" / "parameters.csv")[0]
    assert first["c_in_seconds"] == pytest.approx(record["c_in"])
    assert first["c_test_seconds"] == pytest.approx(record["c_test"])
    second = read_rows(directory / "parts" / "1" / "setup" / "parameters.csv")[0]
    assert second["c_in_seconds"] == pytest.approx(record["c_in"])
