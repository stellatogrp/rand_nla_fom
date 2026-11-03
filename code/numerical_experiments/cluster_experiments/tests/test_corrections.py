import json
import math

from plots.accounting import finite_time, main_eligible
from plots.corrections import (
    corrected,
    hessian_products,
    node_family,
    solve_phase_products,
    timing,
)


def config(**changes):
    return dict(
        dict(family="qp", dataset="synthetic", side=40, inner="gd", seed=0),
        **changes,
    )


def test_v3_final_attempt_is_clean_despite_an_earlier_hard_kill():
    row = dict(
        total_seconds="nan",
        attempt_worker_wall_seconds="12.5",
        setup_accounting_complete="False",
        primary_eligible="False",
        timing_scope="independent_attempts_including_retries",
    )
    assert timing(row) == (12.5, "final_attempt")
    assert main_eligible(row) and finite_time(row)


def test_resumed_legacy_run_has_no_clean_time():
    row = dict(
        total_seconds="30",
        solve_seconds="25",
        generation_seconds="1",
        process_seconds="9",
    )
    seconds, basis = timing(row)
    assert math.isnan(seconds) and basis == "resumed_from_checkpoint"
    row["process_seconds"] = "30"
    assert timing(row) == (30.0, "single_attempt")


def test_historical_work_is_reconstructed_with_an_explicit_scope():
    # Legacy GD smoke run: 91 counted products plus 544 uncounted checks.
    legacy = dict(inner_hessian_products="91", checks="544")
    assert hessian_products(legacy, config()) == (
        635.0,
        "estimated_inner_plus_checks_excludes_final_diagnostics",
    )
    v4 = dict(hessian_products_total="635", check_products="544")
    assert hessian_products(v4, config()) == (
        635.0,
        "partial_recorded_work_excludes_final_diagnostics",
    )
    assert math.isnan(hessian_products(dict(total_inner="25"), config(inner="pdhg"))[0])
    assert math.isnan(hessian_products({}, config(inner="osqp"))[0])
    assert math.isnan(hessian_products(legacy, config(family="logistic"))[0])
    assert math.isnan(hessian_products({}, config())[0])


def test_measured_v5_work_keeps_both_scopes():
    row = dict(
        hessian_products_total="652",
        calibration_hessian_products="17",
        work_count_complete="True",
        work_count_scope="problem_operators_after_build_including_calibration",
    )
    assert hessian_products(row, config()) == (
        652.0,
        "measured_after_build_including_calibration",
    )
    # Subtracting calibration gives the scope shared with every older release.
    assert solve_phase_products(row, config()) == (635.0, "measured_solve_phase")
    assert math.isnan(hessian_products(row, config(family="logistic"))[0])


def test_pdhg_preserves_measurements_including_lasso_design_units():
    # v5 counts LASSO PDHG iterations only as design products.
    row = dict(
        total_inner="25",
        hessian_products_total="3",
        work_count_complete="True",
        work_count_scope="problem_operators_after_build_including_calibration",
    )
    assert hessian_products(row, config(family="lasso", inner="pdhg"))[0] == 3
    assert solve_phase_products(row, config(family="lasso", inner="pdhg"))[0] == 3


def test_node_family_prefers_recorded_cpu_model():
    assert node_family(dict(hostname="della-i14n7")) == "della-i (hostname)"
    assert node_family(dict(cpu_model="Xeon", hostname="della-i1")) == "Xeon"


def test_v3_nystrom_optimal_is_labeled_a_loose_bound():
    row = dict(
        config=json.dumps(config(inner="nystrom_pcg", sigma="optimal")),
        recommendation_kind="bound_cost_model",
    )
    assert corrected(row)["model_label"] == "loose_bound_not_model_optimal"


def test_historical_recurrence_checks_are_not_counted_as_products():
    row = dict(inner_hessian_products="4", checks="100")
    value, basis = solve_phase_products(
        row, config(inner="cg", check_source="recurrence")
    )
    assert math.isnan(value)
    assert basis == "historical_recurrence_products_not_recorded"


def test_unmeasured_pdhg_counts_remain_unknown_for_any_stop_schedule():
    for iterations in [1, 9, 10, 11, 25]:
        value, basis = solve_phase_products(
            dict(total_inner=iterations), config(inner="pdhg")
        )
        assert math.isnan(value)
        assert basis == "historical_pdhg_diagnostics_not_recorded"


def test_lasso_pdhg_hessian_count_is_flagged_and_design_counts_exported():
    row = dict(
        config=json.dumps(config(family="lasso", inner="pdhg")),
        hessian_products_total="19",
        design_forward_products="190",
        design_transpose_products="190",
        work_count_complete="True",
        work_count_scope="problem_operators_after_build_including_calibration",
    )
    out = corrected(row)
    # Only metric evaluations: comparing it with ADMM Hessian counts misleads.
    assert out["work_count_basis"].endswith("metrics_only_compare_design_products")
    assert out["design_forward_products"] == 190
    qp = corrected(dict(row, config=json.dumps(config(inner="pdhg"))))
    assert qp["work_count_basis"] == "measured_after_build_including_calibration"
