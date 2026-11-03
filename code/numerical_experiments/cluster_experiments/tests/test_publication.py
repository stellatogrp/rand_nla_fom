import json
import math

import pytest

from plots.publication import summarize, tables
from plots.work import SCOPE, primary_work


def result(family="lasso", inner="pdhg", **changes):
    config = dict(family=family, inner=inner, dataset="test", seed=0, phase="evaluate")
    return dict(
        config=json.dumps(config),
        run_id="test",
        converged="True",
        status="converged",
        setup_policy="independent",
        timing_scope="independent_final_attempt",
        clean_attempt_seconds="2",
        work_count_complete="True",
        work_count_scope=SCOPE,
        hessian_products_total="3",
        calibration_hessian_products="0",
        design_forward_products="190",
        design_transpose_products="185",
        **changes,
    )


def test_regression_primary_work_uses_design_products_not_metric_hessians():
    for family in ["lasso", "logistic"]:
        row = result(family)
        work = primary_work(row, json.loads(row["config"]))
        assert work["value"] == 375
        assert work["evidence"] == "measured"
        assert work["unit"] == "design_forward_plus_transpose_vector_products"
    qp = result("qp")
    assert primary_work(qp, json.loads(qp["config"]))["value"] == 3


def test_backend_counts_and_incomplete_scope_stay_unknown():
    row = result(inner="osqp")
    assert math.isnan(primary_work(row, json.loads(row["config"]))["value"])
    row = result()
    row["work_count_complete"] = "False"
    assert math.isnan(primary_work(row, json.loads(row["config"]))["value"])


def test_publication_tables_never_pool_estimates_with_measured_totals():
    measured = result("qp", inner="gd")
    historical = result("qp", inner="gd")
    historical.update(run_id="old", work_count_complete="False")
    historical.pop("calibration_hessian_products")
    historical.update(inner_hessian_products="2", checks="10")
    historical.pop("hessian_products_total")
    unknown = result(inner="osqp")
    unknown["run_id"] = "native"
    parts = tables([measured, historical, unknown])
    assert (
        len(parts["measured"])
        == len(parts["historical_estimates"])
        == len(parts["unknown"])
        == 1
    )
    summary = summarize(parts["measured"] + parts["historical_estimates"])
    assert len(summary) == 2
    assert {r["evidence"] for r in summary} == {"measured", "historical_estimate"}


def test_censored_runs_are_visible_but_not_work_to_convergence():
    row = result()
    row.update(converged="False", status="max_inner")
    summary = summarize(tables([row])["measured"])[0]
    assert summary["runs"] == 1 and summary["converged_runs"] == 0
    assert math.isnan(summary["median_work_to_convergence"])


def test_pilots_are_excluded_and_duplicate_input_runs_are_rejected():
    row = result()
    cfg = json.loads(row["config"])
    cfg["phase"] = "pilot"
    row["config"] = json.dumps(cfg)
    assert len(tables([row])["excluded"]) == 1
    assert len(tables([row], include_pilots=True)["measured"]) == 1
    with pytest.raises(ValueError, match="duplicate run identity"):
        tables([row, row])
