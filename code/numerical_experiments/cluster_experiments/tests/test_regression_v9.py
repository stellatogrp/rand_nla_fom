"""Verify common-certificate stopping without changing inner numerical updates."""

import numpy as np
import pytest
import scipy.sparse as sp

from releases.regression_v9.experiments.problems import make_regression
from releases.regression_v9.experiments.solvers import solve
from releases.regression_v9.experiments.tuning import resolve


def lasso():
    return make_regression(
        "lasso",
        sp.eye(1),
        0,
        targets=np.array([1.0]),
        normalization="none",
        lambda_value=0.1,
    )


def config(**changes):
    return dict(
        dict(
            inner="cg",
            seed=0,
            ratio=2e6,
            sigma=0.2,
            stride=1,
            check_source="recurrence",
            tolerance=1e-6,
            max_inner=1,
            time_limit=30,
            terminal_policy="common_certificate",
        ),
        **changes,
    )


def test_common_certificate_accepts_a_valid_lasso_point_without_graph_gate():
    x = np.array([0.9000001, 0.0])
    p = lasso()
    assert max(p.metrics(x)[k] for k in ("stationarity", "feasibility")) < 1e-6
    c = p.L / config()["ratio"]
    assert np.linalg.norm(x - p.project(x - p.gradient(x) / c)) > 1e-6
    result, _ = solve(lasso(), config(), state={"x": x})
    old, _ = solve(lasso(), config(terminal_policy="legacy_graph_gate"), state={"x": x})
    assert result["status"] == "converged" and result["total_inner"] == 0
    assert old["status"] != "converged"
    assert result["certificate_products"] == result["certificate_checks"] == 1


def test_certificate_rejects_a_nonoptimal_point():
    result, _ = solve(lasso(), config())
    assert not result["converged"]
    assert max(result["stationarity"], result["feasibility"]) > 1e-6
    assert result["certificate_products"] > 0


def test_auditing_does_not_change_gd_updates_before_stopping():
    def problem():
        return make_regression(
            "logistic",
            sp.eye(2),
            0,
            targets=np.array([-1.0, 1.0]),
            normalization="none",
        )

    cfg = config(inner="gd", ratio=100.0, max_inner=3, sigma=0.99)
    result, _ = solve(problem(), cfg)
    old, _ = solve(problem(), dict(cfg, terminal_policy="legacy_graph_gate"))
    assert result["total_inner"] == old["total_inner"] == 3
    assert result["objective"] == pytest.approx(old["objective"])
    assert result["stationarity"] == pytest.approx(old["stationarity"])


@pytest.mark.parametrize("stride", [0, -1, 1.5, True])
def test_certificate_stride_must_be_a_positive_integer(stride):
    with pytest.raises(ValueError, match="certificate_stride"):
        solve(lasso(), config(certificate_stride=stride))


def test_old_check_cost_model_cannot_silently_choose_new_policy_parameters():
    with pytest.raises(ValueError, match="new model calibration"):
        resolve(lasso(), config(sigma="optimal"))
