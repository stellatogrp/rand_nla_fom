"""Validate numerical invariance, theoretical envelopes, and figure accounting."""

import json
from pathlib import Path

import numpy as np
import pytest

from figure_campaign.prepare import configurations
from plots.main_summary import common_curve, timing_parts
from releases.accounting_v7.experiments.problems import make_qp as old_problem
from releases.accounting_v7.experiments.solvers import solve as old_solve
from releases.figure_v8.experiments.problems import make_qp
from releases.figure_v8.experiments.solvers import solve
from releases.figure_v8.experiments.tuning import constants


@pytest.mark.parametrize("method", ["gd", "cg"])
@pytest.mark.parametrize("mode", ["warm", "fixed", "gaussian"])
def test_trace_preserves_updates_and_meets_verified_inner_contraction(method, mode):
    config = dict(
        seed=3,
        inner=method,
        ratio=10.0,
        sigma=0.2,
        stride=3,
        inner_initialization=mode,
        check_source="recurrence",
        tolerance=1e-6,
        max_inner=10000,
        time_limit=30,
    )
    problem = make_qp(seed=3, side=4)
    events = []
    result, history = solve(problem, config, trace=events.append)
    reference, _ = old_solve(old_problem(seed=3, side=4), config)
    assert result["status"] == reference["status"] == "converged"
    assert result["total_inner"] == reference["total_inner"]
    assert result["outer"] == reference["outer"]
    assert result["solution_error"] == pytest.approx(reference["solution_error"])
    assert len([e for e in events if e["outer_boundary"]]) == len(history)
    assert all(e["gradient_source"] == "true" for e in events if e["outer_boundary"])
    alpha, c0, deterministic = constants(problem, problem.L / config["ratio"], method)
    assert deterministic
    for event in events:
        ratio = (event["graph_residual"] / event["initial_graph_residual"]) ** 2
        assert ratio <= c0 * alpha ** event["local_inner"] * (1 + 1e-8) + 1e-24
    assert result["hessian_products_total"] > reference["hessian_products_total"]


def test_matched_initialization_grid_preserves_baselines_and_seeds():
    source = Path("results/campaign_v7_plans/qp_evaluate.csv")
    if not source.exists():
        pytest.skip("local completed evaluation manifest unavailable")
    configs = configurations(source)
    assert len(configs) == 183
    groups = {}
    for config in configs:
        if config["inner"] in ("osqp", "scs", "pdhg"):
            assert not config["trace_inner"]
            continue
        key = (config["side"], config["seed"], config["inner"], config.get("rank"))
        groups.setdefault(key, []).append(config)
    assert len(groups) == 54
    for group in groups.values():
        assert {c["inner_initialization"] for c in group} == {
            "warm",
            "fixed",
            "gaussian",
        }
        parameters = [
            {k: v for k, v in c.items() if k != "inner_initialization"} for c in group
        ]
        assert json.dumps(parameters[0], sort_keys=True) == json.dumps(
            parameters[1], sort_keys=True
        )
        assert parameters[0] == parameters[2]


def test_wall_time_keeps_setup_and_launch_overhead():
    row = dict(clean_attempt_seconds="15", solve_seconds="2", solver_setup_seconds="10")
    assert timing_parts(row) == (2, 10, 3, 15)
    with pytest.raises(ValueError, match="overlapping"):
        timing_parts(dict(row, solve_seconds="20"))


def test_convergence_aggregation_never_extends_a_shorter_run():
    series = [
        (np.array([0, 10]), np.array([1, 0.01])),
        (np.array([2, 20]), np.array([0.1, 0.001])),
    ]
    x, median, low, high = common_curve(series)
    assert min(x) >= 2 and max(x) <= 10
    assert np.all(low <= median) and np.all(median <= high)
