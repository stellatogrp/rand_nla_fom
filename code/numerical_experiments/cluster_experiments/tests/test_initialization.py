import json

import numpy as np
import pytest
import scipy.sparse as sp

from experiments.initialization import starting_point
from experiments.inner import Inner
from experiments.io import identity, read_csv, write_csv
from experiments.manifest import group_key, main, winners
from experiments.problems import make_qp, make_regression
from experiments.solvers import solve
from experiments.tuning import resolve


def cfg(**changes):
    return dict(
        dict(
            seed=0,
            inner="cg",
            ratio=10.0,
            sigma=0.2,
            stride=2,
            tolerance=2e-5,
            time_limit=10.0,
            max_inner=100000,
            max_outer=5000,
        ),
        **changes,
    )


@pytest.mark.parametrize("mode", ["warm", "fixed", "gaussian"])
def test_each_outer_solve_receives_requested_start(mode, monkeypatch):
    p = make_qp(side=4)
    starts = []
    original = Inner.start

    def capture(self, x, rhs):
        starts.append(x.copy())
        original(self, x, rhs)

    monkeypatch.setattr(Inner, "start", capture)
    config = cfg(inner_initialization=mode, max_outer=4)
    solve(p, config)
    assert len(starts) == 4
    if mode == "warm":
        np.testing.assert_array_equal(starts[0], np.zeros(p.n))
        assert np.linalg.norm(starts[1]) > 0
        assert not np.array_equal(starts[1], starts[2])
    elif mode == "fixed":
        for point in starts:
            np.testing.assert_array_equal(point, np.zeros(p.n))
    else:
        for outer, point in enumerate(starts):
            np.testing.assert_array_equal(
                point, starting_point(np.zeros(p.n), outer, config)
            )
        assert not np.array_equal(starts[0], starts[1])


@pytest.mark.parametrize("family", ["qp", "lasso", "logistic"])
@pytest.mark.parametrize("mode", ["warm", "fixed", "gaussian"])
def test_modes_solve_the_same_problem(family, mode):
    p = (
        make_qp(side=4)
        if family == "qp"
        else make_regression(
            family, sp.csr_matrix(np.random.default_rng(8).normal(size=(32, 12))), 2
        )
    )
    config = cfg(inner="gd" if family == "logistic" else "cg")
    reference, _ = solve(p, config)
    result, history = solve(p, dict(config, inner_initialization=mode))
    assert result["converged"]
    assert result["objective"] == pytest.approx(reference["objective"], abs=2e-3)
    assert result["inner_initialization"] == mode
    assert result["initialization_seconds"] >= 0
    assert "initial_graph_residual" in history[0]


def test_gaussian_scaling_seed_and_fixed_value():
    previous = np.ones(100000)
    config = cfg(
        inner_initialization="gaussian", initialization_seed=12, gaussian_scale=3.0
    )
    a = starting_point(previous, 4, config)
    assert np.linalg.norm(a) == pytest.approx(3.0, rel=0.02)
    np.testing.assert_array_equal(a, starting_point(previous, 4, config))
    assert not np.array_equal(a, starting_point(previous, 5, config))
    fixed = starting_point(
        previous, 1, cfg(inner_initialization="fixed", fixed_initial_value=0.25)
    )
    np.testing.assert_array_equal(fixed, np.full_like(previous, 0.25))
    with pytest.raises(ValueError, match="positive scale"):
        starting_point(previous, 0, dict(config, gaussian_scale=0))


def test_gaussian_does_not_change_randomized_solver_setup():
    p = make_qp(side=4)
    a = Inner(p, 0.1, "nystrom_pcg", np.random.default_rng(100003), rank=8)
    starting_point(np.zeros(p.n), 0, cfg(inner_initialization="gaussian"))
    b = Inner(p, 0.1, "nystrom_pcg", np.random.default_rng(100003), rank=8)
    np.testing.assert_array_equal(
        a.precondition(np.ones(p.n)), b.precondition(np.ones(p.n))
    )


@pytest.mark.parametrize("mode", ["fixed", "gaussian"])
def test_resume_reproduces_initialization_sequence(mode):
    p = make_qp(side=4)
    captured = []
    config = cfg(inner_initialization=mode, checkpoint_seconds=0)

    def capture(state, history):
        captured.append(
            {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()}
        )

    full, _ = solve(p, config, capture)
    state = captured[len(captured) // 2]
    resumed, history = solve(p, config, state=state)
    assert resumed["objective"] == pytest.approx(full["objective"], abs=1e-12)
    assert resumed["total_inner"] == full["total_inner"]
    assert history[0]["iteration"] == state["outer"]


def test_new_manifest_includes_paired_modes_and_single_baselines(tmp_path, monkeypatch):
    path = tmp_path / "manifest.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "manifest",
            "smoke",
            "--output",
            str(path),
            "--seeds",
            "0",
            "--families",
            "qp",
        ],
    )
    main()
    configs = [json.loads(row["config"]) for row in read_csv(path)]
    ours = [c for c in configs if c["inner"] == "cg"]
    assert {c["inner_initialization"] for c in ours} == {"warm", "fixed", "gaussian"}
    assert len([c for c in configs if c["inner"] == "osqp"]) == 1
    pairs = {}
    for c in ours:
        key = tuple((k, v) for k, v in c.items() if k != "inner_initialization")
        pairs.setdefault(key, []).append(c)
    assert all(len(group) == 3 for group in pairs.values())
    assert len({identity(c) for c in configs}) == len(configs)


def test_ablation_freezes_resolved_warm_parameters(tmp_path, monkeypatch):
    reference = dict(
        family="qp",
        dataset="synthetic",
        orientation="native",
        side=4,
        seed=0,
        inner="cg",
        ratio=10.0,
        sigma="optimal",
        stride=1,
        sweep="sigma",
        inner_initialization="warm",
    )
    run_id = identity(reference)
    results = tmp_path / "reference"
    write_csv(
        results / run_id / "result.csv",
        [
            dict(
                config=json.dumps(reference),
                run_id=run_id,
                converged=True,
                total_seconds=1.0,
                sigma=0.123,
                stride=1,
            )
        ],
    )
    source = tmp_path / "source.csv"
    write_csv(source, [dict(run_id=run_id)])
    output = tmp_path / "ablation.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "manifest",
            "initialization",
            "--from-results",
            str(results),
            "--from-manifest",
            str(source),
            "--output",
            str(output),
            "--seeds",
            "0",
            "1",
        ],
    )
    main()
    configs = [json.loads(row["config"]) for row in read_csv(output)]
    assert len(configs) == 6
    assert {c["inner_initialization"] for c in configs} == {"warm", "fixed", "gaussian"}
    assert all(
        c["sigma"] == 0.123
        and c["stride"] == 1
        and c["ratio"] == 10.0
        and c["reference_run_id"] == run_id
        for c in configs
    )
    assert all(c["initialization_seed"] == c["seed"] + 200003 for c in configs)


def test_selection_and_theory_distinguish_initialization_modes(tmp_path):
    for mode in ["warm", "fixed", "gaussian"]:
        config = dict(
            family="qp",
            dataset="synthetic",
            orientation="native",
            side=4,
            **cfg(inner_initialization=mode),
        )
        write_csv(
            tmp_path / mode / "result.csv",
            [
                dict(
                    config=json.dumps(config),
                    converged=True,
                    total_seconds=1.0,
                    sigma=0.2,
                    stride=2,
                )
            ],
        )
    assert len(winners(tmp_path)) == 3
    assert len(winners(tmp_path, warm_only=True)) == 1
    assert group_key(cfg(inner_initialization="warm")) != group_key(
        cfg(inner_initialization="fixed")
    )
    _, metadata = resolve(make_qp(side=4), cfg(inner_initialization="gaussian"))
    assert metadata["recommendation_kind"] == "warm_start_reference_model"


def test_paired_report_excludes_failures_and_mismatched_parameters():
    from plots.initialization import comparisons

    rows = []
    for mode, converged, seconds in [
        ("warm", "True", 2.0),
        ("fixed", "True", 6.0),
        ("gaussian", "False", 0.01),
    ]:
        config = cfg(inner_initialization=mode)
        rows.append(
            dict(
                config=json.dumps(config),
                converged=converged,
                total_seconds=str(seconds),
                total_inner="20",
                checks="10",
                sigma="0.2",
                stride="2",
            )
        )
    paired = {row["inner_initialization"]: row for row in comparisons(rows)}
    assert paired["fixed"]["total_seconds_ratio_to_warm"] == 3.0
    assert np.isnan(paired["gaussian"]["total_seconds_ratio_to_warm"])
    rows[1]["stride"] = "3"
    paired = {row["inner_initialization"]: row for row in comparisons(rows)}
    assert paired["fixed"]["warm_status"] == "missing"
    assert np.isnan(paired["fixed"]["total_seconds_ratio_to_warm"])
