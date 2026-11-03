import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from experiments.baselines import external, pdhg, qp_data
from experiments.inner import Inner
from experiments.io import write_csv
from experiments.manifest import winners
from experiments.problems import make_qp, make_regression
from experiments.run import load_state, save_state
from experiments.solvers import solve
from experiments.tuning import resolve


def config(**changes):
    return dict(
        dict(
            seed=0,
            inner="cg",
            ratio=10.0,
            sigma=0.2,
            stride=2,
            tolerance=1e-6,
            time_limit=10.0,
            max_inner=100000,
            max_outer=5000,
            block_size=4,
            rank=8,
        ),
        **changes,
    )


def test_qp_condition_and_planted_kkt():
    p = make_qp(side=4)
    eigen = la.eigvalsh(p.Q.toarray())
    assert eigen[0] == pytest.approx(1e-4)
    assert eigen[-1] == pytest.approx(1.0)
    assert eigen[-1] / eigen[0] == pytest.approx(1e4)
    metrics = p.metrics(p.truth)
    assert metrics["stationarity"] < 1e-14
    assert metrics["feasibility"] == 0.0


@pytest.mark.parametrize(
    "method",
    ["gd", "agd", "cg", "gmres", "nystrom_pcg", "block_kaczmarz", "cyclic_kaczmarz"],
)
def test_inner_solvers_solve_same_optimality_system(method):
    p = make_qp(side=4)
    rng = np.random.default_rng(12)
    rhs = rng.normal(size=p.n)
    solver = Inner(p, 0.1, method, rng, block_size=8, rank=8)
    solver.start(np.zeros(p.n), rhs)
    for _ in range(10000):
        solver.step()
        if la.norm((p.Q @ solver.x) + 0.1 * solver.x - rhs) < 1e-7:
            break
    exact = la.solve(p.Q.toarray() + 0.1 * np.eye(p.n), rhs, assume_a="pos")
    assert la.norm(solver.x - exact) / la.norm(exact) < 1e-6
    assert len(solver.cache) <= 8


def test_admm_matches_planted_solution_and_budget():
    p = make_qp(side=4)
    result, _ = solve(p, config())
    assert result["converged"]
    assert result["solution_error"] < 2e-5
    bounded, _ = solve(p, config(stride=20, max_inner=3))
    assert bounded["total_inner"] <= 3
    assert bounded["status"] == "max_inner"


def test_lasso_qp_equivalence_and_no_gram():
    rng = np.random.default_rng(0)
    p = make_regression("lasso", sp.csr_matrix(rng.normal(size=(32, 16))), 1)
    coef = rng.normal(size=16)
    z = np.r_[np.maximum(coef, 0), np.maximum(-coef, 0)]
    r = p.A @ coef - p.target
    expected = 0.5 * r @ r + p.regularization * np.sum(abs(coef))
    assert p.objective(z) == pytest.approx(expected)
    P, q, A, lower, upper, _ = qp_data(p)
    lifted = np.r_[z, r]
    assert 0.5 * lifted @ (P @ lifted) + q @ lifted == pytest.approx(expected)
    assert np.all(A @ lifted >= lower - 1e-12)
    assert np.all(A @ lifted <= upper + 1e-12)
    assert p.Q is None


@pytest.mark.parametrize("family", ["qp", "lasso", "logistic"])
def test_independent_baseline_agreement(family):
    p = (
        make_qp(side=4)
        if family == "qp"
        else make_regression(
            family, sp.csr_matrix(np.random.default_rng(5).normal(size=(32, 16))), 2
        )
    )
    cfg = config(tolerance=2e-5)
    ours, _ = solve(p, cfg)
    reference, _ = external(p, dict(cfg, inner="scs"))
    import json

    json.dumps(reference)  # NumPy booleans must not destroy a completed result.
    first_order, _ = pdhg(p, cfg)
    assert ours["converged"]
    assert reference["stationarity"] < 1e-3
    assert first_order["converged"]
    assert abs(ours["objective"] - reference["objective"]) < 2e-3
    assert abs(ours["objective"] - first_order["objective"]) < 2e-3


def test_recommendations_are_included_and_finite():
    p = make_qp(side=4)
    cfg, metrics = resolve(p, config(inner="gd", sigma="optimal"))
    assert 0 < cfg["sigma"] < 1
    assert metrics["recommendation_kind"] == "bound_cost_model"
    cfg, _ = resolve(p, config(stride="optimal"))
    assert cfg["stride"] >= 1


def test_logistic_is_not_passed_to_osqp():
    p = make_regression("logistic", sp.eye(10), 0)
    row, _ = external(p, config(inner="osqp"))
    assert row["status"] == "not_applicable"


def test_checkpoint_resume_preserves_outer_trajectory(tmp_path):
    p = make_qp(side=4)
    cfg = config(checkpoint_seconds=0)
    captured = []
    full, _ = solve(
        p,
        cfg,
        lambda state, history: captured.append(
            {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in state.items()}
        ),
    )
    state = captured[len(captured) // 2]
    path = tmp_path / "checkpoint.npz"
    save_state(path, state)
    resumed, _ = solve(p, cfg, state=load_state(path))
    assert resumed["objective"] == pytest.approx(full["objective"], abs=1e-12)
    assert resumed["total_inner"] == full["total_inner"]
    assert resumed["checks"] == full["checks"]


def test_winner_selection_rejects_fast_failures(tmp_path):
    import json

    for seed in [0, 1]:
        for sigma, converged, cost in [(0.1, True, 2.0), (0.9, False, 0.01)]:
            cfg = dict(
                family="qp",
                dataset="synthetic",
                orientation="native",
                inner="gd",
                side=4,
                seed=seed,
                sigma=sigma,
                stride=1,
                ratio=10.0,
                sweep="sigma",
            )
            write_csv(
                tmp_path / f"{seed}_{sigma}" / "result.csv",
                [
                    dict(
                        config=json.dumps(cfg),
                        converged=converged,
                        total_seconds=cost,
                        sigma=sigma,
                        stride=1,
                    )
                ],
            )
    selected = winners(tmp_path)
    assert len(selected) == 1
    assert selected[0]["sigma"] == 0.1


def test_data_conversion_validates_source_and_metadata(tmp_path, monkeypatch):
    import scipy.io

    from experiments import data

    monkeypatch.setitem(data.CATALOG, "fixture", ("test", 4, 4, 4))
    raw = tmp_path / "fixture.mat"
    scipy.io.savemat(raw, {"Problem": {"A": sp.eye(4, format="csc")}})
    data.prepare("fixture", tmp_path, "download")
    data.prepare("fixture", tmp_path, "convert")
    assert sp.load_npz(tmp_path / "fixture.npz").shape == (4, 4)
    with open(raw, "ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        data.prepare("fixture", tmp_path, "convert")
