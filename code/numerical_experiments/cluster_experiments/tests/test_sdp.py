"""Algorithm 3 for semidefinite programs (the ``sdp`` package)."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from sdp import run, summarize
from sdp.admm import _measure, certificate, proximal_admm, theory_checks
from sdp.problems import maxcut, planted, reference
from sdp.projection import (
    BlockLanczos,
    ConePair,
    certify,
    restart_block,
    restart_one_sided,
)


def symmetric(n: int, seed: int) -> np.ndarray:
    G = np.random.default_rng(seed).standard_normal((n, n))
    return (G + G.T) / 2.0


def random_block(n: int, size: int, seed: int) -> np.ndarray:
    block = np.random.default_rng(seed).standard_normal((n, size))
    return np.linalg.qr(block)[0]


def assert_cone_pair(pair: ConePair) -> None:
    """``Z`` in the cone and ``B`` in its normal cone there, to rounding error."""
    scale = max(np.linalg.norm(pair.Z), np.linalg.norm(pair.B), 1.0)
    assert np.linalg.eigvalsh(pair.Z)[0] >= -1e-12 * scale
    assert np.linalg.eigvalsh(pair.B)[-1] <= 1e-12 * scale
    assert np.linalg.norm(pair.Z @ pair.B) <= 1e-12 * scale**2


def test_every_krylov_subspace_gives_a_point_of_the_graph():
    n = 30
    M = symmetric(n, 0)
    lanczos = BlockLanczos(M, random_block(n, 4, 1), np.random.default_rng(2))
    residuals = []
    while not lanczos.exhausted:
        lanczos.step()
        pair = lanczos.ritz()
        assert_cone_pair(pair)
        # The residual of the Moreau decomposition is P M P - M, with norm
        # ||M||^2 - ||Q'MQ||^2 in squares.
        k = lanczos.dim
        compression = np.linalg.norm(lanczos.T[:k, :k]) ** 2
        residual = np.linalg.norm(pair.Z + pair.B - M)
        assert residual**2 == pytest.approx(
            np.linalg.norm(M) ** 2 - compression, abs=1e-10
        )
        residuals.append(residual)
    # A growing subspace can only shrink the residual, down to zero at R^n.
    assert all(a >= b - 1e-12 for a, b in zip(residuals, residuals[1:], strict=False))
    assert lanczos.dim == n and lanczos.matvecs == n
    exact = ConePair.exact(M)
    assert np.allclose(pair.Z, exact.Z, atol=1e-10)
    assert np.allclose(pair.B, exact.B, atol=1e-10)
    assert np.allclose(exact.Z + exact.B, M, atol=1e-12)


def test_block_lanczos_grows_past_an_invariant_subspace():
    n = 20
    M = symmetric(n, 3)
    # Starting in an exact invariant subspace, the Krylov space of M stalls;
    # the residual of the projection is then far from zero, so the method
    # must refill the block and keep growing.
    eigenvectors = np.linalg.eigh(M)[1]
    lanczos = BlockLanczos(M, eigenvectors[:, :3], np.random.default_rng(4))
    lanczos.step()
    assert not lanczos.exhausted and lanczos.block.shape[1] == 3
    while not lanczos.exhausted:
        lanczos.step()
    Q = lanczos.Q
    assert np.allclose(Q.T @ Q, np.eye(n), atol=1e-12)
    assert np.allclose(lanczos.ritz().Z, ConePair.exact(M).Z, atol=1e-10)


def test_restart_block_keeps_the_extreme_ritz_values():
    n = 12
    V = np.linalg.qr(np.random.default_rng(5).standard_normal((n, 6)))[0]
    theta = np.array([-3.0, -0.01, 0.002, 0.02, 1.0, 4.0])
    pair = ConePair.from_ritz(theta, V)
    rng = np.random.default_rng(6)
    # Dropping 0.002, 0.01, and 0.02 leaves a Frobenius norm of about 0.0225.
    block = restart_block(pair, tolerance=0.03, block_size=2, rng=rng)
    assert block.shape == (n, 3)
    assert np.allclose(np.abs(block.T @ V[:, [5, 0, 4]]), np.eye(3), atol=1e-12)
    # A tolerance of zero keeps every Ritz vector; a huge one keeps none and
    # the block is filled with orthonormal random directions.
    assert restart_block(pair, 0.0, 2, rng).shape == (n, 6)
    filled = restart_block(pair, 100.0, 4, rng)
    assert filled.shape == (n, 4)
    assert np.allclose(filled.T @ filled, np.eye(4), atol=1e-12)


def test_planted_solution_satisfies_the_optimality_conditions():
    sdp = planted(15, 30, 3, seed=0)
    assert np.allclose(sdp.E.apply(sdp.X_star), sdp.b)
    assert np.allclose(sdp.C - sdp.E.adjoint(sdp.nu_star), sdp.S_star, atol=1e-12)
    assert np.linalg.norm(sdp.X_star @ sdp.S_star) <= 1e-14
    assert np.linalg.matrix_rank(sdp.X_star) == 3
    assert np.linalg.matrix_rank(sdp.S_star) == 12
    assert np.linalg.eigvalsh(sdp.X_star)[0] >= -1e-14
    assert np.linalg.eigvalsh(sdp.S_star)[0] >= -1e-14
    # The adjoint is the adjoint, and the gram matrix is E E*.
    X, u = symmetric(15, 1), np.random.default_rng(2).standard_normal(30)
    assert np.vdot(sdp.E.apply(X), u) == pytest.approx(np.vdot(X, sdp.E.adjoint(u)))
    assert np.allclose(sdp.E.gram() @ u, sdp.E.apply(sdp.E.adjoint(u)))
    cut = maxcut(8, seed=0)
    assert np.allclose(cut.E.gram() @ u[:8], cut.E.apply(cut.E.adjoint(u[:8])))


def test_test_quantities_match_the_definitions_of_the_paper():
    """``_measure`` against equations (fully-inexact-admm-residuals) and
    (admm-test-quantities), written out for the stacked map ``X -> (E(X), X)``."""
    n, m, c, rho = 9, 7, 0.7, 0.3
    sdp = planted(n, m, 2, seed=0)
    rng = np.random.default_rng(1)
    X, t, w_Z = symmetric(n, 2), symmetric(n, 3), symmetric(n, 4)
    w_y = rng.standard_normal(m)
    lanczos = BlockLanczos(symmetric(n, 5), random_block(n, 3, 6), rng)
    lanczos.step()
    pair = lanczos.ritz()

    def stack(u, U):
        return np.concatenate([u, U.ravel()])

    def lift(X):  # the coupling map
        return stack(sdp.E.apply(X), X)

    def lower(v):  # its adjoint
        return sdp.E.adjoint(v[:m]) + v[m:].reshape(n, n)

    EX = sdp.E.apply(X)
    a = sdp.C
    b_y = c * (2.0 * EX - w_y - sdp.b)
    z = stack(sdp.b, pair.Z)
    b = stack(b_y, c * pair.B)
    w = stack(w_y, w_Z)
    d = lift(X) - z
    q = (a + lower(b)) / c - lower(d)
    r_f = a / c + lower(lift(X) - w) + rho * (X - t)
    r_g = z + b / c - 2.0 * lift(X) + w
    Q = math.sqrt(np.linalg.norm(d) ** 2 + np.linalg.norm(q) ** 2 / rho)
    E = math.sqrt(
        np.linalg.norm(r_g) ** 2 + np.linalg.norm(r_f + lower(r_g)) ** 2 / rho
    )
    P = math.sqrt(np.linalg.norm(d + r_g) ** 2 + rho * np.linalg.norm(X - t) ** 2)

    got = _measure(X, EX, -r_f, pair, w_Z, t, sdp.b, rho)
    assert got.Q == pytest.approx(Q, rel=1e-12)
    assert got.E == pytest.approx(E, rel=1e-12)
    assert got.P == pytest.approx(P, rel=1e-12)
    assert np.allclose(got.q, q, atol=1e-12)
    # The y part of r_g vanishes by the choice of b_y, and E <= e_f + e_g.
    assert np.allclose(r_g[:m], 0.0, atol=1e-12)
    assert got.E <= got.e_f + got.e_g + 1e-12


def spectrum(n: int, values, seed: int):
    """A symmetric matrix with the given eigenvalues and its eigenvectors."""
    U = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))[0]
    values = np.asarray(values, dtype=float)
    return (U * values) @ U.T, U


def test_one_sided_pair_needs_only_the_positive_ritz_pairs():
    n, r = 30, 4
    values = np.concatenate([np.linspace(2.0, 1.0, r), -np.linspace(0.5, 1.5, n - r)])
    M, U = spectrum(n, values, 0)
    # A slightly rotated copy of the positive eigenvectors and four of the
    # negative ones nearest zero: what a warm start looks like.
    start = U[:, : r + 4] + 1e-3 * np.random.default_rng(1).standard_normal((n, r + 4))
    lanczos = BlockLanczos(M, np.linalg.qr(start)[0], np.random.default_rng(2))
    lanczos.step()
    pair = lanczos.one_sided(guard=4)
    assert lanczos.dim == r + 4 and pair.rank_positive == r
    V_plus = pair.V[:, pair.theta > 0.0]
    R_plus = M @ V_plus - V_plus * pair.theta[pair.theta > 0.0]
    # Z in the cone, Z B = 0 exactly, and the residual is the Ritz residual
    # of the positive pairs alone, although the subspace is 8 of 30.
    assert np.linalg.eigvalsh(pair.Z)[0] >= -1e-13
    assert np.linalg.norm(pair.Z @ pair.B) <= 1e-12
    residual = np.linalg.norm(pair.Z + pair.B - M)
    assert residual == pytest.approx(math.sqrt(2.0) * np.linalg.norm(R_plus), rel=1e-9)
    assert residual <= 0.1
    two_sided = lanczos.ritz()
    assert np.linalg.norm(two_sided.Z + two_sided.B - M) >= 1.0
    # Every positive direction is in the subspace, so B is in the normal
    # cone: the guard says so and the certificate confirms it.
    assert pair.bound <= 0.0 and not pair.certified
    assert np.linalg.eigvalsh(pair.B)[-1] <= 1e-12
    assert certify(pair) and pair.certified
    # The restart keeps the positive Ritz vectors and the guard.
    assert restart_one_sided(pair).shape == (n, r + 4)
    # At the whole space the pair is the Moreau decomposition.
    while not lanczos.exhausted:
        lanczos.step()
    full, exact = lanczos.one_sided(guard=4), ConePair.exact(M)
    assert np.allclose(full.Z, exact.Z, atol=1e-10)
    assert np.allclose(full.B, exact.B, atol=1e-10)
    assert full.bound <= 0.0


def test_certificate_catches_a_positive_direction_the_guard_cannot_see():
    n = 20
    values = np.concatenate([[3.0, 2.0, 0.4], -np.linspace(0.5, 2.0, n - 3)])
    M, U = spectrum(n, values, 3)
    # An exactly invariant subspace that misses the eigenvalue 0.4: every
    # Ritz residual vanishes, so the guard has no way to know.
    start = np.hstack([U[:, :2], U[:, 3:6]])
    lanczos = BlockLanczos(M, start, np.random.default_rng(4))
    lanczos.step()
    pair = lanczos.one_sided(guard=3, margin=1e-12)
    assert pair.rank_positive == 2 and pair.bound <= 0.0
    assert np.linalg.eigvalsh(pair.B)[-1] == pytest.approx(0.4)
    assert not certify(pair) and not pair.certified
    # An unresolved top of the deflated spectrum is what the guard does see.
    rough = BlockLanczos(M, random_block(n, 5, 5), np.random.default_rng(6))
    rough.step()
    assert rough.one_sided(guard=3).bound > 0.0


def test_cold_start_of_the_one_sided_projection_is_exact():
    n = 12
    start = restart_one_sided(ConePair.zero(n))
    assert np.array_equal(start, np.eye(n))
    M = symmetric(n, 7)
    lanczos = BlockLanczos(M, start, np.random.default_rng(8))
    lanczos.step()
    assert lanczos.exhausted
    pair, exact = lanczos.one_sided(guard=2), ConePair.exact(M)
    assert np.allclose(pair.Z, exact.Z, atol=1e-12)
    assert np.allclose(pair.B, exact.B, atol=1e-12)
    with pytest.raises(ValueError):
        lanczos.one_sided(guard=0)


@pytest.mark.parametrize("cholesky", [False, True])
def test_one_sided_admm_accepts_only_points_of_the_graph(cholesky):
    c, rho, delta, sigma = 1.0, 1.0, 1e-7, 0.5
    sdp = planted(30, 50, 3, seed=4)
    two_sided = proximal_admm(sdp, c=c, rho=rho, delta=delta, sigma=sigma, block_size=3)
    trace = proximal_admm(
        sdp,
        c=c,
        rho=rho,
        delta=delta,
        sigma=sigma,
        block_size=3,
        projection="one_sided",
        cholesky=cholesky,
    )
    assert trace.converged
    cert = certificate(sdp, trace, c, rho, delta)
    assert cert["min_eigenvalue_Z"] >= -1e-12
    assert cert["max_eigenvalue_B"] <= 1e-12
    assert cert["complementarity"] <= 1e-12
    assert cert["coupling_residual"] <= delta
    assert cert["stationarity_residual"] <= cert["stationarity_bound"]
    assert cert["objective"] == pytest.approx(sdp.objective_star, abs=1e-5)
    assert cert["solution_error"] <= 1e-4
    # Every accepted B is in the normal cone (the diagnostic), so the outer
    # guarantees hold exactly as for the two-sided pair.
    assert max(trace.violation) <= 1e-10
    checks = theory_checks(trace, sigma, delta)
    assert checks["decrease_violation"] <= 1e-12
    assert checks["ratio_violation"] <= 1e-12
    for k, how in enumerate(trace.accepted_by):
        if how == "relative":
            assert trace.E[k] <= sigma * trace.P[k]
    # The first decomposition is exact; later ones use small subspaces and
    # far fewer products with M than the two-sided pair.
    assert trace.subspace_dim[0] == sdp.n
    assert np.median(trace.subspace_dim) < sdp.n / 2
    assert sum(trace.matvecs) < sum(two_sided.matvecs) / 2
    assert abs(trace.outer_iterations - two_sided.outer_iterations) <= 10
    if cholesky:
        # One certificate per accepted pair that is new, none of them failed
        # here, and the time spent on them is recorded.
        assert (
            0
            < sum(trace.certificates)
            <= len(trace.Q) + sum(trace.certificate_failures)
        )
        assert trace.certificate_seconds > 0.0
    else:
        assert sum(trace.certificates) == 0 and trace.certificate_seconds == 0.0
    assert trace.projection_seconds > 0.0
    with pytest.raises(ValueError):
        proximal_admm(sdp, projection="two_sided")


# The sdp_v1 campaign ran this version of the package (the hash recorded in
# every settings.json of results/sdp_v1), archived in releases/sdp_r1.
SDP_V1_SOURCES = "28af313fa57510c32db68391d6f477c4d1c6ab6d07e4f8ded74c44654dd6b695"


def test_the_archived_release_is_the_code_of_sdp_v1_and_still_reproduces():
    from pathlib import Path

    release = Path(run.__file__).parents[1] / "releases" / "sdp_r1"
    if not (release / "SHA256SUMS").exists():
        pytest.skip("sdp_r1 is not archived")
    import hashlib

    digest = hashlib.sha256()
    for path in sorted((release / "sdp").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    assert digest.hexdigest() == SDP_V1_SOURCES
    for line in (release / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        assert hashlib.sha256((release / name).read_bytes()).hexdigest() == expected
    # The methods of sdp_v1 are unchanged in the working package: the same
    # iterates, bit for bit.
    from releases.sdp_r1.sdp import admm as archived_admm
    from releases.sdp_r1.sdp import problems as archived_problems

    arguments = dict(n=16, m=30, rank=3, seed=5)
    old, new = archived_problems.planted(**arguments), planted(**arguments)
    for method in (
        dict(f_solver="exact", projection="exact", sigma=0.0),
        dict(f_solver="cg", projection="lanczos", sigma=0.9, block_size=3),
    ):
        before = archived_admm.proximal_admm(old, **method)
        after = proximal_admm(new, **method)
        assert before.Q == after.Q and before.matvecs == after.matvecs
        assert np.array_equal(before.Z, after.Z)


METHODS = [
    dict(f_solver="exact", projection="exact", sigma=0.0),
    dict(f_solver="cg", projection="exact", sigma=0.5),
    dict(f_solver="exact", projection="lanczos", sigma=0.5, block_size=3),
    dict(f_solver="cg", projection="lanczos", sigma=0.1, block_size=3),
    dict(f_solver="cg", projection="lanczos", sigma=0.9, block_size=3, stride=2),
    dict(f_solver="cg", projection="lanczos", sigma=0.0, exact_tolerance=1e-10),
]


@pytest.mark.parametrize("method", METHODS)
def test_admm_returns_a_certified_solution(method):
    c, rho, delta = 1.0, 0.5, 1e-7
    sdp = planted(16, 30, 3, seed=1)
    trace = proximal_admm(sdp, c=c, rho=rho, delta=delta, time_limit=60.0, **method)
    assert trace.converged and trace.stopped_by == "converged"
    assert trace.Q[-1] <= delta
    cert = certificate(sdp, trace, c, rho, delta)
    # The graph inclusions hold exactly, and Proposition 5.2 at termination.
    assert cert["min_eigenvalue_Z"] >= -1e-12
    assert cert["max_eigenvalue_B"] <= 1e-12
    assert cert["complementarity"] <= 1e-12
    assert cert["coupling_residual"] <= delta
    assert cert["stationarity_residual"] <= cert["stationarity_bound"]
    # The returned point solves the SDP: the planted value and solution.
    assert cert["objective"] == pytest.approx(sdp.objective_star, abs=1e-5)
    assert abs(cert["duality_gap"]) <= 1e-5
    assert cert["solution_error"] <= 1e-4
    # Every update accepted by the relative test satisfies it, and the outer
    # guarantees of Section 3 hold against the planted solution.
    sigma = method["sigma"]
    for k, how in enumerate(trace.accepted_by):
        if how == "relative":
            assert trace.E[k] <= sigma * trace.P[k]
    checks = theory_checks(trace, sigma, delta)
    assert checks["decrease_violation"] <= 1e-12
    assert checks["ratio_violation"] <= 1e-12
    assert checks["sum_Q_squared"] <= checks["sum_Q_squared_bound"]
    assert trace.outer_iterations <= checks["K_delta_plus"]
    # The distance to the planted solution never increases.
    assert all(a >= b - 1e-12 for a, b in zip(trace.dist, trace.dist[1:], strict=False))


def test_inexact_methods_count_their_work():
    sdp = planted(16, 30, 3, seed=2)
    exact = proximal_admm(sdp, sigma=0.0, f_solver="exact", projection="exact")
    assert sum(exact.cg_steps) == 0 and sum(exact.matvecs) == 0
    assert exact.eigendecompositions == len(exact.Q)
    inexact = proximal_admm(sdp, sigma=0.5, block_size=4)
    assert inexact.eigendecompositions == 0
    assert sum(inexact.cg_steps) > 0 and sum(inexact.matvecs) > 0
    # Each conjugate gradient step applies E and its adjoint once.
    assert inexact.forward >= sum(inexact.cg_steps)
    assert inexact.adjoint >= sum(inexact.cg_steps)
    assert max(inexact.subspace_dim) <= sdp.n
    # The dimension that passes the test is at least the smallest that could.
    assert all(
        dim >= needed
        for dim, needed, how in zip(
            inexact.subspace_dim, inexact.rank_needed, inexact.accepted_by, strict=True
        )
        if how == "relative"
    )
    # The same seed reproduces the run.
    again = proximal_admm(sdp, sigma=0.5, block_size=4)
    assert again.Q == inexact.Q and again.matvecs == inexact.matvecs


def test_stops_are_recorded():
    sdp = planted(16, 30, 3, seed=3)
    capped = proximal_admm(sdp, sigma=0.5, max_outer=5)
    assert not capped.converged and capped.stopped_by == "max_outer"
    assert len(capped.Q) == 6
    timed = proximal_admm(sdp, sigma=0.5, delta=1e-300, time_limit=0.0)
    assert timed.stopped_by == "time_limit"
    with pytest.raises(ValueError):
        proximal_admm(sdp, rho=0.0)


def test_maxcut_agrees_with_the_reference_solver():
    pytest.importorskip("cvxpy")
    sdp = maxcut(12, seed=0)
    truth = reference(sdp)
    if math.isnan(truth["value"]):
        pytest.skip(f"reference solver: {truth['status']}")
    c, rho, delta = 0.05, 1.0, 1e-7
    trace = proximal_admm(sdp, c=c, rho=rho, delta=delta, sigma=0.5, block_size=3)
    assert trace.converged
    cert = certificate(sdp, trace, c, rho, delta)
    assert cert["objective"] == pytest.approx(truth["value"], abs=1e-5)
    assert abs(cert["duality_gap"]) <= 1e-5
    assert cert["primal_feasibility"] <= 1e-6


ARGS = [
    "--allow-local",
    "--n",
    "12",
    "--m",
    "20",
    "--rank",
    "2",
    "--seeds",
    "0",
    "1",
    "--sigmas",
    "0.5",
    "--block-sizes",
    "3",
    "--delta",
    "1e-6",
    "--name",
    "tiny",
]


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_run_writes_the_tables_and_resumes(tmp_path, capsys):
    arguments = [*ARGS, "--results-dir", str(tmp_path)]
    directory = run.main(arguments)
    assert directory == tmp_path / "tiny"
    rows = read(directory / "runs.csv")
    labels = [
        "exact",
        "cg_only_sigma0.5",
        "lanczos_only_sigma0.5_b3",
        "inexact_sigma0.5_b3",
        "one_sided_sigma0.5_b3",
        "certified_sigma0.5_b3",
        "near_exact_b3",
    ]
    assert [row["label"] for row in rows] == labels * 2
    assert list(rows[0]) == run.RUN_FIELDS
    assert all(row["converged"] == "True" for row in rows)
    history = read(directory / "histories.csv")
    assert list(history[0]) == run.HISTORY_FIELDS
    assert {row["label"] for row in history} == set(labels)
    # A second invocation finds every run done; other settings are refused.
    capsys.readouterr()
    run.main(arguments)
    assert capsys.readouterr().out == ""
    assert len(read(directory / "runs.csv")) == len(rows)
    with pytest.raises(SystemExit):
        run.main([*arguments, "--rho", "0.5"])
    # Outside an allocation the study refuses to compute.
    with pytest.raises(SystemExit):
        run.main([argument for argument in arguments if argument != "--allow-local"])

    summary = summarize.main(["--results-dir", str(tmp_path)])
    assert [row["label"] for row in summary] == labels
    for row in summary:
        for check in (
            "converged",
            "inclusions_ok",
            "proposition_ok",
            "theory_ok",
            "graph_ok",
        ):
            assert row[check] == "2/2"
    one_sided = [row for row in rows if row["projection"] == "one_sided"]
    assert {row["cholesky"] for row in one_sided} == {"True", "False"}
    assert all(row["violations"] == "0" for row in one_sided)
    assert all(row["violations"] == "" for row in rows if row not in one_sided)
    assert list(read(tmp_path / "summary.csv")[0]) == summarize.SUMMARY_FIELDS
    assert "all checks passed on 14 runs" in capsys.readouterr().out
    # Tables written before the one-sided projection lack its columns.
    old = {k: v for k, v in rows[0].items() if k not in ("violations", "cholesky")}
    assert summarize.checks(old)["graph"]
    assert not summarize.checks(dict(rows[0], violations="2"))["graph"]


def test_history_is_thinned_for_long_runs():
    assert run._thin(10, 400) == list(range(10))
    kept = run._thin(5000, 400)
    assert len(kept) <= 400 and kept[0] == 0 and kept[-1] == 4999
    assert kept[:100] == list(range(100))


def test_figure_is_drawn_from_the_tables(tmp_path):
    from plots import sdp as figures

    run.main([*ARGS, "--results-dir", str(tmp_path)])
    written = figures.main(["--results-dir", str(tmp_path)])
    assert [path.name for path in written] == [
        "sanity.png",
        "sanity.pdf",
        "one_sided.png",
        "one_sided.pdf",
    ]
    assert all(path.stat().st_size > 0 for path in written)
    with pytest.raises(SystemExit):
        figures.main(["--results-dir", str(tmp_path / "missing")])
