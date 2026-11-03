import copy

import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from experiments.dense import DenseDesign
from experiments.inner import Inner
from experiments.plan import split
from experiments.problems import make_qp, make_regression
from experiments.run import execute
from experiments.runtime import threads
from experiments.solvers import solve
from experiments.tuning import resolve
from plots.scaling import summarize
from tests.test_experiments import config


def test_resident_products_share_data_and_do_not_reread_file(tmp_path, monkeypatch):
    a = np.random.default_rng(3).normal(size=(31, 19)).astype(np.float32)
    path = tmp_path / "a.npy"
    np.save(path, a)
    streamed = DenseDesign(path, block_rows=7).validate()
    resident = DenseDesign(path, block_rows=7, resident=True).validate()
    assert resident._resident.dtype == np.float64
    assert not resident._resident.flags.writeable
    assert resident.T._resident is resident._resident
    normalized = resident.normalized()
    assert normalized._resident is resident._resident
    expected_norm = streamed.norm_bound()
    for transpose in [False, True]:
        stream = streamed.T if transpose else streamed
        memory = resident.T if transpose else resident
        for columns in [None, 3]:
            shape = (
                (memory.shape[1],) if columns is None else (memory.shape[1], columns)
            )
            x = np.random.default_rng(4).normal(size=shape)
            expected = stream @ x
            with monkeypatch.context() as patch:
                patch.setattr(np, "fromfile", lambda *a, **k: pytest.fail("disk read"))
                np.testing.assert_allclose(memory @ x, expected, atol=1e-12)
    assert resident.norm_bound() == pytest.approx(expected_norm)
    np.testing.assert_allclose(
        normalized @ np.ones(19),
        a.astype(float) @ (1 / la.norm(a.astype(float), axis=0)),
    )
    with pytest.raises(MemoryError, match="payload budget"):
        DenseDesign(path, resident=True, max_resident_bytes=1)


@pytest.mark.parametrize(
    "method", ["gd", "cg", "gmres", "nystrom_pcg", "block_kaczmarz", "cyclic_kaczmarz"]
)
def test_calibration_reuse_preserves_rng_and_trajectory(method):
    p = make_qp(side=4)
    cfg = config(inner=method, block_size=8, max_inner=10000)
    resolved, tuning, inner = resolve(p, cfg, return_inner=True)
    fresh = Inner(p, 0.1, method, np.random.default_rng(100003), block_size=8, rank=8)
    assert inner.rng.bit_generator.state == fresh.rng.bit_generator.state
    assert inner.products == fresh.products
    assert set(inner.cache) == set(fresh.cache)
    assert tuning["tuning_seconds"] >= 0
    reference, _ = solve(p, resolved)
    actual, _ = solve(p, resolved, prepared_inner=inner)
    assert actual["status"] == reference["status"]
    assert actual["total_inner"] == reference["total_inner"]
    assert actual["objective"] == pytest.approx(reference["objective"], abs=1e-12)


def test_nystrom_setup_constructed_once_per_execute(tmp_path, monkeypatch):
    calls = []
    original = Inner.__init__

    def capture(self, *args, **kwargs):
        calls.append(1)
        original(self, *args, **kwargs)

    monkeypatch.setattr(Inner, "__init__", capture)
    row = execute(
        dict(
            config(inner="nystrom_pcg"),
            family="qp",
            side=4,
            dataset="synthetic",
            implementation="speed_v2",
        ),
        tmp_path,
        tmp_path,
    )
    assert len(calls) == 1
    assert row["setup_reused_within_run"]
    assert row["converged"]


def test_local_kaczmarz_update_equals_full_row_projection():
    p = make_qp(side=6)
    inner = Inner(p, 0.1, "cyclic_kaczmarz", np.random.default_rng(12), block_size=4)
    rng = np.random.default_rng(4)
    x, rhs = rng.normal(size=p.n), rng.normal(size=p.n)
    inner.start(x, rhs)
    rows = inner.permutation[:4]
    B = inner.H[rows]
    expected = x + B.T @ la.solve(
        (B @ B.T).toarray(), rhs[rows] - B @ x, assume_a="pos"
    )
    inner.step()
    np.testing.assert_allclose(inner.x, expected, atol=1e-14)
    assert inner.cache[0][1].shape[1] <= 7 * 4


@pytest.mark.parametrize("family", ["qp", "lasso", "logistic"])
def test_reused_metrics_match_independent_metrics(family):
    p = (
        make_qp(side=4)
        if family == "qp"
        else make_regression(
            family, sp.csr_matrix(np.random.default_rng(4).normal(size=(21, 12))), 0
        )
    )
    x = np.random.default_rng(2).normal(size=p.n)
    fresh = p.metrics(x)
    reused = p.metrics(x, gradient=p.gradient(x))
    for key in ["objective", "stationarity", "feasibility"]:
        assert reused[key] == pytest.approx(fresh[key], abs=1e-12)


def test_qp_metrics_reuses_gradient_without_hessian_product(monkeypatch):
    p = make_qp(side=4)
    x = np.ones(p.n)
    g = p.gradient(x)
    monkeypatch.setattr(p, "hess", lambda x: pytest.fail("repeated Hessian product"))
    p.metrics(x, gradient=g)


def test_thread_request_must_fit_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert threads(dict(blas_threads=2)) == 2
    for count in [0, -1, True, 1.5, 4]:
        with pytest.raises(ValueError):
            threads(dict(blas_threads=count))


def test_screen_keeps_each_mode_and_all_blocks():
    configs = []
    for method in ["cg", "block_kaczmarz", "cyclic_kaczmarz"]:
        for block in [1, 8, 32, 128, 512]:
            for side in [40, 100]:
                for mode in ["warm", "fixed", "gaussian"]:
                    configs.append(
                        dict(
                            config(inner=method, block_size=block),
                            side=side,
                            inner_initialization=mode,
                            sweep="sigma",
                            sigma=0.2,
                            stride=1,
                        )
                    )
    fast, screen = split(configs)
    assert all("kaczmarz" not in c["inner"] for c in fast)
    assert len(screen) == 30
    assert {c["block_size"] for c in screen} == {1, 8, 32, 128, 512}
    assert {c["inner_initialization"] for c in screen} == {"warm", "fixed", "gaussian"}
    assert all(c["side"] == 40 and c["time_limit"] == 120 for c in screen)


def test_scaling_requires_matched_hardware_and_good_utilization():
    rows = []
    # Synthetic timing fixtures exercise selection; these are not measurements.
    for n, seconds, efficiency in [(1, 10.0, 0.99), (2, 5.2, 0.95), (4, 5.0, 0.5)]:
        for repeat in range(2):
            rows.append(
                dict(
                    kind="qp",
                    dataset="test",
                    family="qp",
                    storage="sparse",
                    operation="solve",
                    inner="nystrom_pcg",
                    cpu_model="test",
                    converged="True",
                    benchmark_status="completed",
                    threads=n,
                    seconds=seconds,
                    cpu_efficiency=efficiency,
                    repeat=repeat,
                )
            )
    assert {r["recommended_threads"] for r in summarize(rows)} == {2}
    unmatched = copy.deepcopy(rows)
    for row in unmatched:
        if row["threads"] > 1:
            row["cpu_model"] = "other"
    assert {r["recommended_threads"] for r in summarize(unmatched)} == {1}


def test_batched_nystrom_matches_columnwise_construction(monkeypatch):
    p = make_qp(side=4)
    batched = Inner(p, 0.1, "nystrom_pcg", np.random.default_rng(8), rank=8)

    def columns(self, x):
        self.products += 1 if x.ndim == 1 else x.shape[1]
        return (
            p.hess(x)
            if x.ndim == 1
            else np.column_stack([p.hess(column) for column in x.T])
        )

    monkeypatch.setattr(Inner, "qmul", columns)
    reference = Inner(p, 0.1, "nystrom_pcg", np.random.default_rng(8), rank=8)
    np.testing.assert_allclose(
        batched.precondition(np.ones(p.n)),
        reference.precondition(np.ones(p.n)),
        atol=1e-12,
    )


def test_performance_timeout_preserves_completed_trials(tmp_path, monkeypatch):
    import subprocess

    from experiments import performance
    from experiments.io import read_csv, write_csv

    output = tmp_path / "profile.csv"
    write_csv(output, [dict(benchmark_status="completed", repeat=0)])
    events = []

    class Hung:
        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("profile", timeout)
            events.append("reaped")

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setattr(performance.subprocess, "Popen", lambda command: Hung())
    monkeypatch.setattr(performance.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        performance.sys,
        "argv",
        ["profile", "--kind", "qp", "--threads", "1", "--output", str(output)],
    )
    with pytest.raises(SystemExit):
        performance.main()
    assert events == ["terminate", "kill", "reaped"]
    assert [r["benchmark_status"] for r in read_csv(output)] == [
        "completed",
        "worker_timeout",
    ]
