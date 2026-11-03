import json

import numpy as np
import pytest
import scipy.sparse as sp

from releases.accounting_v3.experiments import benchmark_data as data
from releases.accounting_v3.experiments.accounting import Journal


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
            implementation="accounting_v3",
            max_inner=10000,
            max_outer=10000,
        ),
        **changes,
    )


def test_chunked_features_resume_without_recomputing_committed_chunk(tmp_path):
    x = sp.csr_matrix(np.random.default_rng(3).normal(size=(20, 4)))
    path = tmp_path / "features.npy"

    def interrupt(col):
        if col == 0:
            raise InterruptedError("test interruption")

    with pytest.raises(InterruptedError):
        data.random_features(x, path, 13, seed=7, feature_block=5, on_chunk=interrupt)
    chunk = path.with_suffix(".chunks_v3") / "chunk_00000000.npy"
    committed = chunk.stat().st_mtime_ns
    actual = data.random_features(x, path, 13, seed=7, feature_block=5)
    assert chunk.stat().st_mtime_ns == committed
    assert actual["attempts"] == 2
    assert actual["feature_cost_complete"]
    assert actual["feature_seconds"] >= actual["current_attempt_seconds"]
    reference = tmp_path / "reference.npy"
    data.random_features(x, reference, 13, seed=7, feature_block=3)
    np.testing.assert_allclose(np.load(path), np.load(reference), atol=1e-7)
    assert not np.load(path).flags.f_contiguous


def test_corrupted_chunk_is_never_reused(tmp_path):
    x = sp.eye(5, format="csr")
    path = tmp_path / "features.npy"

    def stop(col):
        raise InterruptedError

    with pytest.raises(InterruptedError):
        data.random_features(x, path, 10, feature_block=3, on_chunk=stop)
    chunk = path.with_suffix(".chunks_v3") / "chunk_00000000.npy"
    chunk.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        data.random_features(x, path, 10, feature_block=3)


def test_unknown_hard_kill_is_flagged_in_attempt_accounting(tmp_path):
    journal = Journal(tmp_path)
    journal.checkpoint()
    journal.close()  # Simulates a process exiting without a terminal record.
    resumed = Journal(tmp_path)
    assert not resumed.complete
    resumed.checkpoint("completed")
    assert resumed.known_seconds >= 0
    assert resumed.rows[0]["status"] == "unclean_stop"
    resumed.close()


def test_prepare_cache_retains_measured_setup_cost_and_source_identity(
    tmp_path, monkeypatch
):
    x = sp.csr_matrix(np.random.default_rng(3).normal(size=(20, 4)))
    y = np.tile([-1.0, 1.0], 10)
    monkeypatch.setattr(data, "load_raw", lambda *a, **k: (x, y, []))
    output = tmp_path / "source.json"
    first = data.prepare("cod-rna", tmp_path, output, features=13, feature_block=5)
    text = output.read_text()
    again = data.prepare("cod-rna", tmp_path, output, features=13, feature_block=5)
    assert output.read_text() == text
    assert first == again
    meta = next(iter(first.values()))["feature_provenance"]
    assert meta["preparation_seconds"] > 0
    assert meta["preparation_cost_complete"]
    assert meta["preparation_accounting_finalized"]
    assert not list((tmp_path / "nysadmm/features").glob("*.chunks_v3/chunk_*.npy"))
    assert json.loads(output.read_text()) == first


def test_shared_data_is_rejected_for_main_experiments(tmp_path):
    from releases.accounting_v3.experiments.problems import make_qp
    from releases.accounting_v3.experiments.run import execute

    p = make_qp(side=4)
    with pytest.raises(ValueError, match="main runs"):
        execute(small_config(phase="tune"), tmp_path, tmp_path, problem=p)
    with pytest.raises(ValueError, match="restricted"):
        execute(
            small_config(phase="tune", setup_policy="shared_pilot"),
            tmp_path,
            tmp_path,
            problem=p,
            shared_context=dict(id="x", seconds=1.0),
        )


def test_shared_pilot_reports_reference_without_inventing_cold_time(tmp_path):
    from releases.accounting_v3.experiments.problems import make_qp
    from releases.accounting_v3.experiments.run import execute

    row = execute(
        small_config(setup_policy="shared_pilot"),
        tmp_path,
        tmp_path,
        problem=make_qp(side=4),
        shared_context=dict(id="pilot", seconds=123.0),
    )
    assert not row["primary_eligible"]
    assert row["shared_setup_reference_seconds"] == 123.0
    assert row["total_seconds"] < 123.0
    assert np.isnan(row["end_to_end_seconds"])


def test_unknown_preprocessing_cost_is_not_reported_as_complete(tmp_path):
    from releases.accounting_v3.experiments.run import execute

    config = small_config(
        matrix_source=dict(
            feature_provenance=dict(
                preparation_seconds=123.0, preparation_cost_complete=False
            )
        )
    )
    row = execute(config, tmp_path, tmp_path)
    assert np.isfinite(row["total_seconds"])
    assert not row["preprocessing_cost_complete"]
    assert np.isnan(row["end_to_end_seconds"])


def test_pilot_timings_cannot_win_main_parameter_selection(tmp_path):
    from releases.accounting_v3.experiments.io import write_csv
    from releases.accounting_v3.experiments.manifest import winners

    write_csv(
        tmp_path / "pilot" / "result.csv",
        [
            dict(
                config=json.dumps(small_config(setup_policy="shared_pilot")),
                converged=True,
                total_seconds=0.001,
                primary_eligible=False,
                setup_policy="shared_pilot",
            )
        ],
    )
    with pytest.raises(ValueError, match="no finished"):
        winners(tmp_path)


def test_persistent_pilot_builds_each_problem_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from releases.accounting_v3.experiments import pilot_batch
    from releases.accounting_v3.experiments.io import identity, read_csv

    monkeypatch.setattr(pilot_batch.resource, "setrlimit", lambda *a: None)
    calls = []
    original = pilot_batch.build

    def capture(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(pilot_batch, "build", capture)
    rows = []
    for method in ["cg", "gd"]:
        config = small_config(inner=method, phase="pilot", setup_policy="shared_pilot")
        rows.append(dict(run_id=identity(config), config=json.dumps(config)))
    args = SimpleNamespace(
        results_dir=tmp_path, data_dir=tmp_path, memory_gb=1.0, setup_budget=10.0
    )
    pilot_batch.worker(args, rows)
    assert len(calls) == 1
    assert len(read_csv(tmp_path / "shared_setup.csv")) == 1
    for row in rows:
        result = read_csv(tmp_path / row["run_id"] / "result.csv")[0]
        assert result["primary_eligible"] == "False"
        assert result["setup_policy"] == "shared_pilot"


@pytest.mark.parametrize("namespace", [False, True])
def test_main_parent_records_independent_worker_wall_time(
    tmp_path, monkeypatch, namespace
):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from releases.accounting_v3.experiments.io import identity, read_csv, write_csv

    config = small_config()
    manifest = tmp_path / "manifest.csv"
    run_id = identity(config)
    write_csv(manifest, [dict(run_id=run_id, config=json.dumps(config))])
    release = Path(__file__).resolve().parents[1] / "releases/accounting_v3"
    env = dict(
        os.environ,
        SLURM_JOB_ID="test",
        SLURM_CPUS_PER_TASK="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "releases.accounting_v3.experiments.run"
            if namespace
            else "experiments.run",
            "--manifest",
            str(manifest),
            "--task-id",
            "0",
            "--results-dir",
            str(tmp_path / "results"),
            "--data-dir",
            str(tmp_path),
            "--memory-gb",
            "1",
        ],
        cwd=release.parents[1] if namespace else release,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    row = read_csv(tmp_path / "results" / run_id / "result.csv")[0]
    assert row["setup_policy"] == "independent"
    assert row["setup_accounting_complete"] == "True"
    assert row["attempts"] == "1"
    assert float(row["total_seconds"]) >= float(row["process_seconds"])
    assert float(row["cumulative_solver_setup_seconds"]) >= 0
    assert row["primary_eligible"] == "True"


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("family", ["lasso", "logistic"])
def test_fused_oracles_match_independent_formulas(tmp_path, resident, family):
    from scipy.special import expit

    from releases.accounting_v3.experiments.dense import DenseDesign
    from releases.accounting_v3.experiments.problems import make_regression

    a = np.random.default_rng(3).normal(size=(35, 19)).astype(np.float32)
    path = tmp_path / "a.npy"
    np.save(path, a)
    design = DenseDesign(path, block_rows=7, resident=resident).validate()
    p = make_regression(family, design, 0, normalization="none")
    x = np.random.default_rng(7).normal(size=p.n)
    gradient, value = p.gradient_and_value(x)
    dense = a.astype(float)
    if family == "lasso":
        coef = x[:19] - x[19:]
        residual = dense @ coef - p.target
        expected = dense.T @ residual
        np.testing.assert_allclose(
            gradient,
            np.r_[expected + p.regularization, -expected + p.regularization],
            atol=1e-12,
        )
    else:
        expected = -(dense.T @ (p.target * expit(-p.target * (dense @ x)))) / 35
        np.testing.assert_allclose(
            gradient, expected + p.regularization * x, atol=1e-12
        )
    assert value == pytest.approx(p.objective(x), abs=1e-12)
    fresh = p.metrics(x)
    reused = p.metrics(x, gradient=gradient, objective=value)
    for name in ["objective", "stationarity", "feasibility"]:
        assert reused[name] == pytest.approx(fresh[name], abs=1e-12)
    matrix = np.random.default_rng(8).normal(size=(19, 4))
    np.testing.assert_allclose(
        design.gram_product(matrix), dense.T @ (dense @ matrix), atol=1e-11
    )


def test_fused_metrics_do_not_read_design_again(tmp_path, monkeypatch):
    from releases.accounting_v3.experiments.dense import DenseDesign
    from releases.accounting_v3.experiments.problems import make_regression

    path = tmp_path / "a.npy"
    np.save(path, np.random.default_rng(3).normal(size=(20, 10)))
    p = make_regression("logistic", DenseDesign(path).validate(), 0)
    x = np.ones(p.n)
    gradient, value = p.gradient_and_value(x)
    monkeypatch.setattr(DenseDesign, "blocks", lambda self: pytest.fail("extra pass"))
    p.metrics(x, gradient=gradient, objective=value)


@pytest.mark.parametrize("family", ["lasso", "logistic"])
def test_fused_solver_converges_to_sparse_reference(tmp_path, family):
    from releases.accounting_v3.experiments.dense import DenseDesign
    from releases.accounting_v3.experiments.problems import make_regression
    from releases.accounting_v3.experiments.solvers import solve

    a = np.random.default_rng(8).normal(size=(32, 12))
    path = tmp_path / "a.npy"
    np.save(path, a)
    sparse = make_regression(family, sp.csr_matrix(a), 2)
    dense = make_regression(family, DenseDesign(path).validate(), 2)
    config = small_config(
        inner="gd" if family == "logistic" else "cg",
        tolerance=2e-5,
        max_outer=5000,
        max_inner=100000,
    )
    reference, _ = solve(sparse, config)
    actual, _ = solve(dense, config)
    assert actual["converged"] and reference["converged"]
    assert actual["objective"] == pytest.approx(reference["objective"], abs=2e-3)


def test_thread_policy_uses_whole_job_efficiency():
    from releases.accounting_v3.experiments.core_policy import apply, choose

    # Synthetic fixtures distinguish whole-job efficiency from kernel efficiency.
    reference = dict(
        state="COMPLETED", alloc_cpus=1, elapsed_s=100, total_cpu_s=99, nodes="test"
    )
    candidates = {
        "two": dict(
            state="COMPLETED", alloc_cpus=2, elapsed_s=60, total_cpu_s=115, nodes="test"
        ),
        "four": dict(
            state="COMPLETED", alloc_cpus=4, elapsed_s=40, total_cpu_s=130, nodes="test"
        ),
    }
    profiles = {
        name: [dict(converged="True", operation="solve")] * 2 for name in candidates
    }
    selected, _ = choose(reference, candidates, profiles)
    assert selected == "two"
    policy = dict(
        case=dict(family="qp", inner="nystrom_pcg", side=100, rank=128),
        threads=2,
        cpu_model="test",
        node="test",
    )
    assert apply(dict(policy["case"], blas_threads=1), policy)["blas_threads"] == 2
    assert (
        apply(dict(policy["case"], rank=32, blas_threads=1), policy)["blas_threads"]
        == 1
    )


def test_pilot_cache_keys_include_regularization():
    from releases.accounting_v3.experiments.pilot_batch import key

    assert key(small_config(lambda_ratio=0.1)) != key(small_config(lambda_ratio=0.2))


def test_default_plots_reject_shared_and_unknown_accounting():
    from plots.accounting import finite_time, main_eligible

    assert not main_eligible(dict(primary_eligible=False, setup_policy="shared_pilot"))
    assert not main_eligible(dict(setup_accounting_complete="False"))
    assert not finite_time(dict(total_seconds="nan"))
    assert main_eligible(dict(setup_policy="independent", primary_eligible="True"))


def test_unverified_legacy_output_is_preserved_and_rebuilt(tmp_path, monkeypatch):
    x = sp.csr_matrix(np.random.default_rng(3).normal(size=(20, 4)))
    monkeypatch.setattr(
        data, "load_raw", lambda *a, **k: (x, np.tile([-1.0, 1.0], 10), [])
    )
    output = tmp_path / "source.json"
    source = next(iter(data.prepare("cod-rna", tmp_path, output, features=13).values()))
    path = tmp_path / source["path"]
    original = path.read_bytes()
    path.with_suffix(".json").unlink()
    path.with_suffix(".chunks_v3").joinpath("progress.json").unlink()
    data.prepare("cod-rna", tmp_path, output, features=13)
    quarantined = list(path.parent.glob("*.unverified_*.npy"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == original
    assert path.read_bytes() == original
