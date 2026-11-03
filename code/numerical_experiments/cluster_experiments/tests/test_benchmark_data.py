import json

import numpy as np
import pytest
import scipy.sparse as sp

from experiments import benchmark_data
from experiments.baselines import external
from experiments.dense import DenseDesign
from experiments.matrices import load_matrix, read_specs
from experiments.problems import build, make_regression, norm_bound
from experiments.solvers import solve


def test_streamed_products_and_normalization(tmp_path, monkeypatch):
    a = np.random.default_rng(4).normal(size=(11, 7)).astype("float32")
    path = tmp_path / "dense.npy"
    np.save(path, a)
    # Loading this representation must never map the full file into memory.
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("full-file load")
    )
    design, _ = load_matrix(
        dict(
            dataset="dense",
            matrix_source=dict(
                kind="local", path="dense.npy", storage="streamed_dense", block_rows=3
            ),
        ),
        tmp_path,
    )
    assert isinstance(design, DenseDesign)
    np.testing.assert_allclose(design @ np.ones(7), a.astype(float) @ np.ones(7))
    np.testing.assert_allclose(design.T @ np.ones(11), a.astype(float).T @ np.ones(11))
    np.testing.assert_allclose(design @ np.eye(7), a)
    np.testing.assert_allclose(design.T @ np.eye(11), a.T)
    np.testing.assert_allclose(
        design.normalized().to_sparse().toarray(),
        a / np.linalg.norm(a, axis=0),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        design.T.normalized().to_sparse().toarray(),
        a.T / np.linalg.norm(a.T, axis=0),
        rtol=1e-6,
    )
    assert norm_bound(design) >= np.linalg.norm(a, 2) ** 2
    with pytest.raises(MemoryError, match="budget"):
        design.to_sparse(max_bytes=1)


def test_random_feature_formula_and_tile_independence(tmp_path):
    x = sp.csr_matrix(np.random.default_rng(2).normal(size=(9, 4)))
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    benchmark_data.random_features(
        x, a, 13, gamma=0.3, seed=7, feature_block=3, row_block=2, preprocessing="none"
    )
    benchmark_data.random_features(
        x, b, 13, gamma=0.3, seed=7, feature_block=7, row_block=5, preprocessing="none"
    )
    np.testing.assert_allclose(np.load(a), np.load(b), atol=1e-7)
    reference = []
    for j in range(13):
        rng = np.random.default_rng(np.random.SeedSequence([7, j]))
        w = np.sqrt(0.6) * rng.standard_normal(4)
        phase = rng.uniform(0, 2 * np.pi)
        reference.append(np.sqrt(2 / 13) * np.cos(x @ w + phase))
    np.testing.assert_allclose(np.load(a), np.asarray(reference).T, atol=1e-7)
    with pytest.raises(ValueError, match="parameters"):
        benchmark_data.random_features(x, a, 0)


def test_feature_inner_products_approximate_rbf_kernel(tmp_path):
    x = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0], [0.2, 0.3]])
    path = tmp_path / "features.npy"
    benchmark_data.random_features(
        x, path, 10000, gamma=0.5, seed=13, preprocessing="none", feature_block=500
    )
    z = np.load(path)
    dense = x.toarray()
    kernel = np.exp(-0.5 * np.sum((dense[:, None] - dense[None, :]) ** 2, axis=2))
    np.testing.assert_allclose(z @ z.T, kernel, atol=0.035)


def test_prepared_native_labels_and_checksums(tmp_path, monkeypatch):
    x = sp.csr_matrix(np.random.default_rng(4).normal(size=(40, 5)))
    y = np.tile([-1.0, 1.0], 20)
    monkeypatch.setattr(benchmark_data, "load_raw", lambda *args: (x, y, []))
    output = tmp_path / "sources.json"
    benchmark_data.prepare("cod-rna", tmp_path, output, features=12, max_rows=20)
    first_spec = output.read_text()
    benchmark_data.prepare("cod-rna", tmp_path, output, features=12, max_rows=20)
    assert output.read_text() == first_spec  # cache hits preserve source/task identity
    spec = next(iter(read_specs(output, tmp_path).values()))
    cfg = dict(
        family="logistic",
        seed=0,
        dataset="pilot",
        orientation="native",
        matrix_source=spec,
    )
    p = build(cfg, tmp_path)
    labels = np.load(tmp_path / spec["labels_path"])
    np.testing.assert_array_equal(p.target, labels)
    assert set(p.target) == {-1.0, 1.0}
    assert isinstance(p.A, DenseDesign)
    assert json.loads(p.provenance)["response"] == "native"
    # Native labels must never be reused for a transposed design.
    with pytest.raises(ValueError, match="labels"):
        build(dict(cfg, orientation="transpose"), tmp_path)
    np.save(tmp_path / spec["labels_path"], np.zeros(20))
    with pytest.raises(ValueError, match="checksum"):
        build(cfg, tmp_path)


@pytest.mark.parametrize("family", ["lasso", "logistic"])
def test_streamed_design_solver_and_external_baseline(family, tmp_path):
    a = np.random.default_rng(8).normal(size=(32, 12))
    path = tmp_path / "matrix.npy"
    np.save(path, a)
    p = make_regression(family, DenseDesign(path, block_rows=5).validate(), 2)
    config = dict(
        seed=0,
        inner="gd" if family == "logistic" else "cg",
        ratio=10.0,
        sigma=0.2,
        stride=2,
        tolerance=2e-5,
        time_limit=10.0,
        max_inner=100000,
        max_outer=5000,
    )
    ours, _ = solve(p, config)
    baseline, _ = external(p, dict(config, inner="scs"))
    assert ours["converged"]
    assert abs(ours["objective"] - baseline["objective"]) < 2e-3
    limited, _ = external(p, dict(config, inner="scs", max_explicit_matrix_gb=1e-10))
    assert limited["status"] == "representation_memory_limit"


def test_multiclass_mapping_is_explicit():
    a = sp.eye(4, format="csr")
    labels = np.array([0.0, 1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="positive_class"):
        make_regression("logistic", a, 0, targets=labels)
    p = make_regression("logistic", a, 0, targets=labels, positive_class=1)
    np.testing.assert_array_equal(p.target, [-1, 1, -1, 1])


def test_preparation_report_preserves_source_spec(tmp_path, monkeypatch):
    import sys

    from experiments.io import read_csv

    x = sp.csr_matrix(np.random.default_rng(4).normal(size=(20, 5)))
    y = np.tile([-1.0, 1.0], 10)
    monkeypatch.setattr(benchmark_data, "load_raw", lambda *args: (x, y, []))
    output, report = tmp_path / "sources.json", tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "prepare",
            "--datasets",
            "cod-rna",
            "--features",
            "12",
            "--max-rows",
            "20",
            "--allow-local",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--report-dir",
            str(report),
        ],
    )
    benchmark_data.main()
    assert (report / "sources.json").read_text() == output.read_text()
    row = read_csv(report / "preparation.csv")[0]
    assert (row["dataset"], row["rows"], row["features"]) == ("cod-rna", "20", "12")
