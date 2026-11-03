import json

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from experiments.io import identity, read_csv
from experiments.manifest import group_key, instances, main
from experiments.matrices import freeze_spec, load_matrix, read_specs
from experiments.problems import build
from experiments.run import execute


def test_random_sources_are_sparse_reproducible_and_seeded(tmp_path):
    spec = dict(kind="random_sparse", rows=100, cols=1000000, entries_per_row=3, seed=7)
    cfg = dict(dataset="random", matrix_source=spec)
    a, meta = load_matrix(cfg, tmp_path)
    b, _ = load_matrix(cfg, tmp_path)
    assert a.shape == (100, 1000000)
    assert a.nnz <= 300
    assert a.data.nbytes + a.indices.nbytes + a.indptr.nbytes < 10000
    assert (a != b).nnz == 0
    c, _ = load_matrix(dict(cfg, matrix_source=dict(spec, seed=8)), tmp_path)
    assert (a != c).nnz > 0
    assert meta["kind"] == "random_sparse"


@pytest.mark.parametrize("fmt", ["npz", "mtx", "mtx.gz", "npy", "mat", "nested_mat"])
def test_local_format_adapters_preserve_matrix(tmp_path, fmt):
    a = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    suffix = "mat" if fmt == "nested_mat" else fmt
    path = tmp_path / f"matrix.{suffix}"
    spec = dict(kind="local", path=path.name, source="unit-test fixture")
    if fmt == "npz":
        sp.save_npz(path, sp.csr_matrix(a))
    elif fmt.startswith("mtx"):
        if fmt.endswith("gz"):
            import gzip

            with gzip.open(path, "wb") as f:
                scipy.io.mmwrite(f, sp.csr_matrix(a))
        else:
            scipy.io.mmwrite(path, sp.csr_matrix(a))
    elif fmt == "npy":
        np.save(path, a)
    elif fmt == "mat":
        scipy.io.savemat(path, {"A": a})
    else:
        scipy.io.savemat(path, {"Problem": {"A": sp.csr_matrix(a)}})
        spec["key"] = "Problem.A"
    frozen = freeze_spec(spec, tmp_path)
    matrix, metadata = load_matrix(
        dict(dataset="fixture", matrix_source=frozen), tmp_path
    )
    np.testing.assert_array_equal(matrix.toarray(), a)
    assert metadata["sha256"] == frozen["sha256"]


def test_mat_single_column_retains_orientation(tmp_path):
    scipy.io.savemat(tmp_path / "column.mat", {"A": np.array([[1.0], [2.0]])})
    a, _ = load_matrix(
        dict(dataset="column", matrix_source=dict(kind="local", path="column.mat")),
        tmp_path,
    )
    assert a.shape == (2, 1)


def test_manifest_hash_pins_input_and_detects_changes(tmp_path):
    path = tmp_path / "A.npz"
    sp.save_npz(path, sp.eye(3, format="csr"))
    spec = dict(kind="local", path="A.npz")
    frozen = freeze_spec(spec, tmp_path)
    cfg = dict(dataset="same_name", matrix_source=frozen)
    sp.save_npz(path, 2 * sp.eye(3, format="csr"))
    with pytest.raises(ValueError, match="checksum"):
        load_matrix(cfg, tmp_path)
    changed = dict(cfg, matrix_source=freeze_spec(spec, tmp_path))
    assert identity(cfg) != identity(changed)
    assert group_key(cfg) != group_key(changed)


@pytest.mark.parametrize(
    "a, error",
    [
        (np.array([[np.nan]]), "nonfinite"),
        (np.zeros((2, 3)), "nonzero"),
        (np.array([[1j]]), "real"),
    ],
)
def test_invalid_input_rejected(tmp_path, a, error):
    np.save(tmp_path / "bad.npy", a)
    with pytest.raises(ValueError, match=error):
        load_matrix(
            dict(dataset="bad", matrix_source=dict(kind="local", path="bad.npy")),
            tmp_path,
        )


def test_both_orientations_and_response_seeds_use_same_design(tmp_path):
    spec = dict(kind="random_sparse", rows=40, cols=16, entries_per_row=5, seed=12)
    catalog = {"alternative": spec}
    tasks = [
        c
        for c in instances(smoke=True, seeds=[0, 1], datasets=catalog)
        if c["family"] == "logistic"
    ]
    assert len(tasks) == 4
    native = build(tasks[0], tmp_path)
    transposed = build(tasks[1], tmp_path)
    repeat = build(tasks[2], tmp_path)
    assert native.A.shape == (40, 16)
    assert transposed.A.shape == (16, 40)
    assert (native.A != repeat.A).nnz == 0
    assert not np.array_equal(native.target, repeat.target)
    assert json.loads(native.provenance)["matrix"]["seed"] == 12


def test_custom_manifest_executes_without_dataset_catalog_at_runtime(
    tmp_path, monkeypatch
):
    specs = tmp_path / "sources.json"
    specs.write_text(
        json.dumps(
            {
                "custom": dict(
                    kind="random_sparse", rows=40, cols=16, entries_per_row=6, seed=8
                )
            }
        )
    )
    manifest = tmp_path / "tasks.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "manifest",
            "smoke",
            "--dataset-spec",
            str(specs),
            "--output",
            str(manifest),
            "--seeds",
            "0",
            "--families",
            "lasso",
            "--solvers",
            "cg",
            "--time-limit",
            "2",
        ],
    )
    main()
    cfg = json.loads(read_csv(manifest)[0]["config"])
    specs.unlink()
    result = execute(cfg, tmp_path / "result", tmp_path)
    assert result["converged"]
    assert json.loads(result["provenance"])["dataset"] == "custom"


def test_legacy_cache_and_smoke_still_load(tmp_path):
    sp.save_npz(tmp_path / "legacy.npz", sp.eye(4, format="csr"))
    a, _ = load_matrix(dict(dataset="legacy"), tmp_path)
    assert a.shape == (4, 4)
    smoke, _ = load_matrix(dict(dataset="smoke"), tmp_path)
    assert smoke.shape == (48, 24)


def test_missing_source_never_silently_falls_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_matrix(dict(dataset="missing"), tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        load_matrix(dict(dataset="x", matrix_source=dict(kind="unknown")), tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        load_matrix(
            dict(
                dataset="x", matrix_source=dict(kind="random_sparse", rows=0, cols=10)
            ),
            tmp_path,
        )


def test_catalog_loading_pins_files(tmp_path):
    sp.save_npz(tmp_path / "a.npz", sp.eye(3, format="csr"))
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"mine": dict(kind="local", path="a.npz")}))
    assert "sha256" in read_specs(path, tmp_path)["mine"]
