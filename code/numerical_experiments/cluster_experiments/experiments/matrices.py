"""Matrix source adapters shared by every regression family and sweep.

Specs are plain JSON objects embedded in task manifests. Add an adapter with
register_source; it must return a sparse matrix and JSON-serializable metadata.
No adapter downloads data during a solver run.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

from .data import digest
from .dense import DenseDesign
from .io import write_csv

LOADERS = {}


def register_source(name, loader):
    if name in LOADERS:
        raise ValueError(f"matrix source already registered: {name}")
    LOADERS[name] = loader


def validate(matrix):
    if isinstance(matrix, DenseDesign):
        return matrix.validate()
    if np.iscomplexobj(matrix):
        raise ValueError("design matrices must be real")
    matrix = sp.csr_matrix(matrix, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if min(matrix.shape) < 1 or not matrix.nnz:
        raise ValueError("design matrix must be nonempty and nonzero")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("design matrix contains nonfinite values")
    return matrix


def _path(spec, data_dir):
    path = Path(spec["path"])
    return path if path.is_absolute() else Path(data_dir) / path


def _local(spec, data_dir):
    path = _path(spec, data_dir)
    checksum = digest(path)
    if spec.get("sha256") and checksum != spec["sha256"]:
        raise ValueError(f"matrix checksum mismatch: {path}")
    fmt = spec.get("format", path.suffix.lstrip(".")).lower()
    if fmt == "npz":
        matrix = sp.load_npz(path)
    elif fmt in ("mtx", "mtx.gz") or path.name.endswith(".mtx.gz"):
        matrix = scipy.io.mmread(path)
    elif fmt == "npy":
        if spec.get("storage") in ("streamed_dense", "resident_dense"):
            matrix = DenseDesign(
                path,
                spec.get("block_rows", 128),
                resident=spec["storage"] == "resident_dense",
                max_resident_bytes=int(spec.get("max_resident_gb", 32.0) * 1024**3),
            )
        else:
            matrix = np.load(path, mmap_mode="r", allow_pickle=False)
            if matrix.ndim != 2:
                raise ValueError("NPY input must be a two-dimensional matrix")
    elif fmt == "mat":
        content = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=False)
        key = spec.get("key", "A")
        value = content
        for part in key.split("."):
            if (
                isinstance(value, np.ndarray)
                and value.dtype == object
                and value.size == 1
            ):
                value = value.item()
            value = value[part] if isinstance(value, dict) else getattr(value, part)
        matrix = value
    else:
        raise ValueError(f"unsupported matrix format: {fmt}")
    return matrix, dict(
        kind="local",
        path=str(path),
        sha256=checksum,
        source=spec.get("source", "user-supplied"),
        format=fmt,
    )


def _random_sparse(spec, data_dir):
    rows, cols = spec["rows"], spec["cols"]
    degree = spec.get("entries_per_row", 20)
    for name, value in [("rows", rows), ("cols", cols), ("entries_per_row", degree)]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    seed = spec.get("seed", 314159)
    rng = np.random.default_rng(seed)
    count = rows * degree
    indices = rng.integers(
        cols, size=count, dtype=np.int32 if cols < 2**31 else np.int64
    )
    distribution = spec.get("distribution", "normal")
    if distribution == "normal":
        values = rng.standard_normal(count)
    elif distribution == "rademacher":
        values = (2 * rng.integers(2, size=count) - 1).astype(float)
    else:
        raise ValueError(f"unsupported random distribution: {distribution}")
    values /= np.sqrt(degree)
    indptr = np.arange(rows + 1, dtype=np.int64) * degree
    # Sampling with replacement avoids O(rows*cols) allocation. Duplicate
    # columns are summed, so actual nnz can be below rows*entries_per_row.
    matrix = sp.csr_matrix((values, indices, indptr), shape=(rows, cols))
    return matrix, dict(
        kind="random_sparse",
        seed=seed,
        rows=rows,
        cols=cols,
        entries_per_row=degree,
        distribution=distribution,
        sampling="with_replacement",
        numpy_version=np.__version__,
    )


def _cached(spec, data_dir):
    path = Path(data_dir) / f"{spec['name']}.npz"
    return _local(
        dict(
            path=str(path.resolve()),
            sha256=spec.get("sha256"),
            source=spec.get("source", "cached-matrix"),
        ),
        data_dir,
    )


register_source("local", _local)
register_source("random_sparse", _random_sparse)
register_source("cached", _cached)


def freeze_spec(spec, data_dir):
    """Pin local input bytes before emitting task IDs; paths stay relocatable."""
    spec = dict(spec)
    if spec.get("kind") not in LOADERS:
        raise ValueError(f"unknown matrix source: {spec.get('kind')}")
    if spec["kind"] == "local":
        checksum = digest(_path(spec, data_dir))
        if spec.get("sha256") and spec["sha256"] != checksum:
            raise ValueError("matrix checksum differs from dataset specification")
        spec["sha256"] = checksum
    elif spec["kind"] == "cached":
        checksum = digest(Path(data_dir) / f"{spec['name']}.npz")
        if spec.get("sha256") and spec["sha256"] != checksum:
            raise ValueError("matrix checksum differs from dataset specification")
        spec["sha256"] = checksum
    if "labels_path" in spec:
        labels = dict(path=spec["labels_path"])
        checksum = digest(_path(labels, data_dir))
        if spec.get("labels_sha256") and spec["labels_sha256"] != checksum:
            raise ValueError("label checksum differs from dataset specification")
        spec["labels_sha256"] = checksum
    return spec


def read_specs(path, data_dir):
    with open(path) as f:
        catalog = json.load(f)
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("dataset specification must be a nonempty name-to-spec object")
    for name, spec in catalog.items():
        if not isinstance(spec, dict):
            raise ValueError(f"dataset {name} must have an object specification")
        catalog[name] = freeze_spec(spec, data_dir)
    return catalog


def load_matrix(config, data_dir):
    spec = config.get("matrix_source")
    if spec is None:
        if config["dataset"] == "smoke":
            # Preserve the original smoke manifests exactly.
            matrix = sp.csr_matrix(np.random.default_rng(314).normal(size=(48, 24)))
            return matrix, dict(kind="legacy_smoke", seed=314)
        spec = dict(kind="cached", name=config["dataset"], source="legacy-cache")
    kind = spec.get("kind")
    if kind not in LOADERS:
        raise ValueError(f"unknown matrix source: {kind}")
    matrix, metadata = LOADERS[kind](spec, data_dir)
    matrix = validate(matrix)
    if "shape" in spec and list(matrix.shape) != spec["shape"]:
        raise ValueError("matrix shape differs from dataset specification")
    metadata.update(
        rows=matrix.shape[0], cols=matrix.shape[1], nnz=matrix.nnz, spec=spec
    )
    return matrix, metadata


def main():
    parser = argparse.ArgumentParser(description="Validate matrix sources offline")
    parser.add_argument("--dataset-spec", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/matrices.csv"))
    parser.add_argument(
        "--allow-local",
        action="store_true",
        help="validate small sources on a workstation, not a login node",
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") and not args.allow_local:
        parser.error(
            "validate large sources in Slurm; --allow-local is for small tests"
        )
    rows = []
    for name, spec in read_specs(args.dataset_spec, args.data_dir).items():
        _, metadata = load_matrix(dict(dataset=name, matrix_source=spec), args.data_dir)
        rows.append(
            dict(
                dataset=name,
                rows=metadata["rows"],
                cols=metadata["cols"],
                nnz=metadata["nnz"],
                provenance=json.dumps(metadata, sort_keys=True),
            )
        )
    write_csv(args.output, rows)
    print(f"validated {len(rows)} sources -> {args.output}")


if __name__ == "__main__":
    main()
