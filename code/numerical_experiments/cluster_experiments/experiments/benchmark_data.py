"""Cached LIBSVM downloads and offline Rahimi-Recht RBF feature preparation."""

import argparse
import bz2
import json
import os
import urllib.request
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.io
import scipy.sparse as sp

from .data import digest
from .io import read_csv, write_csv

BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
CATALOG = {
    "cod-rna": dict(files=["binary/cod-rna"], rows=59535, cols=8, features=60000),
    "realsim": dict(
        files=["binary/real-sim.bz2"], rows=72309, cols=20958, features=50000
    ),
    "rcv1": dict(
        files=["binary/rcv1_train.binary.bz2"], rows=20242, cols=47236, features=30000
    ),
    "smallnorb": dict(
        files=["multiclass/smallNORB-32x32.mat"],
        rows=24300,
        cols=2048,
        features=30000,
        variant="LIBSVM downsampled 32x32 stereo",
        matrix_key="Z",
        label_key="y",
    ),
    "cifar10": dict(
        files=["multiclass/cifar10.mat", "multiclass/cifar10.t.mat"],
        rows=60000,
        cols=3072,
        features=60000,
        variant="LIBSVM train+test combined; optimization benchmark only",
        matrix_key="Z",
        label_key="y",
    ),
}


def download(name, directory):
    directory = Path(directory) / "nysadmm" / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    for relative in CATALOG[name]["files"]:
        path = directory / Path(relative).name
        provenance = path.with_suffix(path.suffix + ".source.csv")
        if path.exists() and provenance.exists():
            row = read_csv(provenance)[0]
            if digest(path) != row["sha256"]:
                raise ValueError(f"cached download checksum mismatch: {path}")
            print(f"cached: {path}", flush=True)
            continue
        if path.exists():
            raise ValueError(f"cached file lacks provenance: {path}")
        url = BASE + relative
        temporary = path.with_suffix(path.suffix + ".partial")
        count = 0
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "rand-nla-fom cached dataset preparation; no retries"
                },
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                final_url = response.url
                with open(temporary, "wb") as f:
                    while chunk := response.read(1024**2):
                        count += len(chunk)
                        if count > 1024**3:
                            raise ValueError("raw file exceeds 1 GiB safety ceiling")
                        f.write(chunk)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        write_csv(
            provenance,
            [
                dict(
                    dataset=name,
                    url=url,
                    final_url=final_url,
                    sha256=digest(path),
                    bytes=count,
                )
            ],
        )
        print(f"downloaded: {path} ({count} bytes)", flush=True)


def load_raw(name, data_dir, selected_rows=None):
    """Load verified source files, keeping labels separate from the design."""
    from sklearn.datasets import load_svmlight_file

    matrices, labels, sources = [], [], []
    spec = CATALOG[name]
    full_rows = 0
    for relative in spec["files"]:
        path = Path(data_dir) / "nysadmm/raw" / Path(relative).name
        row = read_csv(path.with_suffix(path.suffix + ".source.csv"))[0]
        if digest(path) != row["sha256"]:
            raise ValueError("raw data checksum mismatch")
        if path.suffix == ".mat":
            content = scipy.io.loadmat(path)
            # Use catalog keys verified from the downloaded MAT headers.
            matrix_key, label_key = (
                spec.get("matrix_key", "instance"),
                spec.get("label_key", "label"),
            )
            if matrix_key not in content or label_key not in content:
                raise ValueError(
                    f"{path}: expected LIBSVM MAT keys {matrix_key} and {label_key}; "
                    f"found {sorted(k for k in content if not k.startswith('_'))}"
                )
            raw, y = content[matrix_key], content[label_key].ravel()
            if raw.shape != (len(y), spec["cols"]):
                raise ValueError("raw MAT shape differs from label/feature dimensions")
            count = len(y)
            if selected_rows is not None:
                local = (
                    selected_rows[
                        (selected_rows >= full_rows)
                        & (selected_rows < full_rows + count)
                    ]
                    - full_rows
                )
                raw, y = raw[local], y[local]
            x = sp.csr_matrix(raw, dtype=float)
        else:
            opener = bz2.open if path.suffix == ".bz2" else open
            with opener(path, "rb") as f:
                x, y = load_svmlight_file(f, n_features=spec["cols"], zero_based=False)
            count = len(y)
            if selected_rows is not None:
                local = (
                    selected_rows[
                        (selected_rows >= full_rows)
                        & (selected_rows < full_rows + count)
                    ]
                    - full_rows
                )
                x, y = x[local], y[local]
        full_rows += count
        if x.shape[1] != spec["cols"] or len(y) != x.shape[0]:
            raise ValueError("raw feature/label dimensions differ from catalog")
        matrices.append(x)
        labels.append(y)
        sources.append(row)
    x, y = sp.vstack(matrices, format="csr"), np.concatenate(labels)
    expected_rows = spec["rows"] if selected_rows is None else len(selected_rows)
    if full_rows != spec["rows"] or x.shape != (expected_rows, spec["cols"]):
        raise ValueError(f"raw dimensions {x.shape} differ from verified catalog")
    if not np.all(np.isfinite(x.data)) or not np.all(np.isfinite(y)):
        raise ValueError("raw data contains nonfinite values")
    return x, y, sources


def subset_indices(y, count, seed):
    if count < len(np.unique(y)) or count > len(y):
        raise ValueError("row subset must fit the data and include every class")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 123]))
    mandatory = np.array(
        [rng.choice(np.flatnonzero(y == label)) for label in np.unique(y)]
    )
    remaining = np.setdiff1d(np.arange(len(y)), mandatory)
    return np.sort(
        np.r_[mandatory, rng.choice(remaining, count - len(mandatory), replace=False)]
    )


def random_features(
    x,
    destination,
    features,
    gamma=1.0,
    seed=0,
    feature_block=128,
    row_block=256,
    preprocessing="row_l2",
    dtype="float32",
):
    """z_j(x)=sqrt(2/D) cos(w_j.T x+b_j), w_j~N(0,2 gamma I).

    Per-feature RNGs make the map independent of computational tile sizes.
    Dense output is a disk-backed NPY file; no Gram matrix is constructed.
    """
    if (
        not isinstance(features, int)
        or features < 1
        or gamma <= 0
        or not np.isfinite(gamma)
        or feature_block < 1
        or row_block < 1
        or seed < 0
    ):
        raise ValueError("invalid random feature parameters")
    if dtype not in ("float32", "float64"):
        raise ValueError("feature dtype must be float32 or float64")
    x = sp.csr_matrix(x, dtype=float)
    if not x.nnz or not np.all(np.isfinite(x.data)):
        raise ValueError("feature inputs must be nonzero and finite")
    if preprocessing == "row_l2":
        norms = np.sqrt(np.asarray(x.power(2).sum(axis=1)).ravel())
        x = (sp.diags(1 / np.maximum(norms, 1e-15)) @ x).tocsr()
    elif preprocessing != "none":
        raise ValueError("input preprocessing must be row_l2 or none")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.npy")
    tiles = destination.with_suffix(".tiles.npy")
    started = perf_counter()
    out = np.lib.format.open_memmap(
        tiles, mode="w+", dtype=dtype, shape=(x.shape[0], features), fortran_order=True
    )
    try:
        for col in range(0, features, feature_block):
            count = min(feature_block, features - col)
            weights = np.empty((x.shape[1], count), order="F")
            phases = np.empty(count)
            for j in range(count):
                rng = np.random.default_rng(np.random.SeedSequence([seed, col + j]))
                weights[:, j] = np.sqrt(2 * gamma) * rng.standard_normal(x.shape[1])
                phases[j] = rng.uniform(0, 2 * np.pi)
            for row in range(0, x.shape[0], row_block):
                values = np.asarray(x[row : row + row_block] @ weights)
                values += phases
                np.cos(values, out=values)
                values *= np.sqrt(2 / features)
                out[row : row + len(values), col : col + count] = values
        out.flush()
        # Column tiles are contiguous in the intermediate file. Writing them
        # directly into a C-order mmap repeatedly dirties pages across the full
        # matrix. Convert once to the C-order layout used by solver workers.
        with open(temporary, "wb") as f:
            np.lib.format.write_array_header_1_0(
                f,
                dict(
                    descr=np.dtype(dtype).str,
                    fortran_order=False,
                    shape=(x.shape[0], features),
                ),
            )
            for row in range(0, x.shape[0], row_block):
                f.write(out[row : row + row_block].tobytes(order="C"))
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        del out
        temporary.unlink(missing_ok=True)
        tiles.unlink(missing_ok=True)
        raise
    del out
    tiles.unlink()
    os.replace(temporary, destination)
    return dict(
        features=features,
        gamma=gamma,
        seed=seed,
        preprocessing=preprocessing,
        dtype=dtype,
        feature_block=feature_block,
        row_block=row_block,
        feature_seconds=perf_counter() - started,
        formula="sqrt(2/D)*cos(XW+b); W_ij~N(0,2*gamma); b_j~Uniform(0,2*pi)",
    )


def prepare(
    name,
    data_dir,
    output,
    features=None,
    gamma=1.0,
    seed=0,
    max_rows=None,
    feature_block=128,
    row_block=256,
    positive_class=1.0,
    preprocessing="row_l2",
    dtype="float32",
):
    started = perf_counter()
    if max_rows is not None and "matrix_key" in CATALOG[name]:
        spec = CATALOG[name]
        all_y = np.concatenate(
            [
                scipy.io.loadmat(
                    Path(data_dir) / "nysadmm/raw" / Path(relative).name,
                    variable_names=[spec["label_key"]],
                )[spec["label_key"]].ravel()
                for relative in spec["files"]
            ]
        )
        selected = subset_indices(all_y, max_rows, seed)
        x, y, sources = load_raw(name, data_dir, selected_rows=selected)
    else:
        x, y, sources = load_raw(name, data_dir)
        if max_rows is not None:
            indices = subset_indices(y, max_rows, seed)
            x, y = x[indices], y[indices]
    features = features or CATALOG[name]["features"]
    run_spec = dict(
        dataset=name,
        features=features,
        gamma=gamma,
        seed=seed,
        rows=x.shape[0],
        raw_features=x.shape[1],
        raw_sha256=[source["sha256"] for source in sources],
        variant=CATALOG[name].get("variant", "LIBSVM native training set"),
        row_selection="all"
        if max_rows is None
        else "one-per-class plus seeded uniform",
        preprocessing=preprocessing,
        dtype=dtype,
    )
    from .io import identity

    tag = f"{name}_rf_{identity(run_spec)[:12]}"
    directory = Path(data_dir) / "nysadmm/features"
    path = directory / f"{tag}.npy"
    metadata_path = directory / f"{tag}.json"
    prior = None
    if path.exists():
        if not metadata_path.exists():
            raise ValueError("existing feature file lacks provenance")
        prior = json.loads(metadata_path.read_text())
        if prior["spec"] != run_spec or digest(path) != prior["sha256"]:
            raise ValueError("cached feature provenance/checksum mismatch")
        feature_info = prior["features"]
    else:
        feature_info = random_features(
            x,
            path,
            features,
            gamma,
            seed,
            feature_block,
            row_block,
            preprocessing=preprocessing,
            dtype=dtype,
        )
    labels_path = directory / f"{tag}.labels.npy"
    if labels_path.exists():
        if not np.array_equal(np.load(labels_path, allow_pickle=False), y):
            raise ValueError("cached label mismatch")
    else:
        temporary_labels = labels_path.with_suffix(".partial.npy")
        np.save(temporary_labels, y)
        os.replace(temporary_labels, labels_path)
    metadata = prior or dict(
        spec=run_spec,
        sources=sources,
        features=feature_info,
        sha256=digest(path),
        labels_sha256=digest(labels_path),
        preparation_seconds=perf_counter() - started,
        replication="explicit kernel/scaling/class map; not an exact reproduction",
    )
    if prior and (
        metadata["sources"] != sources
        or metadata["labels_sha256"] != digest(labels_path)
    ):
        raise ValueError("cached feature source/label provenance mismatch")
    if prior is None:
        temporary_metadata = metadata_path.with_suffix(".partial.json")
        temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary_metadata, metadata_path)
    source = dict(
        kind="local",
        path=str(path.relative_to(data_dir)),
        format="npy",
        storage="streamed_dense",
        block_rows=row_block,
        sha256=metadata["sha256"],
        labels_path=str(labels_path.relative_to(data_dir)),
        labels_sha256=metadata["labels_sha256"],
        response="native",
        positive_class=positive_class,
        normalization="none",
        orientations=["native"],
        source="LIBSVM; NysADMM dataset family; see feature provenance",
        feature_provenance=metadata,
        lasso_lambda=1.0,
    )
    write = {tag: source}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(write, indent=2) + "\n")
    print(f"prepared {x.shape[0]} x {features}: {output}", flush=True)
    return write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["download", "prepare", "inspect"])
    parser.add_argument("--datasets", nargs="+", choices=CATALOG, default=["cod-rna"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("manifests/nysadmm_sources.json")
    )
    parser.add_argument("--features", type=int)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--feature-block", type=int, default=128)
    parser.add_argument("--row-block", type=int, default=256)
    parser.add_argument("--positive-class", type=float, default=1.0)
    parser.add_argument("--preprocessing", choices=["row_l2", "none"], default="row_l2")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="also save source specs and preparation CSV for result syncing",
    )
    args = parser.parse_args()
    if args.mode == "download" and os.environ.get("SLURM_JOB_ID"):
        parser.error("download on an internet-enabled host, not a compute node")
    if args.mode == "prepare" and not os.environ.get("SLURM_JOB_ID"):
        if (
            not args.allow_local
            or not args.max_rows
            or args.max_rows > 512
            or (not args.features or args.features > 512)
        ):
            parser.error(
                "full feature preparation requires Slurm; local pilots need "
                "--allow-local --max-rows <=512 --features <=512"
            )
    all_sources = {}
    for name in args.datasets:
        if args.mode == "download":
            download(name, args.data_dir)
        elif args.mode == "inspect":
            for relative in CATALOG[name]["files"]:
                path = args.data_dir / "nysadmm/raw" / Path(relative).name
                print(
                    path,
                    scipy.io.whosmat(path)
                    if path.suffix == ".mat"
                    else read_csv(path.with_suffix(path.suffix + ".source.csv")),
                )
        else:
            temporary_spec = args.output.with_suffix(f".{name}.json")
            all_sources.update(
                prepare(
                    name,
                    args.data_dir,
                    temporary_spec,
                    args.features,
                    args.gamma,
                    args.seed,
                    args.max_rows,
                    args.feature_block,
                    args.row_block,
                    args.positive_class,
                    args.preprocessing,
                    args.dtype,
                )
            )
    if args.mode == "prepare":
        args.output.write_text(json.dumps(all_sources, indent=2) + "\n")
        if args.report_dir:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / "sources.json").write_text(
                json.dumps(all_sources, indent=2) + "\n"
            )
            write_csv(
                args.report_dir / "preparation.csv",
                [
                    dict(
                        dataset=source["feature_provenance"]["spec"]["dataset"],
                        rows=source["feature_provenance"]["spec"]["rows"],
                        features=source["feature_provenance"]["spec"]["features"],
                        preparation_seconds=source["feature_provenance"][
                            "preparation_seconds"
                        ],
                        sha256=source["sha256"],
                    )
                    for source in all_sources.values()
                ],
            )


if __name__ == "__main__":
    main()
