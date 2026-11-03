"""Download on an internet-enabled host; convert only in a compute allocation."""

import argparse
import csv
import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

CATALOG = {
    "GL7d16": ("JGD_GL7d", 955128, 460261, 14488881),
    "lp_osa_30": ("LPnetlib", 4350, 104374, 604488),
    "lp_ken_18": ("LPnetlib", 105127, 154699, 358171),
    "GL7d12": ("JGD_GL7d", 8899, 1019, 37519),
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def prepare(name, directory, mode):
    group, rows, cols, nnz = CATALOG[name]
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / f"{name}.mat"
    source = f"https://sparse.tamu.edu/mat/{group}/{name}.mat"
    if mode == "download":
        if not raw.exists():
            temporary = raw.with_suffix(".mat.tmp")
            with urllib.request.urlopen(source, timeout=120) as response:
                with open(temporary, "wb") as f:
                    while chunk := response.read(8 * 1024**2):
                        f.write(chunk)
            os.replace(temporary, raw)
        checksum = digest(raw)
        path = directory / f"{name}.source.csv"
        if path.exists():
            with open(path, newline="") as f:
                prior = next(csv.DictReader(f))
            if prior["sha256"] != checksum:
                raise ValueError("cached download differs from recorded checksum")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "url", "sha256"])
            writer.writeheader()
            writer.writerow(dict(name=name, url=source, sha256=checksum))
        return
    # .mat is downloaded intact; no archive extraction or arbitrary paths.
    with open(directory / f"{name}.source.csv", newline="") as f:
        provenance = next(csv.DictReader(f))
    if digest(raw) != provenance["sha256"]:
        raise ValueError("download checksum mismatch")
    problem = scipy.io.loadmat(raw, struct_as_record=False, squeeze_me=True)["Problem"]
    matrix = sp.csr_matrix(problem.A, dtype=float)
    matrix.eliminate_zeros()
    if matrix.shape != (rows, cols) or matrix.nnz != nnz:
        raise ValueError(f"SuiteSparse metadata mismatch: {matrix.shape}, {matrix.nnz}")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("nonfinite matrix")
    destination = directory / f"{name}.npz"
    temporary = directory / f"{name}.tmp.npz"
    sp.save_npz(temporary, matrix)
    os.replace(temporary, destination)
    with open(directory / f"{name}.matrix.csv", "w", newline="") as f:
        fields = ["name", "url", "sha256", "npz_sha256", "rows", "cols", "nnz"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            dict(
                **provenance,
                npz_sha256=digest(destination),
                rows=rows,
                cols=cols,
                nnz=nnz,
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["download", "convert"])
    parser.add_argument(
        "--datasets", nargs="+", choices=CATALOG, default=["GL7d16", "lp_osa_30"]
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.mode == "download" and os.environ.get("SLURM_JOB_ID"):
        parser.error("compute nodes have no internet; download before submitting")
    if args.mode == "convert" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("conversion must run in a Slurm compute allocation")
    for name in args.datasets:
        prepare(name, args.data_dir, args.mode)
        print(f"{args.mode}: {name}", flush=True)


if __name__ == "__main__":
    main()
