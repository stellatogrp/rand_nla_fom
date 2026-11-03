"""Durable RFF chunks, bounded local staging, and sequential final output."""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import digest


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    with open(temporary, "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def staging_context(staging, directory, chunk_bytes):
    if staging is None:
        return nullcontext(str(directory))
    staging.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(staging).free < 2 * chunk_bytes + 1024**3:
        raise OSError("node-local staging lacks chunk space and 1 GiB headroom")
    return tempfile.TemporaryDirectory(prefix="rnla-rff-", dir=staging)


def build(
    x,
    destination,
    features,
    gamma,
    seed,
    feature_block,
    row_block,
    dtype,
    work_dir=None,
    max_assembly_bytes=32 * 1024**3,
    prelude_seconds=0.0,
    on_chunk=None,
):
    destination = Path(destination)
    directory = destination.with_suffix(".chunks_v3")
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _build(
            x,
            destination,
            directory,
            features,
            gamma,
            seed,
            feature_block,
            row_block,
            dtype,
            work_dir,
            max_assembly_bytes,
            prelude_seconds,
            on_chunk,
        )


def _build(
    x,
    destination,
    directory,
    features,
    gamma,
    seed,
    feature_block,
    row_block,
    dtype,
    work_dir,
    max_assembly_bytes,
    prelude_seconds,
    on_chunk,
):
    started = perf_counter() - prelude_seconds
    required = x.shape[0] * features * np.dtype(dtype).itemsize
    if required > max_assembly_bytes:
        raise MemoryError("final feature assembly exceeds its RAM payload budget")
    fingerprint = hashlib.sha256()
    for array in (x.data, x.indices, x.indptr):
        fingerprint.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    spec = dict(
        implementation="accounting_v3",
        writer_version=1,
        shape=list(x.shape),
        x_sha256=fingerprint.hexdigest(),
        features=features,
        gamma=gamma,
        seed=seed,
        dtype=dtype,
        feature_block=feature_block,
    )
    journal = directory / "progress.json"
    if journal.exists():
        progress = json.loads(journal.read_text())
        if progress["spec"] != spec:
            raise ValueError("feature checkpoint parameters/input differ")
    else:
        if destination.exists():
            raise ValueError("unverified feature file lacks checkpoint provenance")
        progress = dict(spec=spec, chunks={}, attempts=[])
    for attempt in progress["attempts"]:
        if attempt["status"] == "running":
            attempt["status"] = "unclean_stop"
    attempt = dict(
        status="running",
        elapsed_seconds=0.0,
        job_id=os.environ.get("SLURM_JOB_ID", "local"),
    )
    progress["attempts"].append(attempt)

    def checkpoint(status="running"):
        attempt.update(status=status, elapsed_seconds=perf_counter() - started)
        atomic_json(journal, progress)

    checkpoint()
    staging = Path(work_dir) if work_dir else None
    try:
        if progress.get("output_sha256") and destination.exists():
            if digest(destination) != progress["output_sha256"]:
                raise ValueError("assembled feature checksum mismatch")
        else:
            chunk_bytes = (
                x.shape[0] * min(feature_block, features) * np.dtype(dtype).itemsize
            )
            with staging_context(
                staging, directory, chunk_bytes
            ) as temporary_directory:
                for col in range(0, features, feature_block):
                    count = min(feature_block, features - col)
                    key = str(col)
                    path = directory / f"chunk_{col:08d}.npy"
                    if key in progress["chunks"]:
                        if digest(path) != progress["chunks"][key]["sha256"]:
                            raise ValueError(
                                "committed feature chunk checksum mismatch"
                            )
                        continue
                    weights = np.empty((x.shape[1], count), order="F")
                    phases = np.empty(count)
                    for j in range(count):
                        rng = np.random.default_rng(
                            np.random.SeedSequence([seed, col + j])
                        )
                        weights[:, j] = np.sqrt(2 * gamma) * rng.standard_normal(
                            x.shape[1]
                        )
                        phases[j] = rng.uniform(0, 2 * np.pi)
                    chunk = np.empty((x.shape[0], count), dtype=dtype)
                    for row in range(0, x.shape[0], row_block):
                        values = np.asarray(x[row : row + row_block] @ weights)
                        values += phases
                        np.cos(values, out=values)
                        values *= np.sqrt(2 / features)
                        chunk[row : row + len(values)] = values
                    local = Path(temporary_directory) / (
                        "current.partial.npy" if staging else "current.commit.npy"
                    )
                    with open(local, "wb") as stream:
                        np.save(stream, chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
                    durable = directory / "current.commit.npy"
                    if local != durable:
                        with open(local, "rb") as source, open(durable, "wb") as target:
                            shutil.copyfileobj(source, target, length=8 * 1024**2)
                            target.flush()
                            os.fsync(target.fileno())
                        local.unlink()
                    os.replace(durable, path)
                    progress["chunks"][key] = dict(columns=count, sha256=digest(path))
                    checkpoint()
                    print(f"RFF committed columns {col + count}/{features}", flush=True)
                    if on_chunk:
                        on_chunk(col)
                # All strided assembly occurs in RAM. Only contiguous output
                # is written to GPFS; no full-file strided mmap conversion.
                array = np.empty((x.shape[0], features), dtype=dtype)
                for col in range(0, features, feature_block):
                    block = np.load(
                        directory / f"chunk_{col:08d}.npy", allow_pickle=False
                    )
                    array[:, col : col + block.shape[1]] = block
                temporary = destination.with_suffix(".assembled.partial.npy")
                with open(temporary, "wb") as stream:
                    np.save(stream, array)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
                progress["output_sha256"] = digest(destination)
        checkpoint("completed")
    except BaseException:
        checkpoint("interrupted")
        raise
    known = sum(a["elapsed_seconds"] for a in progress["attempts"])
    return dict(
        feature_seconds=known,
        current_attempt_seconds=attempt["elapsed_seconds"],
        feature_cost_complete=all(
            a["status"] != "unclean_stop" for a in progress["attempts"]
        ),
        attempts=len(progress["attempts"]),
        chunk_directory=str(directory),
        assembly="float32_or_float64_RAM_then_sequential_NPY",
        timing_scope="all_recorded_prepare_attempts",
    )


def cleanup(destination):
    directory = Path(destination).with_suffix(".chunks_v3")
    for path in directory.glob("chunk_*.npy"):
        path.unlink()
