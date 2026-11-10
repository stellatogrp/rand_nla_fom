"""Bounded CPU-scaling and resident/streamed benchmarks on compute nodes."""

import argparse
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from . import solvers
from .data import digest
from .io import read_csv, write_csv
from .matrices import read_specs
from .problems import build, make_qp
from .runtime import threads
from .tuning import resolve


def cpu_model():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def benchmark(args):
    count = threads(dict(blas_threads=args.threads))
    if sys.platform == "linux":
        memory = int(args.memory_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    started = perf_counter()
    if args.kind == "qp":
        p = make_qp(side=args.side)
        dataset = f"qp_{args.side}"
    else:
        if not args.source_spec:
            raise ValueError("regression profiling requires --source-spec")
        catalog = read_specs(args.source_spec, args.data_dir)
        if len(catalog) != 1:
            raise ValueError("profile one dataset per job")
        dataset, spec = next(iter(catalog.items()))
        spec.update(storage=args.storage, max_resident_gb=args.max_resident_gb)
        p = build(
            dict(
                family=args.family,
                dataset=dataset,
                matrix_source=spec,
                seed=0,
                orientation="native",
            ),
            args.data_dir,
        )
    loading = perf_counter() - started
    rows = []
    solvers.install_signals()
    with threadpool_limits(limits=count):
        for repeat in range(args.repeats):
            if solvers.INTERRUPTED or perf_counter() - started >= args.time_limit:
                break
            tick, cpu_tick = perf_counter(), process_time()
            if args.operation == "gradient":
                x = np.random.default_rng(repeat).normal(size=p.n) / np.sqrt(p.n)
                g = p.gradient(x)
                metrics = p.metrics(x, gradient=g)
                outcome = dict(status="measured", **metrics)
            else:
                config = dict(
                    seed=repeat,
                    inner="gd" if p.family == "logistic" else args.inner,
                    rank=args.rank,
                    ratio=10.0,
                    sigma=0.2,
                    stride=1,
                    time_limit=min(120.0, args.time_limit),
                    max_inner=1000,
                    max_outer=2000,
                    tolerance=1e-6,
                )
                config, tuning, inner = resolve(p, config, return_inner=True)
                outcome, _ = solvers.solve(p, config, prepared_inner=inner)
                outcome["tuning_seconds"] = tuning["tuning_seconds"]
            elapsed, cpu = perf_counter() - tick, process_time() - cpu_tick
            rows.append(
                dict(
                    kind=args.kind,
                    dataset=dataset,
                    family=p.family,
                    storage=args.storage if args.kind == "regression" else "sparse",
                    operation=args.operation,
                    inner="gd" if p.family == "logistic" else args.inner,
                    repeat=repeat,
                    threads=count,
                    seconds=elapsed,
                    cpu_seconds=cpu,
                    cpu_efficiency=cpu / max(elapsed * count, 1e-12),
                    loading_seconds=loading,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                    hostname=platform.node(),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    cpu_model=cpu_model(),
                    threadpools=json.dumps(threadpool_info(), sort_keys=True),
                    source_sha256=json.dumps(
                        {
                            str(path): digest(path)
                            for path in sorted(Path("experiments").glob("*.py"))
                        },
                        sort_keys=True,
                    ),
                    benchmark_status="completed",
                    **outcome,
                )
            )
            write_csv(args.output, rows)
            if solvers.INTERRUPTED:
                break
    if len(rows) != args.repeats:
        raise RuntimeError("profiling interrupted before all repeats completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["qp", "regression"], required=True)
    parser.add_argument("--source-spec", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--family", choices=["lasso", "logistic"], default="logistic")
    parser.add_argument(
        "--storage",
        choices=["streamed_dense", "resident_dense"],
        default="resident_dense",
    )
    parser.add_argument("--operation", choices=["solve", "gradient"], default="solve")
    parser.add_argument("--inner", choices=["gd", "nystrom_pcg"], default="nystrom_pcg")
    parser.add_argument("--side", type=int, default=100)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--memory-gb", type=float, default=12.0)
    parser.add_argument("--max-resident-gb", type=float, default=32.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("profiling requires a Slurm compute allocation")
    if any(
        not math.isfinite(v) or v <= 0
        for v in [args.time_limit, args.memory_gb, args.max_resident_gb, args.repeats]
    ):
        parser.error("all profiling budgets must be finite and positive")
    threads(dict(blas_threads=args.threads))
    if args.worker:
        benchmark(args)
        return
    child = subprocess.Popen(
        [sys.executable, "-m", "experiments.performance", *sys.argv[1:], "--worker"]
    )

    def forward(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGUSR1, forward)
    signal.signal(signal.SIGTERM, forward)
    try:
        child.wait(timeout=args.time_limit)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        rows = read_csv(args.output)
        rows.append(
            dict(
                benchmark_status="worker_timeout",
                threads=args.threads,
                kind=args.kind,
                storage=args.storage,
                operation=args.operation,
            )
        )
        write_csv(args.output, rows)
        raise SystemExit(1) from None
    if child.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
