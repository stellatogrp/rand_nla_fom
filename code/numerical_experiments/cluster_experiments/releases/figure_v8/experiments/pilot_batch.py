"""Shared-data pilots only; record real batch setup separately from each solve."""

import argparse
import gc
import json
import math
import os
import resource
import signal
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from threadpoolctl import threadpool_limits

from . import solvers
from .feature_store import atomic_json
from .io import identity, read_csv, write_csv
from .limits import budgets
from .problems import build
from .run import execute
from .runtime import threads


def key(config):
    return identity(
        {
            k: config[k]
            for k in [
                "family",
                "dataset",
                "matrix_source",
                "orientation",
                "seed",
                "side",
                "kappa",
                "lambda_ratio",
            ]
            if k in config
        }
    )


def worker(args, rows):
    if sys.platform == "linux":
        size = int(args.memory_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (size, size))
    events = read_csv(args.results_dir / "shared_setup.csv")
    last_key, p = None, None
    solvers.install_signals()
    for row in rows:
        config = json.loads(row["config"])
        output = args.results_dir / row["run_id"]
        previous = read_csv(output / "result.csv")
        if previous and previous[0]["status"] != "interrupted":
            continue
        atomic_json(
            args.results_dir / "active.json",
            dict(
                run_id=row["run_id"],
                started=perf_counter(),
                deadline_seconds=config.get("time_limit", 1800.0) + args.setup_budget,
            ),
        )
        current = key(config)
        with threadpool_limits(limits=threads(config)):
            if current != last_key:
                p = None  # one-entry cache: do not retain two full resident designs
                gc.collect()
                tick = perf_counter()
                p = build(config, args.data_dir)
                context = dict(
                    id=(
                        f"{os.environ.get('SLURM_JOB_ID', 'local')}:"
                        f"{current}:{len(events)}"
                    ),
                    seconds=perf_counter() - tick,
                )
                events.append(dict(**context, problem_key=current))
                write_csv(args.results_dir / "shared_setup.csv", events)
                last_key = current
            execute(config, output, args.data_dir, problem=p, shared_context=context)
        if solvers.INTERRUPTED:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--memory-gb", type=float, default=56.0)
    parser.add_argument("--setup-budget", type=float, default=600.0)
    parser.add_argument("--wall-seconds", type=float, default=7200.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("shared-data pilots require a compute allocation")
    if any(
        not math.isfinite(v) or v <= 0
        for v in [args.memory_gb, args.setup_budget, args.wall_seconds]
    ):
        parser.error("pilot budgets must be finite and positive")
    rows = read_csv(args.manifest)
    if not rows:
        parser.error("empty pilot manifest")
    for row in rows:
        config = json.loads(row["config"])
        budgets(config)
        threads(config)
        if identity(config) != row["run_id"]:
            parser.error("manifest hash mismatch")
        if config.get("setup_policy") != "shared_pilot" or config.get("phase") not in (
            "pilot",
            "smoke",
            "profile",
        ):
            parser.error("persistent workers accept only explicit shared-data pilots")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args, rows)
        return
    (args.results_dir / "active.json").unlink(missing_ok=True)
    started = perf_counter()
    child = subprocess.Popen(
        [sys.executable, "-m", f"{__package__}.pilot_batch", *sys.argv[1:], "--worker"]
    )

    def forward(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGUSR1, forward)
    signal.signal(signal.SIGTERM, forward)
    reason = "completed"
    while child.poll() is None:
        active_path = args.results_dir / "active.json"
        active = json.loads(active_path.read_text()) if active_path.exists() else {}
        if perf_counter() - started > args.wall_seconds or (
            active and perf_counter() - active["started"] > active["deadline_seconds"]
        ):
            reason = "worker_timeout"
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            if active:
                row = next(r for r in rows if r["run_id"] == active["run_id"])
                write_csv(
                    args.results_dir / row["run_id"] / "result.csv",
                    [
                        dict(
                            config=row["config"],
                            run_id=row["run_id"],
                            status=reason,
                            converged=False,
                            primary_eligible=False,
                            setup_policy="shared_pilot",
                        )
                    ],
                )
            break
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    terminal = 0
    for row in rows:
        observed = read_csv(args.results_dir / row["run_id"] / "result.csv")
        terminal += bool(observed and observed[0]["status"] != "interrupted")
    if reason == "completed" and child.returncode:
        reason = "worker_failed"
    elif reason == "completed" and terminal != len(rows):
        reason = "incomplete"
    accounting = read_csv(args.results_dir / "batch_accounting.csv")
    accounting.append(
        dict(
            batch_wall_seconds=perf_counter() - started,
            status=reason,
            worker_returncode=child.returncode,
            timing_scope="shared_data_pilot_batch",
            primary_eligible=False,
            terminal_tasks=terminal,
            requested_tasks=len(rows),
        )
    )
    write_csv(args.results_dir / "batch_accounting.csv", accounting)
    if child.returncode or reason != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
