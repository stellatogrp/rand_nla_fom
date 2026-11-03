"""Run a bounded, disjoint slice of a manifest in one Slurm array task."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .io import read_csv
from .limits import budgets


def task_ids(length, batch_id, batch_size):
    if length < 1 or batch_id < 0 or batch_size < 1:
        raise ValueError("manifest must be nonempty; batch ID >= 0; batch size >= 1")
    start = batch_id * batch_size
    if start >= length:
        raise ValueError("batch ID outside manifest")
    return range(start, min(start + batch_size, length))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-id", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--memory-gb", type=float, default=12.0)
    parser.add_argument("--setup-budget", type=float, default=600.0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("batches require a Slurm compute allocation")
    rows = read_csv(args.manifest)
    ids = task_ids(len(rows), args.batch_id, args.batch_size)
    for task in ids:
        budgets(json.loads(rows[task]["config"]))
    child = None
    stopped = False

    def forward(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGUSR1, forward)
    signal.signal(signal.SIGTERM, forward)
    failed = False
    for task in ids:
        if stopped:
            break
        print(
            f"batch={args.batch_id} task={task} run_id={rows[task]['run_id']}",
            flush=True,
        )
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.run",
                "--manifest",
                str(args.manifest),
                "--task-id",
                str(task),
                "--results-dir",
                str(args.results_dir),
                "--memory-gb",
                str(args.memory_gb),
                "--setup-budget",
                str(args.setup_budget),
            ]
        )
        code = child.wait()
        # A failed configuration must not suppress later configurations. Its
        # parent writes a failure CSV; unexpected parent deaths remain missing.
        failed |= code != 0
    if stopped or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
