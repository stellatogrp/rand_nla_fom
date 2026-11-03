"""Verify campaign inputs and run bounded, independent v7 workers serially."""

import argparse
import hashlib
import json
import signal
import subprocess
import sys
from pathlib import Path

from releases.accounting_v7.experiments.batch import task_ids
from releases.accounting_v7.experiments.io import read_csv
from validation_v7.releases import ROOT, verify


def campaign_sources():
    return [
        *sorted((ROOT / "campaign_v7").glob("*.py")),
        *sorted((ROOT / "campaign_v7/manifests").glob("*.csv")),
        ROOT / "campaign_v7/plan.json",
        ROOT / "slurm/campaign_v7.slurm",
    ]


def seal():
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in campaign_sources()
    }


def verify_campaign():
    expected = json.loads((ROOT / "campaign_v7/source_seal.json").read_text())
    if expected != seal():
        raise ValueError("campaign sources/inputs changed after sealing")
    verify(ROOT, "accounting_v7")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-id", type=int)
    group.add_argument("--batch-id", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-gb", type=float, default=12)
    parser.add_argument("--setup-budget", type=float, default=600)
    args = parser.parse_args()
    verify_campaign()
    protected = json.loads((ROOT / "campaign_v7/source_seal.json").read_text())
    relative = str(args.manifest.resolve().relative_to(ROOT))
    if relative not in protected:
        metadata = json.loads(args.manifest.with_suffix(".seal.json").read_text())
        if (
            metadata["manifest_sha256"]
            != hashlib.sha256(args.manifest.read_bytes()).hexdigest()
            or metadata["campaign_seal_sha256"]
            != hashlib.sha256(
                (ROOT / "campaign_v7/source_seal.json").read_bytes()
            ).hexdigest()
            or metadata["numerical_archive_sha256"]
            != verify(ROOT, "accounting_v7")["archive_sha256"]
        ):
            raise ValueError("follow-up manifest or source seal changed")
    rows = read_csv(args.manifest)
    ids = (
        [args.task_id]
        if args.task_id is not None
        else task_ids(len(rows), args.batch_id, args.batch_size)
    )
    child, stopped = None, False

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
        if not 0 <= task < len(rows):
            raise ValueError("task ID outside manifest")
        print(f"task={task} run_id={rows[task]['run_id']}", flush=True)
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "releases.accounting_v7.experiments.run",
                "--manifest",
                str(args.manifest),
                "--task-id",
                str(task),
                "--results-dir",
                str(args.results_dir),
                "--data-dir",
                "data",
                "--memory-gb",
                str(args.memory_gb),
                "--setup-budget",
                str(args.setup_budget),
            ],
            cwd=ROOT,
        )
        failed |= child.wait() != 0
    if stopped or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
