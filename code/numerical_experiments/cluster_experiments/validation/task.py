"""Verify frozen sources/manifest, run a bounded worker, and audit its output."""

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path

from releases.accounting_v6.experiments.io import write_csv

from .audit import audit
from .releases import ROOT, digest, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    seal = verify(ROOT, "accounting_v6")
    manifest_seal = json.loads(args.manifest.with_suffix(".seal.json").read_text())
    if (
        manifest_seal["manifest_sha256"] != digest(args.manifest.read_bytes())
        or manifest_seal["archive_sha256"] != seal["archive_sha256"]
    ):
        raise ValueError("manifest or numerical release differs from the sealed pilot")
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "releases.accounting_v6.experiments.run",
            "--manifest",
            str(args.manifest),
            "--task-id",
            str(args.task_id),
            "--results-dir",
            str(args.results_dir),
            "--data-dir",
            "data",
            "--memory-gb",
            "6",
            "--setup-budget",
            "120",
        ],
        cwd=ROOT,
    )

    def forward(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGUSR1, forward)
    signal.signal(signal.SIGTERM, forward)
    if child.wait():
        raise RuntimeError("pilot worker failed; inspect its recorded exception")
    checks = audit(args.manifest, args.results_dir, args.task_id)
    write_csv(args.results_dir / "audits" / f"task_{args.task_id}.csv", checks)
    if not all(c["passed"] for c in checks):
        raise RuntimeError(f"pilot accounting audit failed: {checks}")


if __name__ == "__main__":
    main()
