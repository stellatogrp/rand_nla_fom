"""Verify immutable inputs before running a bounded regression pilot worker."""

import argparse
import hashlib
import json
import signal
import subprocess
import sys
from pathlib import Path

from validation_v7.releases import ROOT, verify


def verified(path):
    metadata = json.loads(path.with_suffix(".seal.json").read_text())
    if metadata["manifest_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("regression manifest changed")
    if (
        metadata["numerical_archive_sha256"]
        != verify(ROOT, "regression_v9")["archive_sha256"]
    ):
        raise ValueError("numerical source archive mismatch")
    for name, expected in metadata["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"regression launcher source changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    verified(args.manifest)
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "releases.regression_v9.experiments.run",
            "--manifest",
            str(args.manifest),
            "--results-dir",
            str(args.results_dir),
            "--task-id",
            str(args.task_id),
            "--data-dir",
            "data",
            "--memory-gb",
            "56",
            "--setup-budget",
            "900",
        ],
        cwd=ROOT,
    )

    def forward(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGUSR1, forward)
    signal.signal(signal.SIGTERM, forward)
    raise SystemExit(child.wait())


if __name__ == "__main__":
    main()
