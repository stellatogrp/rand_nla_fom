"""Wait for the plotting job, then collect its figures and scientific CSVs.

Every cluster operation uses the Slurm skill helper. Any nonzero helper exit
stops collection immediately, including exit 3; there are no connection retries.
"""

import argparse
import csv
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--max-minutes", type=int, default=120)
    parser.add_argument(
        "--helper",
        type=Path,
        default=Path.home() / ".agents/skills/slurm/scripts/slurm_agent.py",
    )
    args = parser.parse_args()
    if not args.job_id.isdigit() or args.max_minutes < 1:
        parser.error("job ID must be numeric and the waiting budget positive")
    command = [sys.executable, str(args.helper)]
    steps = [
        [
            "watch",
            args.job_id,
            "--host",
            "della-gpu",
            "--max-minutes",
            str(args.max_minutes),
        ],
        ["sync", "down", "--config", "slurm/figure_v8_results.env"],
        ["sync", "down", "--config", "slurm/figure_v8_outputs.env"],
    ]
    destination = ROOT / "main_figures"

    def status(state, code):
        path = destination / "collection_status.csv"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["job_id", "state", "exit_code", "utc"]
            )
            writer.writeheader()
            writer.writerow(
                dict(
                    job_id=args.job_id,
                    state=state,
                    exit_code=code,
                    utc=datetime.now(UTC).isoformat(),
                )
            )
        temporary.replace(path)

    status("waiting_for_plot_job", "")
    print(
        f"Waiting for plotting job {args.job_id}; "
        f"log: {destination / 'collection.log'}",
        flush=True,
    )
    with (destination / "collection.log").open("a", buffering=1) as log:
        for step in steps:
            result = subprocess.run(
                command + step, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            if result.returncode:
                status("stopped", result.returncode)
                print(
                    f"Collection stopped with helper exit {result.returncode}; "
                    "see collection.log.",
                    flush=True,
                )
                if result.returncode == 3:
                    print(
                        "The SSH master is unavailable. "
                        "A human must reopen ssh della-gpu before any retry.",
                        flush=True,
                    )
                raise SystemExit(result.returncode)
    status("downloaded", 0)
    print(
        f"Scientific results and figures downloaded to {destination / 'fresh_v8'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
