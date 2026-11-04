"""Retry a global Slurm job-count rejection at a bounded, courteous interval.

Only the Slurm skill helper contacts the cluster. Connection errors and all
other failures stop immediately; an ambiguous submission is never retried.
"""

import argparse
import csv
import fcntl
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "regression_followups_v1_20261104"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-minutes", type=int, default=60)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--initial-delay", type=int, default=300)
    args = parser.parse_args()
    if args.max_minutes <= 0 or args.interval_seconds < 300 or args.initial_delay < 0:
        parser.error(
            "use a positive budget and a retry interval of at least 300 seconds"
        )
    helper = Path.home() / ".agents/skills/slurm/scripts/slurm_agent.py"
    command = [
        sys.executable,
        str(helper),
        "submit",
        "slurm/campaign_v7.slurm",
        "--config",
        "slurm/figure_v8.env",
        "--cpus",
        "1",
        "--mem",
        "64G",
        "--time",
        "22:45:00",
        "--constraint",
        "icelake",
        "--array",
        "0-1%2",
        "--job-name",
        "rnla-reg-followups-v1",
        "--export",
        "MANIFEST=regression_v1/inputs/followup_candidates.csv",
        "--export",
        f"EXPERIMENT={EXPERIMENT}",
        "--export",
        "BATCH_SIZE=10",
        "--export",
        "WORKER_MEMORY_GB=56",
        "--export",
        "SETUP_BUDGET=900",
        "--json",
    ]
    directory = ROOT / "regression_v1"
    with (directory / "followup_submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = directory / "followup_submission_status.csv"
        if status_path.exists():
            with status_path.open(newline="") as stream:
                previous = list(csv.DictReader(stream))
            if previous and (
                previous[0].get("job_id")
                or previous[0].get("state")
                in ("submitting", "submission_outcome_unknown")
            ):
                raise SystemExit(
                    "a submitted job is already recorded; refusing duplicate submission"
                )
        started = time.monotonic()
        deadline = started + args.max_minutes * 60
        attempts = 0

        def status(state, code="", job_id="", detail=""):
            temporary = status_path.with_suffix(".tmp")
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "state",
                        "job_id",
                        "experiment",
                        "attempts",
                        "exit_code",
                        "utc",
                        "detail",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    dict(
                        state=state,
                        job_id=job_id,
                        experiment=EXPERIMENT,
                        attempts=attempts,
                        exit_code=code,
                        utc=datetime.now(UTC).isoformat(),
                        detail=detail,
                    )
                )
            temporary.replace(status_path)

        def call(extra):
            result = subprocess.run(
                command + extra, cwd=ROOT, text=True, capture_output=True
            )
            with (directory / "followup_submission.log").open("a") as log:
                log.write(
                    f"\n{datetime.now(UTC).isoformat()} "
                    f"helper exit={result.returncode}\n"
                )
                log.write(result.stdout + result.stderr)
            return result

        status("waiting_for_global_job_capacity", detail="MaxJobCount limit reached")
        print(
            "Waiting for Slurm capacity; retry interval 5 minutes, budget 1 hour.",
            flush=True,
        )
        time.sleep(args.initial_delay)
        while time.monotonic() < deadline:
            attempts += 1
            test = call([])
            if test.returncode == 4 and "MaxJobCount limit reached" in test.stdout:
                status(
                    "waiting_for_global_job_capacity",
                    test.returncode,
                    detail="MaxJobCount limit reached",
                )
            elif test.returncode == 2:
                verdict = json.loads(test.stdout)
                if not verdict.get("test_only", {}).get("ok"):
                    status(
                        "stopped",
                        test.returncode,
                        detail="unexpected test-only response",
                    )
                    raise SystemExit(test.returncode)
                status("submitting", detail="test-only validation passed")
                submitted = call(["--yes"])
                if submitted.returncode == 0:
                    data = json.loads(submitted.stdout)
                    job_id = str(data["job_id"])
                    status(
                        "submitted",
                        0,
                        job_id,
                        "20 configurations in two serial batches",
                    )
                    print(
                        f"Submitted array {job_id}: "
                        "20 experiments in two serial batches.",
                        flush=True,
                    )
                    return
                if (
                    submitted.returncode != 4
                    or "MaxJobCount limit reached" not in submitted.stdout
                ):
                    status(
                        "submission_outcome_unknown",
                        submitted.returncode,
                        detail="submission failed; inspect log before any retry",
                    )
                    raise SystemExit(submitted.returncode)
                status(
                    "waiting_for_global_job_capacity",
                    submitted.returncode,
                    detail="capacity filled between validation and submission",
                )
            else:
                status(
                    "stopped",
                    test.returncode,
                    detail="validation failed; inspect log before any retry",
                )
                raise SystemExit(test.returncode or 1)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(args.interval_seconds, remaining))
        status(
            "waiting_budget_exhausted", 4, detail="global capacity remained unavailable"
        )
        raise SystemExit(4)


if __name__ == "__main__":
    main()
