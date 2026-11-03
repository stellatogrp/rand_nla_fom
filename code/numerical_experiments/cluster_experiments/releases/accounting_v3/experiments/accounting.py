"""Measured attempt journals; unknown hard-kill costs remain explicitly unknown."""

import fcntl
import json
import os
import time
from pathlib import Path
from time import perf_counter

from .feature_store import atomic_json


class Journal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.directory / "lock", "a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.directory / "attempts.json"
        self.rows = json.loads(self.path.read_text()) if self.path.exists() else []
        for row in self.rows:
            if row["status"] == "running":
                row["status"] = "unclean_stop"
        self.started = perf_counter()
        self.current = dict(
            id=str(time.time_ns()),
            status="running",
            elapsed_seconds=0.0,
            job_id=os.environ.get("SLURM_JOB_ID"),
        )
        self.rows.append(self.current)
        self.checkpoint()

    def checkpoint(self, status="running", **fields):
        self.current.update(
            fields, status=status, elapsed_seconds=perf_counter() - self.started
        )
        atomic_json(self.path, self.rows)

    @property
    def known_seconds(self):
        return sum(row["elapsed_seconds"] for row in self.rows)

    @property
    def complete(self):
        return all(row["status"] != "unclean_stop" for row in self.rows)

    def close(self):
        self.lock.close()
