"""Validate explicit BLAS parallelism against the compute allocation."""

import os
import platform
from pathlib import Path


def cpu_model():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def threads(config):
    if config.get("core_cpu_model") and config["core_cpu_model"] != cpu_model():
        raise ValueError("core policy does not match this CPU model")
    count = config.get("blas_threads", int(os.environ.get("EXPERIMENT_THREADS", "1")))
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("blas_threads must be a positive integer")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1)))
    if count > allocated:
        raise ValueError("blas_threads exceeds allocated CPUs")
    return count
