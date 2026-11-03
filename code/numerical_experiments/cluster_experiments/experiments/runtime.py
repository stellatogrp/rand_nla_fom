"""Validate explicit BLAS parallelism against the compute allocation."""

import os


def threads(config):
    count = config.get("blas_threads", int(os.environ.get("EXPERIMENT_THREADS", "1")))
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("blas_threads must be a positive integer")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1)))
    if count > allocated:
        raise ValueError("blas_threads exceeds allocated CPUs")
    return count
