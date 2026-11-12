"""The cluster problems run as the original studies of ``numerical_experiments``.

Each study draws instances, computes a ground truth and the paper's
parameters, and runs every configuration of the original studies (the
classical baseline, the near-exact baseline, the ``sigma`` sweep with
``sigma*``, and the stride sweep with ``s*``) to the original stopping rule.
It writes the original tables, ``parameters.csv``, ``results.csv``,
``histories.csv``, ``curves.csv``, and ``summary.csv``, with the original
columns.  The problems, matrix sources, and inner solvers are those of the
sealed numerical release named by ``ENGINE``.  See ``studies/README.md``.
"""

import os

# One BLAS thread per process, as in the original studies; set before NumPy
# loads, including in the spawned workers of the local pool.  On Slurm,
# slurm/common.sh exports the same variables from the allocation.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

# The sealed numerical release whose problems and inner solvers the studies use.
ENGINE = "regression_v9"
