# Cluster experiments

This folder is a standalone uv project; run commands from this directory.
Use US English in prose, comments, docstrings, CLI help, plot text, and new
identifiers. Preserve external names, APIs, URLs, quotations, and source titles.
Use scripts in `experiments/` and CSV-reading plots in `plots/`. No notebooks.
`studies/` runs these problems in the original studies' format and must keep
their table columns (`tests/test_studies.py`); `studies/_io.py`,
`studies/aggregate.py`, and `plots/studies.py` are snapshots of the original
modules, changed only where their docstrings say.
Generated `data/`, `manifests/`, and `results/` are ignored; never edit results.
Do not change the original numerical studies as part of this setup.

## Cluster (Slurm)

Use the slurm skill helper for cluster interaction, never raw SSH from an agent.
The human opens `ssh della-gpu` and bootstraps the environment. On helper exit 3,
stop connection attempts until the human reopens the session. Poll at least 60 s
apart. Validate submissions with `sbatch --test-only` through the helper.
Submission, resize, and cancellation follow the skill's explicit confirmation rule.

Host: `della-gpu`; discovered account: `bstellato`. Project scratch directory,
explicitly confirmed by the user: `/scratch/gpfs/BSTELLATO/pranav/rand_nla_fom`.
That directory contains this standalone project, not the parent repository.
The SSH login name is `pr5741`; the confirmed directory name is `pranav`.

Della policy (reviewed 2026-10-30):

- CPU experiments belong on CPU compute nodes, even when submitted from della-gpu.
- Omit partition and QoS; let Slurm route jobs. Discover live limits each session.
- One core per task by default. BLAS/OpenMP threads must fit the allocation.
  The user explicitly authorized thread scaling and resident-data pilots;
  benchmark 1/2/4/8 cores before promoting a production BLAS profile. No GPUs.
- Never run these experiments or dataset conversion on public login nodes.
- Download and install dependencies before jobs; compute nodes have no internet.
- Keep active output on group scratch. Do not use /projects for active output.
- Scratch is not backed up. Sync results down and archive completed studies.
- Pilot each workload before scaling; inspect stats for representative task IDs.
  The user authorized separate scheduling lanes: at most eight fast QP tasks,
  one Kaczmarz screening task, two data-preparation/regression tasks, and one
  scaling-profile task. Keep these caps separate; chains sharing a lane must
  use dependencies. Size memory/time from pilots, not assumptions.
- Do not commit results, data, logs, job ledgers, or local discovery caches.
- While jobs are queued or running, the code they will load is frozen: the
  release they use, root `experiments/`, `slurm/common.sh`,
  `slurm/job_array.slurm`, `manifests/`, and `datasets/`. Batch tasks import
  from disk when each task starts. Put changes in a new `releases/` directory
  and run `sync up --dry-run` before every `sync up`.

Sources:
https://researchcomputing.princeton.edu/systems/della
https://researchcomputing.princeton.edu/get-started/dos-and-donts
https://researchcomputing.princeton.edu/get-started/mistakes-avoid
https://researchcomputing.princeton.edu/support/knowledge-base/data-storage

Before committing: `uv run ruff check . && uv run ruff format . && uv run pytest`.
