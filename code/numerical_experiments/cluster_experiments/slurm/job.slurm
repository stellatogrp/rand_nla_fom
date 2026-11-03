#!/bin/bash
# Single representative task; review resource requests after stats.
# The default covers the default 1,800 s solve budget, the 600 s setup
# allowance, and shutdown; pass a smaller --time with a smaller --time-limit.
#SBATCH --job-name=rnla-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%j.out

export SLURM_ARRAY_TASK_ID="${TASK_ID:-0}"
source "${SLURM_SUBMIT_DIR}/slurm/job_array.slurm"
