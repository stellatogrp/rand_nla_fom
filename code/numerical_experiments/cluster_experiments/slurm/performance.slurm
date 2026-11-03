#!/bin/bash
# Pilot one core, inspect stats, then benchmark 2/4/8 with matching requests.
#SBATCH --job-name=rnla-performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%j.out

export EXPERIMENT_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
args=(--kind "${PROFILE_KIND:-qp}" --threads "$EXPERIMENT_THREADS"
      --output "${PROFILE_OUTPUT:?export PROFILE_OUTPUT}"
      --time-limit "${PROFILE_SECONDS:-600}" --memory-gb "${WORKER_MEMORY_GB:-12}"
      --repeats "${PROFILE_REPEATS:-2}" --side "${PROFILE_SIDE:-100}"
      --operation "${PROFILE_OPERATION:-solve}" --inner "${PROFILE_INNER:-nystrom_pcg}"
      --family "${PROFILE_FAMILY:-logistic}" --storage "${PROFILE_STORAGE:-resident_dense}")
if [ -n "${PROFILE_SOURCE_SPEC:-}" ]; then args+=(--source-spec "$PROFILE_SOURCE_SPEC"); fi
exec "$PY" -m experiments.performance "${args[@]}"
