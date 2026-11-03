#!/bin/bash
# Batch serial configurations to respect scheduler submission limits.
#SBATCH --job-name=rnla-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=22:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
MANIFEST="${MANIFEST:?export MANIFEST}"
EXPERIMENT="${EXPERIMENT:?export a unique EXPERIMENT name}"
BATCH_SIZE="${BATCH_SIZE:?export BATCH_SIZE}"
"$PY" -m experiments.batch --manifest "$MANIFEST" \
    --batch-id "${SLURM_ARRAY_TASK_ID:?submit as an array}" --batch-size "$BATCH_SIZE" \
    --results-dir "$RESULTS_DIR/$EXPERIMENT" \
    --memory-gb "${WORKER_MEMORY_GB:-12}" &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$pid" 2>/dev/null || true' TERM
wait "$pid" || {
    status=$?
    if kill -0 "$pid" 2>/dev/null; then wait "$pid"; else exit "$status"; fi
}
