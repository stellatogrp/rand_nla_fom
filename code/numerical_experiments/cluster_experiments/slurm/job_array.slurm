#!/bin/bash
# Pilot array=0 before a larger array; submit through the slurm skill.
#SBATCH --job-name=rnla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
MANIFEST="${MANIFEST:?export MANIFEST=manifests/pilot.csv}"
EXPERIMENT="${EXPERIMENT:?export a unique EXPERIMENT name}"
TASK="${SLURM_ARRAY_TASK_ID:?submit as an array, even for one pilot}"
export EXPERIMENT_SOURCE_REVISION="${EXPERIMENT_SOURCE_REVISION:-unknown}"
"$PY" -m experiments.run --manifest "$MANIFEST" --task-id "$TASK" \
    --results-dir "$RESULTS_DIR/$EXPERIMENT" --data-dir data \
    --memory-gb "${WORKER_MEMORY_GB:-12}" &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$pid" 2>/dev/null || true' TERM
wait "$pid" || {
    status=$?
    if kill -0 "$pid" 2>/dev/null; then wait "$pid"; else exit "$status"; fi
}
