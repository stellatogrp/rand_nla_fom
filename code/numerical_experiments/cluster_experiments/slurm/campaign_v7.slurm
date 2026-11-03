#!/bin/bash
#SBATCH --job-name=rnla-campaign-v7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
cd "$PROJECT_DIR"
export EXPERIMENT_SOURCE_REVISION=sealed-accounting-v7-campaign
args=(--manifest "${MANIFEST:?export MANIFEST}" --results-dir "results/${EXPERIMENT:?export EXPERIMENT}"
      --memory-gb "${WORKER_MEMORY_GB:-12}" --setup-budget "${SETUP_BUDGET:-600}")
if [ -n "${PILOT_TASK:-}" ]; then
    args+=(--task-id "$PILOT_TASK")
else
    args+=(--batch-id "${SLURM_ARRAY_TASK_ID:?submit as an array}" --batch-size "${BATCH_SIZE:-8}")
fi
"$PY" -m campaign_v7.runner "${args[@]}" &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$pid" 2>/dev/null || true' TERM
wait "$pid" || {
    status=$?
    if kill -0 "$pid" 2>/dev/null; then wait "$pid"; else exit "$status"; fi
}
