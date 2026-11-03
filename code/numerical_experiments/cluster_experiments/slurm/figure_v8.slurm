#!/bin/bash
#SBATCH --job-name=rnla-figures-v8
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
export EXPERIMENT_SOURCE_REVISION=sealed-figure-v8
"$PY" -m figure_campaign.runner --manifest "${MANIFEST:?}" \
    --results-dir "results/${EXPERIMENT:?}" --task-id "${SLURM_ARRAY_TASK_ID:?}" &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$pid" 2>/dev/null || true' TERM
wait "$pid" || {
    status=$?
    if kill -0 "$pid" 2>/dev/null; then wait "$pid"; else exit "$status"; fi
}
