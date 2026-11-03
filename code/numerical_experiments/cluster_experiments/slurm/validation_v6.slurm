#!/bin/bash
#SBATCH --job-name=rnla-v6-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
cd "$PROJECT_DIR"
export EXPERIMENT_SOURCE_REVISION="sealed-accounting-v6"
"$PY" -m validation.task \
    --manifest "${MANIFEST:?export a sealed validation manifest}" \
    --task-id "${SLURM_ARRAY_TASK_ID:?submit as an array}" \
    --results-dir "results/${EXPERIMENT:?export a unique experiment name}" &
pid=$!
trap 'kill -TERM "$pid" 2>/dev/null || true' USR1 TERM
wait "$pid"
