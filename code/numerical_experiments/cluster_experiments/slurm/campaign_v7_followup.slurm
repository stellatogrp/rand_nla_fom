#!/bin/bash
#SBATCH --job-name=rnla-v7-next-stage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
cd "$PROJECT_DIR"
# This job plans only. It never selects from incomplete results or submits jobs.
exec "$PY" -m campaign_v7.advance "${FOLLOWUP_STAGE:?export FOLLOWUP_STAGE}" \
    --manifest "${MANIFEST:?export MANIFEST}" \
    --results-dir "results/${EXPERIMENT:?export EXPERIMENT}" \
    --output "${FOLLOWUP_OUTPUT:?export a new follow-up manifest filename}"
