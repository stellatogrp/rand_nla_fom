#!/bin/bash
#SBATCH --job-name=rnla-main-figures
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
cd "$PROJECT_DIR"
exec "$PY" -m plots.main_summary --results-dir "results/${EXPERIMENT:?}" \
    --manifest "${MANIFEST:?}" --output "main_figures/fresh_v8"
