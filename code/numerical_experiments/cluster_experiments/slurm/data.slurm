#!/bin/bash
# Offline conversion is computation; downloads must already exist on scratch.
# No --signal: this code has no SIGUSR1 handler, and with `exec` the warning
# would terminate conversion early.
#SBATCH --job-name=rnla-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
if [ -n "${DATASET_SPEC:-}" ]; then
    exec "$PY" -m experiments.matrices --dataset-spec "$DATASET_SPEC" \
        --data-dir data --output "$RESULTS_DIR/matrix_validation.csv"
fi
read -r -a names <<< "${DATASETS:-GL7d12 GL7d16 lp_osa_30}"
exec "$PY" -m experiments.data convert --datasets "${names[@]}" --data-dir data
