#!/bin/bash
# Durable chunks plus RAM assembly; one CPU, measured memory payload bounds.
#SBATCH --job-name=rnla-features-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%j.out

export EXPERIMENT_RELEASE=accounting_v3
source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
args=(--datasets "${DATASET:?export DATASET}" --data-dir data
      --features "${FEATURE_COUNT:?export FEATURE_COUNT}"
      --output "${OUTPUT_SPEC:?export OUTPUT_SPEC}"
      --report-dir "${PREPARATION_REPORT_DIR:?export PREPARATION_REPORT_DIR}"
      --max-assembly-gb "${ASSEMBLY_GB:-32}")
if [ -n "${SLURM_TMPDIR:-}" ]; then args+=(--work-dir "$SLURM_TMPDIR"); fi
if [ -n "${MAX_ROWS:-}" ]; then args+=(--max-rows "$MAX_ROWS"); fi
exec "$PY" -m experiments.benchmark_data prepare "${args[@]}"
