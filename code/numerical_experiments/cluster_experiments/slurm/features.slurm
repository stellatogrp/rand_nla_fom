#!/bin/bash
# Pilot small feature maps first, then size time/memory from completed stats.
# Superseded by features_v3.slurm (durable chunks, SIGUSR1/SIGTERM handling).
# No --signal here: this root code has no SIGUSR1 handler, and with `exec` a
# warning signal would terminate the job two minutes early.
#SBATCH --job-name=rnla-features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
DATASET="${DATASET:?export DATASET=cod-rna (one dataset per preparation job)}"
FEATURE_COUNT="${FEATURE_COUNT:?export the chosen number of random features}"
OUTPUT_SPEC="${OUTPUT_SPEC:?export OUTPUT_SPEC=manifests/cod_rna_sources.json}"
args=(--datasets "$DATASET" --data-dir data --features "$FEATURE_COUNT"
      --gamma "${RBF_GAMMA:-1}" --seed "${FEATURE_SEED:-0}" --output "$OUTPUT_SPEC"
      --positive-class "${POSITIVE_CLASS:-1}" --preprocessing "${INPUT_SCALING:-row_l2}"
      --dtype "${FEATURE_DTYPE:-float32}")
if [ -n "${PREPARATION_REPORT_DIR:-}" ]; then
    args+=(--report-dir "$PREPARATION_REPORT_DIR")
fi
if [ -n "${MAX_ROWS:-}" ]; then args+=(--max-rows "$MAX_ROWS"); fi
exec "$PY" -m experiments.benchmark_data prepare "${args[@]}"
