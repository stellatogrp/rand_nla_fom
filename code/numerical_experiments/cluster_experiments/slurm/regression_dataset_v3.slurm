#!/bin/bash
# Explicit warm-data pilot. Main sweeps use independent experiments.run workers.
#SBATCH --job-name=rnla-shared-dataset-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:15:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%j.out

export EXPERIMENT_RELEASE=accounting_v3
source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
dataset="${DATASET:?export DATASET}"
case "$dataset" in cod-rna|cifar10|realsim|smallnorb|rcv1) ;; *) exit 2 ;; esac
experiment="shared_dataset_pilots_v3_${dataset}"
manifest="manifests/${experiment}.csv"
"$PY" -m experiments.manifest pilot \
    --dataset-spec "manifests/full_features_20261030_${dataset}.json" \
    --families lasso logistic --inner-initializations warm --seeds 0 \
    --dense-storage resident_dense --max-resident-gb 32 --setup-policy shared_pilot \
    --time-limit 60 --inner-time-limit 60 --max-inner 1000 --max-outer 1000 \
    --output "$manifest" --shard-size 10000
exec "$PY" -m experiments.pilot_batch --manifest "$manifest" \
    --results-dir "$RESULTS_DIR/$experiment" --memory-gb 56 --wall-seconds 7200
