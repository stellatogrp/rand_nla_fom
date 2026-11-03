#!/bin/bash
# One dataset per job. Dependencies belong only to that dataset's lane.
#SBATCH --job-name=rnla-dataset-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:15:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%j.out

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
dataset="${DATASET:?export one completed DATASET}"
case "$dataset" in cod-rna|cifar10|realsim|smallnorb|rcv1) ;; *) exit 2 ;; esac
source_spec="${SOURCE_SPEC:-manifests/full_features_20261030_${dataset}.json}"
experiment="${EXPERIMENT:-resident_dataset_pilots_v2}"
manifest="manifests/${experiment}/${dataset}.csv"
"$PY" -m experiments.manifest pilot --dataset-spec "$source_spec" \
    --families lasso logistic --inner-initializations warm --seeds 0 \
    --dense-storage resident_dense --max-resident-gb 32 \
    --blas-threads "$EXPERIMENT_THREADS" \
    --time-limit 60 --inner-time-limit 60 --max-inner 1000 --max-outer 1000 \
    --output "$manifest" --shard-size 10000
exec "$PY" -m experiments.batch --manifest "$manifest" --batch-id 0 --batch-size 11 \
    --results-dir "$RESULTS_DIR/$experiment" --memory-gb 56
