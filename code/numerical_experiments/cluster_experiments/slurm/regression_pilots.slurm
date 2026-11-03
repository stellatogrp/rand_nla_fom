#!/bin/bash
# Run after all full feature maps complete; inspect stats before tuning arrays.
#SBATCH --job-name=rnla-regression-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:15:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
datasets=(cod-rna cifar10 realsim smallnorb rcv1)
dataset="${datasets[${SLURM_ARRAY_TASK_ID:?submit an array from 0 to 4}]}"
source_spec="manifests/full_features_20261030_${dataset}.json"
manifest="manifests/resident_regression_pilots_speed_v2_${dataset}.csv"
"$PY" -m experiments.manifest pilot --dataset-spec "$source_spec" \
    --families lasso logistic --inner-initializations warm --seeds 0 \
    --dense-storage resident_dense --max-resident-gb 32 \
    --blas-threads "$EXPERIMENT_THREADS" \
    --time-limit 60 --inner-time-limit 60 --max-inner 1000 --max-outer 1000 \
    --output "$manifest" --shard-size 10000
exec "$PY" -m experiments.batch --manifest "$manifest" --batch-id 0 --batch-size 11 \
    --results-dir "$RESULTS_DIR/resident_regression_pilots_speed_v2" \
    --memory-gb 56
