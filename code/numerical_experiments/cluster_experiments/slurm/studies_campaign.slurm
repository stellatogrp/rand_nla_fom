#!/bin/bash
# One task of a campaign of original-format studies (studies/campaign.py).
# Submit through the slurm skill, one array per manifest, exporting
#   MANIFEST  <campaign>/inputs/<lane>.calibrate.csv, <lane>.run.partNNN.csv,
#             or <lane>.collect.csv
# The array covers the manifest's rows (task 0 to N-1); every task checks the
# campaign seal first. Lanes are separate arrays: chain each lane's run parts
# afterany on its calibrate array and its collect afterany on its run parts.
# Size --mem and --time per lane from a pilot; these are provisional.
#SBATCH --job-name=rnla-studies
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
MANIFEST="${MANIFEST:?export MANIFEST=<campaign>/inputs/<lane>.<stage>.csv}"
TASK="${SLURM_ARRAY_TASK_ID:?submit as an array, even for one task}"
if ! [ -f "$MANIFEST" ]; then echo "missing $MANIFEST" >&2; exit 2; fi
export EXPERIMENT_SOURCE_REVISION="${EXPERIMENT_SOURCE_REVISION:-studies-campaign}"
"$PY" -m studies.campaign task --manifest "$MANIFEST" --task-id "$TASK" &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$pid" 2>/dev/null || true' TERM
wait "$pid" || {
    status=$?
    if kill -0 "$pid" 2>/dev/null; then wait "$pid"; else exit "$status"; fi
}
