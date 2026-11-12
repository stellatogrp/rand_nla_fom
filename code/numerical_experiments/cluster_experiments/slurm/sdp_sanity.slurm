#!/bin/bash
# One study of the SDP sanity experiments (sdp/run.py): one line of a study
# list. Submit through the slurm skill as an array over the lines (task i runs
# line i + 1), exporting
#   STUDIES   <campaign>/studies.txt   the arguments of sdp.run, one study per line
#   RESULTS   results/<campaign>
# A study appends each run as it finishes, so a resubmitted task skips the
# runs already in its runs.csv.
#SBATCH --job-name=rnla-sdp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
STUDIES="${STUDIES:?export STUDIES=<campaign>/studies.txt}"
RESULTS="${RESULTS:?export RESULTS=results/<campaign>}"
TASK="${SLURM_ARRAY_TASK_ID:?submit as an array, even for one task}"
if ! [ -f "$STUDIES" ]; then echo "missing $STUDIES" >&2; exit 2; fi
line="$(sed -n "$((TASK + 1))p" "$STUDIES")"
if [ -z "$line" ]; then echo "no study $TASK in $STUDIES" >&2; exit 2; fi
read -ra arguments <<< "$line"
"$PY" -m sdp.run --results-dir "$RESULTS" "${arguments[@]}"
