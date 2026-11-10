#!/bin/bash
# One job of a one-node timing benchmark (bench/one_node.py): the planned runs
# of one or more problems, timed in one process, interleaved instance by
# instance. Submit through the slurm skill, pinned to one node
# (--sbatch-arg=--nodelist=<node>), one job at a time, exporting
#   PLAN                  <benchmark>/plan.json (sealed: <benchmark>/seal.json)
#   RESULTS               results/<benchmark>
#   PROBLEMS              problem slugs separated by colons, e.g. qp_side40:qp_side64
# and, for a pilot, optionally
#   INSTANCES             time the first N instances only
#   NEAR_EXACT_INSTANCES  time the near-exact baseline on at most K of them
# Runs are appended as they finish and a resubmission skips them, so a job cut
# short by its time limit loses only the run in progress.
#SBATCH --job-name=rnla-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
PLAN="${PLAN:?export PLAN=<benchmark>/plan.json}"
RESULTS="${RESULTS:?export RESULTS=results/<benchmark>}"
PROBLEMS="${PROBLEMS:?export PROBLEMS=<problem slug>[:<problem slug>...]}"
if ! [ -f "$PLAN" ]; then echo "missing $PLAN" >&2; exit 2; fi
args=()
IFS=: read -ra slugs <<< "$PROBLEMS"
for slug in "${slugs[@]}"; do args+=(--problem "$slug"); done
if [ -n "${INSTANCES:-}" ]; then args+=(--instances "$INSTANCES"); fi
if [ -n "${NEAR_EXACT_INSTANCES:-}" ]; then
    args+=(--near-exact-instances "$NEAR_EXACT_INSTANCES")
fi
export EXPERIMENT_SOURCE_REVISION="${EXPERIMENT_SOURCE_REVISION:-bench-one-node}"
"$PY" -m bench.one_node run --plan "$PLAN" --results-dir "$RESULTS" "${args[@]}" &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$pid" 2>/dev/null || true' TERM
wait "$pid" || {
    status=$?
    if kill -0 "$pid" 2>/dev/null; then wait "$pid"; else exit "$status"; fi
}
