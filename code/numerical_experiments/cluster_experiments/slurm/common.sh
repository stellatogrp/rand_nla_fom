# Shared environment for every job script. Sourced from $SLURM_SUBMIT_DIR/slurm/common.sh.
# Requires slurm/cluster.env next to it.
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SLURM_DIR/cluster.env"

PROJECT_DIR="${REMOTE_PROJECT_DIR}"   # $USER expands here, on the cluster
cd "$PROJECT_DIR"

# Site modules from cluster.env (MODULES=a/1.0,b/2.0), e.g. a licensed solver. `module` is a shell
# function that non-login shells may lack, and its init scripts are not `set -u` clean.
if [ -n "${MODULES:-}" ]; then
    set +u
    type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null || source /etc/profile.d/lmod.sh 2>/dev/null || true
    IFS=, read -ra _mods <<< "$MODULES"
    module load "${_mods[@]}"
    set -u
fi

# The same entrypoint as on your laptop: the project venv created by env_setup.sh.
PY="$PROJECT_DIR/.venv/bin/python"

# Serial by default. Only measured BLAS profiles may set EXPERIMENT_THREADS;
# every request must fit the allocation and is also enforced by threadpoolctl.
export EXPERIMENT_THREADS="${EXPERIMENT_THREADS:-1}"
if ! [[ "$EXPERIMENT_THREADS" =~ ^[1-9][0-9]*$ ]] || \
    [ "$EXPERIMENT_THREADS" -gt "${SLURM_CPUS_PER_TASK:-1}" ]; then
    echo "EXPERIMENT_THREADS must be positive and fit SLURM_CPUS_PER_TASK" >&2
    exit 2
fi
export OMP_NUM_THREADS="$EXPERIMENT_THREADS"
export OPENBLAS_NUM_THREADS="$EXPERIMENT_THREADS"
export MKL_NUM_THREADS="$EXPERIMENT_THREADS"

mkdir -p "$PROJECT_DIR/${RESULTS_DIR:-results}" "$PROJECT_DIR/slurm/logs"
echo "[common.sh] host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} python=$PY"

export VECLIB_MAXIMUM_THREADS="$EXPERIMENT_THREADS"
export NUMEXPR_NUM_THREADS="$EXPERIMENT_THREADS"
export MKL_DYNAMIC=FALSE

# New releases run from their own source tree; existing jobs keep the root code.
if [ -n "${EXPERIMENT_RELEASE:-}" ]; then
    if ! [[ "$EXPERIMENT_RELEASE" =~ ^[A-Za-z0-9_]+$ ]]; then exit 2; fi
    cd "$PROJECT_DIR/releases/$EXPERIMENT_RELEASE"
fi
