#!/bin/bash
# One-time environment bootstrap ON THE CLUSTER (login node is fine: this is setup, not compute).
# Run it yourself from a laptop terminal after `sync up`, replacing the path with your
# REMOTE_PROJECT_DIR (with $USER spelled out or left for the remote shell to expand):
#   ssh <alias> 'cd /scratch/.../$USER/myproject && bash slurm/env_setup.sh'
# Creates the uv venv inside the project (on scratch) with the uv cache and uv-managed Python
# installs on scratch too, so nothing heavy lands in your small home quota.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/cluster.env
PROJECT_DIR="${REMOTE_PROJECT_DIR}"

# Site modules from cluster.env (MODULES=a/1.0,b/2.0), e.g. a licensed solver. `module` is a shell
# function that non-login shells may lack, and its init scripts are not `set -u` clean.
if [ -n "${MODULES:-}" ]; then
    set +u
    type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null || source /etc/profile.d/lmod.sh 2>/dev/null || true
    IFS=, read -ra _mods <<< "$MODULES"
    module load "${_mods[@]}"
    set -u
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$PROJECT_DIR")/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$(dirname "$PROJECT_DIR")/.uv-python}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" slurm/logs results data

# Rerunnable: keep an existing venv and let `uv sync` bring it up to date with pyproject.toml.
[ -x .venv/bin/python ] || uv venv --python "${PYTHON_VERSION:-3.12}" .venv
uv sync --frozen
.venv/bin/python -c "import sys; print('venv OK', sys.version.split()[0])"
if [ -n "${MODULES:-}" ]; then set +u; module list; set -u; fi
